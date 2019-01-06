optim = require 'optim'
opt.cls_weight = 0.5
opt.data = '/var/scratch/wshang/celebA/'
data = {}
data.train = torch.load(opt.data .. '/trainData.t7')
data.train_attr = data.train.train_attributes
data.train_im = data.train.train_images:mul(2):add(-1)
data.test = torch.load(opt.data .. '/testData.t7')
data.test_attr = data.test.test_attributes
data.test_im = data.test.test_images:mul(2):add(-1)
data.val = torch.load(opt.data .. '/valData.t7')
data.val_attr = data.val.val_attributes
data.val_im = data.val.val_images:mul(2):add(-1)
opt.scales = {64}
opt.latentDims = {4}
opt.latentFilters = {opt.nf}
opt.preprocess_train = image_utils.HorizontalFlip(0.5)
opt.preprocess_test = image_utils.HorizontalFlip(1)
-- load models
local disc, gen = table.unpack(models)
local criterion = criterions
local optimStateG, optimStateD

-- reload if previous checkpoint exists
-- otherwise initialize optimStates
if opt.retrain ~= 'none' and opt.optimState ~= 'none' then
  local models_resume = torch.load(opt.retrain)
  local states_resume = torch.load(opt.optimState)
  disc, gen = nil, nil
  disc, gen = table.unpack(models_resume)
  optimStateG, optimStateD = table.unpack(states_resume)
  collectgarbage()
else
  optimStateG = {learningRate = opt.LR, beta1 = 0.9}  --beta1 for gan training is 0.9
  optimStateD = {learningRate = opt.LR/2, beta1 = opt.beta1, optimize = true, numUpdates = 0}
end

local parametersD, gradParametersD = disc:getParameters()
local parametersG, gradParametersG = gen:getParameters()

--set up place holders
local inputs_im = torch.Tensor(opt.batchSize, 3, opt.scales[1], opt.scales[1])
local inputs_im2 = torch.Tensor(opt.batchSize, 3, opt.scales[1], opt.scales[1])
local inputs_attr = torch.Tensor(opt.batchSize, opt.attrDim)

inputs_im = inputs_im:cuda()
inputs_im2 = inputs_im2:cuda()
inputs_attr = inputs_attr:cuda()

local w = opt.latentDims[1]
local z = opt.nf
local noise = torch.Tensor(opt.batchSize, z*w*w)
local label = torch.Tensor(opt.batchSize)

local real_label = 1
local fake_label = 0

noise = noise:cuda()
label = label:cuda()

local errD, errG, errW

--fix val noise and val attribute
if (val_im == nil) and (val_attr == nil) then
  for n, sample in valLoader:run() do
    if n == 1 then
      val_im, val_attr = sample.input, sample.target
      break;
    end
  end
  val_im, val_attr = val_im:cuda(), val_attr:cuda()
end
local val_noise = torch.Tensor(opt.batchSize, z*w*w)
val_noise:normal(0, 1)


fake_score = 0.5
local fDx = function(x)
  gradParametersD:zero()
  --counting the number of mistakes
  local top1 = 0
  local total = 0

  -- train with real
  label:fill(real_label)
  local output = disc:forward{inputs_attr, inputs_im}
  local errD_real = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  disc:backward({inputs_attr, inputs_im}, df_do)
  --count correct ones
  local predictions = torch.round(output)
  local label_gan = label:clone():float()
  total = total + output:size(1)
  for j = 1, output:size(1) do
      if label_gan[j] == predictions[j][1] then
          top1 = top1 + 1
      end
  end

  errD_wrong = 0
  if opt.cls_weight > 0 then
    -- train with wrong
    label:fill(fake_label)

    -- Here input fake images
    local output = disc:forward{inputs_attr, inputs_im2}
    errD_wrong = opt.cls_weight*criterion:forward(output, label)
    local df_do = criterion:backward(output, label)
    df_do:mul(opt.cls_weight)
    disc:backward({inputs_attr, inputs_im2}, df_do)
    local predictions = torch.round(output)
    local label_gan = label:clone():float()
    total = total + output:size(1)
    for j = 1, output:size(1) do
      if label_gan[j] == predictions[j][1] then
          top1 = top1 + 1
      end
    end
  end

  -- train with fake
  if opt.noise == 'uniform' then -- regenerate random noise
    noise:uniform(-1, 1)
  elseif opt.noise == 'normal' then
    noise:normal(0, 1)
  end

  local fake = gen:forward{inputs_attr, noise}
  inputs_im:copy(fake)
  label:fill(fake_label)

  local output = disc:forward{inputs_attr, inputs_im}
  local predictions = torch.round(output)
  local label_gan = label:clone():float()
  total = total + output:size(1)
  for j = 1, output:size(1) do
    if label_gan[j] == predictions[j][1] then
        top1 = top1 + 1
    end
  end

  -- update fake score tracker
  local cur_score = output:mean()
  fake_score = 0.99 * fake_score + 0.01 * cur_score

  local errD_fake = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  local fake_weight = 1 - opt.cls_weight
  errD_fake = errD_fake*fake_weight
  df_do:mul(fake_weight)
  disc:backward({inputs_attr, inputs_im}, df_do)

  errD = errD_real + errD_fake + errD_wrong
  errW = errD_wrong

  local gan_erate = 1 - top1/total
  --[[
  optimStateD.optimize = true
  if gan_erate < opt.margin then
    print('not training gan, current gan error rate is' .. gan_erate)
    optimStateD.optimize = false
  end
  --]]
  print('current gan error rate is' .. gan_erate)

  return errD, gradParametersD
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
  gradParametersG:zero()

  if opt.noise == 'uniform' then -- regenerate random noise
    noise:uniform(-1, 1)
  elseif opt.noise == 'normal' then
    noise:normal(0, 1)
  end
  local fake = gen:forward{inputs_attr, noise}
  inputs_im:copy(fake)
  label:fill(real_label) -- fake labels are real for generator cost

  local output = disc:forward{inputs_attr, inputs_im}

  local cur_score = output:mean()

  errG = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  local df_dg = disc:updateGradInput({inputs_attr, inputs_im}, df_do)

  gen:backward({inputs_attr, noise}, df_dg[2])
  return errG, gradParametersG
end

function val(epoch)
  gen:evaluate()
  if epoch == 1 then
    image.save(opt.save .. 'original.png', image.toDisplayTensor(val_im:float():add(1):mul(0.5)))
  end
  local fake = gen:forward{val_attr, val_noise}
  print('SAVING IMAGE: ' .. opt.save .. 'gen_' .. epoch .. '_LR_' .. opt.LR  .. '.png')
  image.save(opt.save .. 'gen_' .. epoch .. '_LR_' .. opt.LR  .. '.png', image.toDisplayTensor(fake:add(1):mul(0.5)))
  parametersD, gradParametersD = nil, nil
  parametersG, gradParametersG = nil, nil
  collectgarbage()
  if epoch % opt.epochStep == 0 then
    torch.save(opt.save .. 'models_' .. epoch .. '.t7', {gen:clearState(), disc:clearState()}) 
    torch.save(opt.save .. 'states_' .. epoch .. '.t7', {optimStateG, optimStateD})
  end
  if epoch % opt.step == 0 then
    optimStateG.learningRate = optimStateG.learningRate * opt.decayLR
    optimStateD.learningRate = optimStateD.learningRate * opt.decayLR
  end
  parametersD, gradParametersD = disc:getParameters()
  parametersG, gradParametersG = gen:getParameters()
  print('Saved image to: ' .. opt.save)
end

function train()
  gen:training()
  disc:training()
  --fix val noise and val attribute
  local size = trainLoader:size()
  print('Starting epoch ' .. epoch)
  local N = 0
    --[[
    for t, sample in trainLoader:run() do
      local timer = torch.Timer()
      if t == 1 then
          N = N + 1
          inputs_im2 = sample.input:cuda()
      else
          N = N + 1 
          inputs_im, inputs_attr = sample.input:cuda(), sample.target:cuda()
          local inputs_im2_prep =  inputs_im:clone()
          local timer = torch.Timer()
 

          --inputs_im:copy(inputs_im_original)
          --inputs_im2:copy(inputs_im2_original)
          --inputs_attr:copy(inputs_attr_original)

          --optim.adam_gan(fDx, parametersD, optimStateD)
          optim.adam(fDx, parametersD, optimStateD)
          optim.adam(fGx, parametersG, optimStateG)
          inputs_im2:copy(inputs_im2_prep)
          print((' | Train: [%d][%d/%d]    Time %.3f   lr=%.4g   errG=%.3f   errD=%.3f   errW=%.3f   fs=%.2f'):format(
               epoch, t, size, timer:time().real,  optimStateG.learningRate, errG and errG or -1, errD and errD or -1,
               errW and errW or -1, fake_score))
          timer:reset()
      end
    end
    val(epoch)
    --]]
  indices_real = torch.randperm(data.train_im:size(1)):long():split(opt.batchSize)
  indices_wrong = torch.randperm(data.train_im:size(1)):long():split(opt.batchSize)
  for i = 1, (#indices_real - 1) do
    local timer = torch.Timer()

    local inputs_im_original = data.train_im:index(1, indices_real[i])
    local inputs_im2_original = data.train_im:index(1, indices_wrong[i])
    local inputs_attr_original = data.train_attr:index(1, indices_real[i])

    -- for last batch with different size
    if inputs_im:size(1) ~= inputs_im_original:size(1) then
      inputs_im = torch.Tensor(inputs_im_original:size(1), 3, opt.scales[1], opt.scales[1])
      inputs_im2 = torch.Tensor(inputs_im2_original:size(1), 3, opt.scales[1], opt.scales[1])
      inputs_attr = torch.Tensor(inputs_attr_original:size(1), data.train_attr:size(2))

      inputs_im = inputs_im:cuda()
      inputs_im2 = inputs_im2:cuda()
      inputs_attr = inputs_attr:cuda()

      fixDimensions()
    end

    inputs_im:copy(inputs_im_original)
    inputs_im2:copy(inputs_im2_original)
    inputs_attr:copy(inputs_attr_original)

    --tm:reset()

    --optim.adam_gan(fDx, parametersD, optimStateD)
    if epoch > 1 then print("optimize fDx") end
    optim.adam(fDx, parametersD, optimStateD)
    if epoch > 1 then print("optimize fGx") end
    optim.adam(fGx, parametersG, optimStateG)

    print((' | Train: [%d][%d/%d]    Time %.3f   lr=%.4g   errG=%.3f   errD=%.3f   errW=%.3f   fs=%.2f'):format(
         epoch, i, #indices_real, timer:time().real,  optimStateG.learningRate, errG and errG or -1, errD and errD or -1,
         errW and errW or -1, fake_score))

    timer:reset()
    --[[
    local fake = gen:forward{val_attr, val_noise}
    if epoch == 1 then
        image.save(opt.save .. 'original_LR_' .. opt.LR .. '_beta1_' .. opt.beta1 .. '.png', image.toDisplayTensor(data.val_im[{{1,128},{},{},{}}]:add(1):mul(0.5)))
    end

    image.save(opt.save .. 'gen_' .. epoch .. '_LR_' .. opt.LR .. '_beta1_' .. opt.beta1 .. '.png', image.toDisplayTensor(fake:add(1):mul(0.5)))
    --]]
  end
end

--[[
-- train
function train()
  local function fGx()      return 0.0, gradParametersG; end
  local function fDx()      return 0.0, gradParametersD; end
  disc:training(); gen:training()

  epoch = epoch  or 1
  print_freq = opt.print_freq or 1
  print('==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  --for epoch = 1, opt.nEpochs do
  print('Starting epoch ' .. epoch)
  local size = trainLoader:size()
  local tic = torch.tic()
  local dataTimer = torch.Timer()
  local inputs_im, inputs_attr, inputs_im2 
  local N = 0 
  for t, sample in trainLoader:run() do
    local timer = torch.Timer()
    if t == 1 then
      N = N + 1
      inputs_im2 = sample.input:cuda()
    else
      N = N + 1 
      inputs_im, inputs_attr = sample.input:cuda(), sample.target:cuda()
      
      --training discriminator
      gradParametersD:zero()
      local top1 = 0
      local total = 0

      -- train with real
      label:fill(real_label)
      local output = disc:forward{inputs_attr, inputs_im}
      local errD_real = criterion:forward(output, label)
      local df_do = criterion:backward(output, label)
      disc:backward({inputs_attr, inputs_im}, df_do)
      --count correct ones
      local predictions = torch.round(output)
      local label_gan = label:clone():float()
      total = total + output:size(1)
      for j = 1, output:size(1) do
          if label_gan[j] == predictions[j][1] then
              top1 = top1 + 1
          end
      end

      errD_wrong = 0
      --train with wrong images
      label:fill(fake_label)
      local output = disc:forward{inputs_attr, inputs_im2}
      errD_wrong = 0.5*criterion:forward(output, label)
      local df_do = criterion:backward(output, label)
      df_do:mul(0.5)
      disc:backward({inputs_attr, inputs_im2}, df_do)
      --count correct ones
      local predictions = torch.round(output)
      local label_gan = label:clone():float()
      total = total + output:size(1)
      for j = 1, output:size(1) do
        if label_gan[j] == predictions[j][1] then
            top1 = top1 + 1
        end
      end
      inputs_im2:copy(inputs_im)

      -- train with fake
      noise:normal(0, 1)
      local fake = gen:forward{inputs_attr, noise}
      inputs_im:copy(fake)
      label:fill(fake_label)
      local output = disc:forward{inputs_attr, inputs_im}
      local errD_fake = criterion:forward(output, label)
      local df_do = criterion:backward(output, label)
      errD_fake = errD_fake*0.5
      df_do:mul(0.5)
      disc:backward({inputs_attr, inputs_im}, df_do)
      --count correct ones
      local predictions = torch.round(output)
      local label_gan = label:clone():float()
      total = total + output:size(1)
      for j = 1, output:size(1) do
        if label_gan[j] == predictions[j][1] then
            top1 = top1 + 1
        end
      end
      
      optim.adam(fDx, parametersD, optimStateD)
      errD = errD_real + errD_fake + errD_wrong
      errW = errD_wrong
      local gan_erate = 1 - top1/total

      --train generator
      gradParametersG:zero()
      noise:normal(0, 1)
      local fake = gen:forward{inputs_attr, noise}
      inputs_im:copy(fake)
      label:fill(real_label) -- fake labels are real for generator cost
      local output = disc:forward{inputs_attr, inputs_im}
      errG = criterion:forward(output, label)
      local df_do = criterion:backward(output, label)
      local df_dg = disc:updateGradInput({inputs_attr, inputs_im}, df_do)
      gen:backward({inputs_attr, noise}, df_dg[2])
      optim.adam(fGx, parametersG, optimStateG)

      if t % print_freq == 1 or t == size then
        print((' | Train: [%d][%d/%d]    Time %.3f   lr=%.4g   errG=%.3f   errD=%.3f   errW=%.3f   errRate=%.2f'):format(
             epoch, t, size, timer:time().real,  optimStateG.learningRate, errG and errG or -1, errD and errD or -1,
             errW and errW or -1, gan_erate))
      end
      timer:reset()
      collectgarbage()
    end
  end
end
--]]


