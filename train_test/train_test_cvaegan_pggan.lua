--[[
  progressively growing CVAE-GAN from 64x64 to 128x128

  local upsample = nn.SpatialUpSamplingNearest(2)
  local downsample = nn.SpatialAveragePooling(2,2,2,2)
--]]

optim = require 'optim'
require 'utils.adam_gan'

local function init_from_rgb_encoder(encoder_conv, from_rgb_encoder_layer)
  local from_rgb_encoder_layer1 = from_rgb_encoder_layer:clone()
  local downsample1 = nn.SpatialAveragePooling(2,2,2,2):cuda()
  local downsample2 = nn.SpatialAveragePooling(2,2,2,2):cuda()
  -- branch1: downsample followed by from_rgb_encoder_layer
  local branch1 = nn.Sequential()
  branch1:add(downsample1):add(from_rgb_encoder_layer1)
  -- branch2: from_rgb_layer2 followed by encoder_conv followed by downsample
  local branch2 = nn.Sequential()
  branch2:add(from_rgb_encoder_layer):add(encoder_conv):add(downsample2)
  local from_rgb_encoder = nn.Sequential()
  from_rgb_encoder:add(nn.ConcatTable(2):add(branch1):add(branch2))
  from_rgb_encoder:cuda()
  
  local merge_from_rgb_encoder = nn.Sequential()
  merge_from_rgb_encoder:add(nn.ParallelTable(2):add(nn.MulConstant(1.0)):add(nn.MulConstant(0.0))):add(nn.CAddTable())
  merge_from_rgb_encoder:cuda()

  if opt.cudnn == 'deterministic' then
    from_rgb_encoder:apply(function(m)
      if m.setMode then m:setMode(1, 1, 1) end
    end)
    merge_from_rgb_encoder:apply(function(m)
      if m.setMode then m:setMode(1, 1, 1) end
    end)
  end
  return from_rgb_encoder, merge_from_rgb_encoder
end

local function init_to_rgb(decoder_conv, to_rgb_layer)
  local to_rgb_layer1 = to_rgb_layer:clone()
  local upsample = nn.SpatialUpSamplingNearest(2):cuda()
  local branch1 = nn.Sequential()
  branch1:add(to_rgb_layer1)
  local branch2 = nn.Sequential()
  branch2:add(decoder_conv):add(to_rgb_layer)
  local to_rgb = nn.Sequential()
  to_rgb:add(upsample):add(nn.ConcatTable(2):add(branch1):add(branch2))
  to_rgb:cuda()

  local merge_to_rgb = nn.Sequential()
  merge_to_rgb:add(nn.ParallelTable(2):add(nn.MulConstant(1.0)):add(nn.MulConstant(0.0))):add(nn.CAddTable())
  merge_to_rgb:cuda()
  if opt.cudnn == 'deterministic' then
    to_rgb:apply(function(m)
      if m.setMode then m:setMode(1, 1, 1) end
    end)
    merge_to_rgb:apply(function(m)
      if m.setMode then m:setMode(1, 1, 1) end
    end)
  end
  return to_rgb, merge_to_rgb
end

local function init_from_rgb(disc_conv, from_rgb_layer)
  local from_rgb_layer1 = from_rgb_layer:clone()
  local downsample1 = nn.SpatialAveragePooling(2,2,2,2):cuda()
  local downsample2 = nn.SpatialAveragePooling(2,2,2,2):cuda()
  -- branch1: downsample followed by from_rgb_layer
  local branch1 = nn.Sequential()
  branch1:add(downsample1):add(from_rgb_layer1)
  -- branch2: from_rgb_layer2 followed by disc_conv followed by downsample
  local branch2 = nn.Sequential()
  branch2:add(from_rgb_layer):add(disc_conv):add(downsample2)
  local from_rgb = nn.Sequential()
  from_rgb:add(nn.ConcatTable(2):add(branch1):add(branch2))
  from_rgb:cuda()

  local merge_from_rgb = nn.Sequential()
  merge_from_rgb:add(nn.ParallelTable(2):add(nn.MulConstant(1.0)):add(nn.MulConstant(0.0))):add(nn.CAddTable())
  merge_from_rgb:cuda()
  if opt.cudnn == 'deterministic' then
    from_rgb:apply(function(m)
      if m.setMode then m:setMode(1, 1, 1) end
    end)
    merge_from_rgb:apply(function(m)
      if m.setMode then m:setMode(1, 1, 1) end
    end)
  end
  return from_rgb, merge_from_rgb
end

-- multiGPU training
local function makeDataParallelTable(model, nGPU)
  local gpus = torch.range(1, nGPU):totable()
  local fastest, benchmark = cudnn.fastest, cudnn.benchmark

  local dpt = nn.DataParallelTable(1, true, true)
    :add(model, gpus)
    :threads(function()
       local cudnn = require 'cudnn'
       require 'stn'
       cudnn.fastest, cudnn.benchmark = fastest, benchmark
    end)
  --dpt.gradInput = nil
  model = dpt:cuda()
  return model
end

local function deepCopy(tbl)
  -- creates a copy of a network with new modules and the same tensors
  local copy = {}
  for k, v in pairs(tbl) do
    if type(v) == 'table' then
      copy[k] = deepCopy(v)
    else
      copy[k] = v
    end
  end
  if torch.typename(tbl) then
    torch.setmetatable(copy, torch.typename(tbl))
  end
  return copy
end


-- load models and training criterions
local vae_encoder, encoder_conv, from_rgb_encoder_layer, vae_decoder, decoder_conv, to_rgb_layer, prior, sampling_z, from_rgb_layer, disc_conv, disc = table.unpack(models)
local KLD, ReconCriterion, BCECriterion = table.unpack(criterions)
local sampling_z2 = sampling_z:clone()
local optimState_enc, optimState_dec, optimState_prior, optimState_disc, optimState_from_rgb_enc, optimState_to_rgb, optimState_from_rgb

-- reload if previous checkpoint exists
-- otherwise initialize optimStates
if opt.init_weight_from ~= 'none' then
  -- initialize weight from pretrained model
  local models_resume = torch.load(opt.init_weight_from)
  vae_encoder, from_rgb_encoder_layer, vae_decoder, to_rgb_layer, prior, from_rgb_layer, disc = table.unpack(models_resume)
  optimState_enc  = { learningRate = opt.LR_mult*opt.LR, optimize = true, numUpdates = 0}
  optimState_dec  = { learningRate = opt.LR_mult*opt.LR, optimize = true, numUpdates = 0}
  optimState_prior         = { learningRate = opt.LR_mult*opt.LR, optimize = true, numUpdates = 0}
  optimState_disc          = { learningRate = opt.LR_mult*opt.LR/2, beta1 = opt.beta1, optimize = true, numUpdates = 0}
  optimState_from_rgb_enc = { learningRate = opt.LR, optimize = true, numUpdates = 0}
  optimState_to_rgb       = { learningRate = opt.LR, optimize = true, numUpdates = 0}
  optimState_from_rgb     = { learningRate = opt.LR/2, beta1 = opt.beta1, optimize = true, numUpdates = 0}
elseif opt.retrain ~= 'none' then
  local models_resume = torch.load(opt.retrain)
  if opt.optimState ~= 'none' then
      local states_resume = torch.load(opt.optimState)
      optimState_enc, optimState_from_rgb_enc, optimState_dec, optimState_to_rgb, optimState_prior, optimState_from_rgb, optimState_disc = table.unpack(states_resume)
  end
  vae_encoder, encoder_conv, from_rgb_encoder_layer, vae_decoder, decoder_conv, to_rgb_layer, prior, from_rgb_layer, disc_conv, disc = nil, nil, nil, nil, nil, nil, nil, nil, nil, nil
  vae_encoder, encoder_conv, from_rgb_encoder_layer, vae_decoder, decoder_conv, to_rgb_layer, prior, from_rgb_layer, disc_conv, disc = table.unpack(models_resume)
  collectgarbage()
else
  optimState_enc          = { learningRate = opt.LR, optimize = true, numUpdates = 0}
  optimState_dec          = { learningRate = opt.LR, optimize = true, numUpdates = 0}
  optimState_prior        = { learningRate = opt.LR, optimize = true, numUpdates = 0}
  optimState_disc         = { learningRate = opt.LR/2, beta1 = opt.beta1, optimize = true, numUpdates = 0}
  optimState_from_rgb_enc = { learningRate = opt.LR, optimize = true, numUpdates = 0}
  optimState_to_rgb       = { learningRate = opt.LR, optimize = true, numUpdates = 0}
  optimState_from_rgb     = { learningRate = opt.LR/2, beta1 = opt.beta1, optimize = true, numUpdates = 0}
end

-- construct from_rgb_encoder, to_rgb, from_rgb as well as mergers
local from_rgb_encoder, merge_from_rgb_encoder = init_from_rgb_encoder(encoder_conv, from_rgb_encoder_layer)
local to_rgb, merge_to_rgb = init_to_rgb(decoder_conv, to_rgb_layer)
local from_rgb, merge_from_rgb = init_from_rgb(disc_conv, from_rgb_layer)

-- multi-GPU
if opt.nGPU > 1 then
  vae_encoder = makeDataParallelTable(vae_encoder, opt.nGPU)
  vae_decoder = makeDataParallelTable(vae_decoder, opt.nGPU)
  prior = makeDataParallelTable(prior, opt.nGPU)
  disc = makeDataParallelTable(disc, opt.nGPU)
  from_rgb_encoder = makeDataParallelTable(from_rgb_encoder, opt.nGPU)
  to_rgb = makeDataParallelTable(to_rgb, opt.nGPU)
  from_rgb = makeDataParallelTable(from_rgb, opt.nGPU)
end

-- model parameters and gradient parameters
params_enc,           gradParams_enc          = vae_encoder:getParameters()
params_dec,           gradParams_dec          = vae_decoder:getParameters()
params_prior,         gradParams_prior        = prior:getParameters()
params_disc,          gradParams_disc         = disc:getParameters()
params_from_rgb_enc,  gradParams_from_rgb_enc = from_rgb_encoder:getParameters()
params_to_rgb,        gradParams_to_rgb       = to_rgb:getParameters()
params_from_rgb,      gradParams_from_rgb     = from_rgb:getParameters()


-- train VAE-GAN
function train()
  local function f_vae_encoder()      return 0.0, gradParams_enc; end
  local function f_vae_decoder()      return 0.0, gradParams_dec; end
  local function f_prior()            return 0.0, gradParams_prior; end
  local function f_gan()              return 0.0, gradParams_disc; end
  local function f_from_rgb_encoder() return 0.0, gradParams_from_rgb_enc; end
  local function f_to_rgb()           return 0.0, gradParams_to_rgb; end
  local function f_from_rgb()         return 0.0, gradParams_from_rgb; end
  
  vae_encoder:training(); vae_decoder:training(); prior:training(); disc:training();
  from_rgb_encoder:training(); to_rgb:training(); from_rgb:training()

  local function generate_discriminator_batch(inputs, labels, recon, sample, batchSize, fakeLabel)
    local randperm = torch.randperm(batchSize)
    local outputs = inputs:clone()
    for index = 1, batchSize do
      if index % fakeLabel == 0 then
        labels[index] = 0
        outputs[index]:copy(recon[index])
      end
      if (index+1) % fakeLabel == 0 then
        labels[index] = 0
        outputs[index]:copy(sample[index])
      end
      if (index+2) % fakeLabel == 0 then
        -- mismatch between attribute and original image
        labels[index] = 0
        outputs[index]:copy(inputs[randperm[index]])
      end
    end
    return outputs, labels
  end

  local function change_weight(merger, alpha)
    merger:get(1):get(1).constant_scalar = 1-alpha
    merger:get(1):get(2).constant_scalar = alpha
    return merger
  end

  local function schedule_alpha(epoch)
    -- alpha linearly scheduled from 0 to 1
    local alpha = math.min(1-(opt.nEpochs/2-(epoch-1))/(opt.nEpochs/2), opt.max_alpha)
    print(string.format('alpha increases to %.4f', alpha))
    merge_from_rgb_encoder = change_weight(merge_from_rgb_encoder, alpha)
    merge_to_rgb = change_weight(merge_to_rgb, alpha)
    merge_from_rgb = change_weight(merge_from_rgb, alpha)
  end


  epoch = epoch or 1
  print_freq = opt.print_freq or 1
  print('==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
  local size = trainLoader:size()
  local N, err_vae_encoder_total, err_vae_decoder_total, err_gan_total = 0, 0.0, 0.0, 0.0
  local reconstruction, inputs, input_im, input_attr
  local gan_update_rate, gan_error_rate, iteration = 0.0, 0.0, 0
  local label_recon, label_sample, label_gan = torch.ones(opt.batchSize):cuda(), torch.ones(opt.batchSize):cuda(), torch.ones(opt.batchSize):cuda()
  local tic = torch.tic()
  local dataTimer = torch.Timer()
  schedule_alpha(epoch)
  for t, sample in trainLoader:run() do
    N = N + 1
    local timer = torch.Timer()
    local dataTime = dataTimer:time().real

    --[[
          load data (horizontal flip) 
    --]]

    input_im, input_attr = sample.input:cuda(), sample.target:cuda()
    inputs = {input_attr, input_im}

    local reconstruction_sample, z_sample, latent_z, df_do

    --[[
          inference and backprop
          >> from_rgb_encoder > vae_encoder > sampling_z > vae_decoder > to_rgb
          >> prior > KLD
          >> from_rgb > disc
    --]]

    -- encoder > sampling > decoder
    from_rgb_encoder:forward(inputs[2])
    merge_from_rgb_encoder:forward(from_rgb_encoder.output)
    local output_mean_log_var = vae_encoder:forward({inputs[1], merge_from_rgb_encoder.output});
    latent_z = sampling_z(output_mean_log_var):clone()
    vae_decoder:forward({inputs[1], latent_z})
    to_rgb:forward(vae_decoder.output)
    reconstruction = merge_to_rgb:forward(to_rgb.output):clone()

    -- prior > KL divergence
    local output_prior = prior:forward(inputs[1])
    KLDerr = KLD:forward(output_mean_log_var, output_prior)
    dKLD_dtheta = KLD:backward(output_mean_log_var, output_prior)
    for j = 1,4 do dKLD_dtheta[j]:mul(opt.alpha) end
    
    -- loss & backprop
    local Dislikerr = ReconCriterion:forward(reconstruction, inputs[2])
    local df_do = ReconCriterion:backward(reconstruction, inputs[2])

    merge_to_rgb:updateGradInput(to_rgb.output, df_do)
    to_rgb:updateGradInput(vae_decoder.output, merge_to_rgb.gradInput)
    vae_decoder:updateGradInput({inputs[1], latent_z}, to_rgb.gradInput)
    local df_dsampler = sampling_z:updateGradInput(output_mean_log_var, vae_decoder.gradInput[2])

    
    -- update vae_encoder, from_rgb_encoder
    vae_encoder:zeroGradParameters(); from_rgb_encoder:zeroGradParameters();
    vae_encoder:backward({inputs[1], merge_from_rgb_encoder.output}, {df_dsampler[1] + dKLD_dtheta[1], df_dsampler[2] + dKLD_dtheta[2]})
    merge_from_rgb_encoder:backward(from_rgb_encoder.output, vae_encoder.gradInput[2])
    from_rgb_encoder:backward(inputs[2], merge_from_rgb_encoder.gradInput)
    
    -- optimize & loss
    optim[opt.optimization](f_vae_encoder, params_enc, optimState_enc)
    optim[opt.optimization](f_from_rgb_encoder, params_from_rgb_enc, optimState_from_rgb_enc)
    local err_vae_encoder = Dislikerr + KLDerr * opt.alpha
    err_vae_encoder_total = err_vae_encoder_total + err_vae_encoder

    -- update prior
    prior:zeroGradParameters()
    prior:backward(inputs[1], {dKLD_dtheta[3], dKLD_dtheta[4]})
    
    -- optimize & loss
    optim[opt.optimization](f_prior, params_prior, optimState_prior)
    local err_prior = Dislikerr + KLDerr * (opt.alpha) * 0.5


    -- update vae_decoder, to_rgb
    vae_decoder:zeroGradParameters(); to_rgb:zeroGradParameters();

    -- GAN + decoder for reconstructed images
    from_rgb:forward(reconstruction)
    merge_from_rgb:forward(from_rgb.output)
    disc:forward({inputs[1], merge_from_rgb.output})
    local gan_err = BCECriterion:forward(disc.output, label_recon)
    BCECriterion:backward(disc.output, label_recon)
    disc:updateGradInput({inputs[1], merge_from_rgb.output}, BCECriterion.gradInput)
    merge_from_rgb:updateGradInput(from_rgb.output, disc.gradInput[2])
    from_rgb:updateGradInput(reconstruction, merge_from_rgb.gradInput)
    merge_to_rgb:backward(to_rgb.output, from_rgb.gradInput * opt.beta + df_do)
    to_rgb:backward(vae_decoder.output, merge_to_rgb.gradInput)
    vae_decoder:backward({inputs[1], latent_z}, to_rgb.gradInput)

    -- generate images from random sample
    local output_prior = prior:forward(inputs[1])
    local z_sample = sampling_z:forward(output_prior)
    vae_decoder:forward({inputs[1], z_sample})
    to_rgb:forward(vae_decoder.output)
    reconstruction_sample = merge_to_rgb:forward(to_rgb.output):clone()

    -- GAN for sampled images
    from_rgb:forward(reconstruction_sample)
    merge_from_rgb:forward(from_rgb.output)
    disc:forward({inputs[1], merge_from_rgb.output})
    local gan_err_sample = BCECriterion:forward(disc.output, label_sample)
    BCECriterion:backward(disc.output, label_sample)
    disc:updateGradInput({inputs[1], merge_from_rgb.output}, BCECriterion.gradInput)
    merge_from_rgb:updateGradInput(from_rgb.output, disc.gradInput[2])
    from_rgb:updateGradInput(reconstruction_sample, merge_from_rgb.gradInput)
    merge_to_rgb:backward(to_rgb.output, from_rgb.gradInput * opt.beta)
    to_rgb:backward(vae_decoder.output, merge_to_rgb.gradInput)
    vae_decoder:backward({inputs[1], z_sample}, to_rgb.gradInput)
    
    -- optimize & loss
    optim[opt.optimization](f_vae_decoder, params_dec, optimState_dec)
    optim[opt.optimization](f_to_rgb, params_to_rgb, optimState_to_rgb)
    err_vae_decoder = Dislikerr + gan_err * opt.beta
    err_vae_decoder_total = err_vae_decoder_total + err_vae_decoder

    
    -- from_rgb > disc    
    local inputs_im_gan, label_gan = generate_discriminator_batch(inputs[2], label_gan:fill(1), reconstruction, reconstruction_sample, opt.batchSize, opt.fakeLabel)
    from_rgb:forward(inputs_im_gan)
    merge_from_rgb:forward(from_rgb.output)
    local outputs_gan = disc:forward({inputs[1], merge_from_rgb.output})
    local gan_err = BCECriterion:forward(outputs_gan, label_gan)

    -- compute disc error rate
    local predictions = torch.round(outputs_gan):squeeze()
    local top1 = torch.eq(predictions:float(), label_gan:float()):sum()
    local gan_erate = 1 - top1/outputs_gan:size(1)
    gan_error_rate = gan_error_rate + gan_erate
    optimState_disc.optimize = true
    optimState_from_rgb.optimize = true
    if gan_erate < opt.margin then
      optimState_disc.optimize = false
      optimState_from_rgb.optimize = false
    else
      -- update discriminator, from_rgb
      gan_update_rate = gan_update_rate + 1
      disc:zeroGradParameters(); from_rgb:zeroGradParameters();
      BCECriterion:backward(disc.output, label_gan)
      disc:backward({inputs[1], merge_from_rgb.output}, BCECriterion.gradInput)
      merge_from_rgb:backward(from_rgb.output, disc.gradInput[2])
      from_rgb:backward(inputs_im_gan, merge_from_rgb.gradInput)
    end
    optim[opt.optimization .. '_gan'](f_gan, params_disc, optimState_disc)
    optim[opt.optimization .. '_gan'](f_from_rgb, params_from_rgb, optimState_from_rgb)
    err_gan_total = err_gan_total + gan_err
    

    -- print scores
    iteration = iteration + 1
    if t % print_freq == 1 or t == size then
      -- print only every 10 epochs
      print((' | Train: [%d][%d/%d]    Time %.3f (%.3f)  encoder %7.3f (%7.3f)  decoder %7.3f (%7.3f)  gan %7.3f (%7.3f)  gan update rate %.3f (erate %.3f)'):format(
         epoch, t, size, timer:time().real, dataTime, err_vae_encoder, err_vae_encoder_total/N, err_vae_decoder, err_vae_decoder_total/N, gan_err, err_gan_total/N, gan_update_rate/iteration, gan_error_rate/iteration))
      gan_update_rate, gan_error_rate, iteration = 0.0, 0.0, 0.0
    end
    reconstruction_sample, z_sample, latent_z, df_do, Dislikerr = nil, nil, nil, nil, nil
    timer:reset()
    dataTimer:reset()
    collectgarbage()
  end

  print(('Train loss (vae encoder, vae decoder, gan: '..'%.2f ,'..'%.2f ,'..'%.2f '):format(err_vae_encoder_total/N, err_vae_decoder_total/N, err_gan_total/N))
end

function val()
  vae_encoder:evaluate(); vae_decoder:evaluate(); prior:evaluate(); disc:evaluate();
  encoder_conv:evaluate(); decoder_conv:evaluate(); disc_conv:evaluate();
  from_rgb_encoder_layer:evaluate(); to_rgb_layer:evaluate(); from_rgb_layer:evaluate();
  
  if (val_im == nil) and (val_attr == nil) then
    for n, sample in valLoader:run() do
      if n == 1 then
        val_im, val_attr = sample.input, sample.target
        break;
      end
    end
    val_im, val_attr = val_im:cuda(), val_attr:cuda()
  end

  if epoch == 1 then
    image.save(opt.save .. 'original.png', image.toDisplayTensor(val_im:float():add(1):mul(0.5)))
  end

  --(1) test reconstruction
  from_rgb_encoder:forward(val_im)
  vae_encoder:forward({val_attr, from_rgb_encoder.output[2]})
  local val_latent_z = sampling_z:forward(vae_encoder.output)
  vae_decoder:forward({val_attr,val_latent_z})
  local reconstruction_save = to_rgb:forward(vae_decoder.output)[2]
  image.save(opt.save .. 'recon_' .. epoch .. '_LR_' .. opt.LR .. '_alphas_' .. opt.alpha .. '.png', image.toDisplayTensor(reconstruction_save:float():add(1):mul(0.5)))

  --(2) test generation
  val_prior = prior:forward(val_attr)
  local val_latent_z = sampling_z:forward(val_prior)
  vae_decoder:forward({val_attr,val_latent_z})
  local generation_val = to_rgb:forward(vae_decoder.output)[2]
  image.save( opt.save .. 'gen_' .. epoch .. '_LR_' .. opt.LR .. '_alphas_' .. opt.alpha ..  '.png', image.toDisplayTensor(generation_val:float():add(1):mul(0.5)))
  params_enc,           gradParams_enc            = nil, nil
  params_dec,           gradParams_dec            = nil, nil
  params_prior,         gradParams_prior          = nil, nil
  params_disc,          gradParams_disc           = nil, nil
  params_from_rgb_enc,  gradParams_from_rgb_enc   = nil, nil
  params_to_rgb,        gradParams_to_rgb         = nil, nil
  params_from_rgb,      gradParams_from_rgb       = nil, nil
  collectgarbage()
  if epoch % opt.epochStep == 0 then
    torch.save(opt.save .. 'models_' .. epoch .. '.t7', {vae_encoder:clearState(), encoder_conv:clearState(), from_rgb_encoder_layer:clearState(), vae_decoder:clearState(), decoder_conv:clearState(), to_rgb_layer:clearState(), prior:clearState(), from_rgb_layer:clearState(), disc_conv:clearState(), disc:clearState()})
    torch.save(opt.save .. 'states_' .. epoch .. '.t7', {optimState_enc, optimState_from_rgb_enc, optimState_dec, optimState_to_rgb, optimState_prior, optimState_from_rgb, optimState_disc})
  end

  if epoch % opt.step == 0 then
    optimState_enc.learningRate           = optimState_enc.learningRate*opt.decayLR
    optimState_dec.learningRate           = optimState_dec.learningRate*opt.decayLR
    optimState_prior.learningRate         = optimState_prior.learningRate*opt.decayLR
    optimState_disc.learningRate          = optimState_disc.learningRate*opt.decayLR
    optimState_from_rgb_enc.learningRate  = optimState_from_rgb_enc.learningRate*opt.decayLR
    optimState_to_rgb.learningRate        = optimState_to_rgb.learningRate*opt.decayLR
    optimState_from_rgb.learningRate      = optimState_from_rgb.learningRate*opt.decayLR
  end
  params_enc,           gradParams_enc          = vae_encoder:getParameters()
  params_dec,           gradParams_dec          = vae_decoder:getParameters()
  params_prior,         gradParams_prior        = prior:getParameters()
  params_disc,          gradParams_disc         = disc:getParameters()
  params_from_rgb_enc,  gradParams_from_rgb_enc = from_rgb_encoder:getParameters()
  params_to_rgb,        gradParams_to_rgb       = to_rgb:getParameters()
  params_from_rgb,      gradParams_from_rgb     = from_rgb:getParameters()
  print('Saved image to: ' .. opt.save)
end

function generate(opt)
  print('Saving Generateions to: ' .. opt.testSample)
  os.execute('mkdir -p ' .. opt.testSample)
  vae_encoder:evaluate(); vae_decoder:evaluate(); prior:evaluate(); disc:evaluate();
  encoder_conv:evaluate(); decoder_conv:evaluate(); disc_conv:evaluate();
  from_rgb_encoder_layer:evaluate(); to_rgb_layer:evaluate(); from_rgb_layer:evaluate();
  local upsample = nn.SpatialUpSamplingNearest(2):cuda()
  local time = sys.clock()
  for t = 1, data.test_im:size(1) do
    local inputs_attr_original = data.test_attr[t]
    inputs_attr = torch.Tensor(1, opt.attrDim)
    inputs_attr[1] = inputs_attr_original:view(1,opt.attrDim)
    inputs_attr = inputs_attr:cuda()
    inputs_attr = inputs_attr:repeatTensor(opt.batchSize,1)
    local val_prior = prior:forward(inputs_attr)
    local val_latent_z = sampling_z:forward(val_prior)
    vae_decoder:forward({inputs_attr,val_latent_z})
    upsample:forward(vae_decoder.output)
    decoder_conv:forward(upsample.output)
    local generation_val = to_rgb_layer(decoder_conv.output)
    generation_val = generation_val:float()
    generation_val:add(1):mul(0.5)
    image.save(paths.concat(opt.testSample, t .. '.png'), image.toDisplayTensor(generation_val[1]))
  end
end

