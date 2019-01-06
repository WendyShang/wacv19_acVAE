optim = require 'optim'

-- load models and training criterions
local cvae_encoder, cvae_decoder, prior, sampling_z = table.unpack(models)
local KLD, ReconCriterion = table.unpack(criterions)
local optimStatecvae_encoder, optimStatecvae_decoder,optimStatePrior

-- reload if previous checkpoint exists
-- otherwise initialize optimStates
if opt.retrain ~= 'none' and opt.optimState ~= 'none' then
  local models_resume = torch.load(opt.retrain)
  local states_resume = torch.load(opt.optimState)
  cvae_encoder, cvae_decoder, prior = nil, nil, nil
  cvae_encoder, cvae_decoder, prior = table.unpack(models_resume)
  optimStatecvae_encoder, optimStatecvae_decoder, optimStatePrior = table.unpack(states_resume)
  collectgarbage()
else
  optimStatecvae_encoder = { learningRate = opt.LR, optimize = true, numUpdates = 0}
  optimStatecvae_decoder = { learningRate = opt.LR, optimize = true, numUpdates = 0}
  optimStatePrior        = { learningRate = opt.LR, optimize = true, numUpdates = 0}
end

-- model parameters and gradient parameters
parameterscvae_encoder, gradParameterscvae_encoder = cvae_encoder:getParameters()
parameterscvae_decoder, gradParameterscvae_decoder = cvae_decoder:getParameters()
parametersPrior,        gradParametersPrior        = prior:getParameters()


-- train cVAE
function train()
  local function f_vae_encoder()      return 0.0, gradParameterscvae_encoder; end
  local function f_vae_decoder()      return 0.0, gradParameterscvae_decoder; end
  local function f_prior()            return 0.0, gradParametersPrior; end
  cvae_encoder:training(); cvae_decoder:training(); prior:training();

  --[[
  epoch = epoch or 1
  print_freq = opt.print_freq or 1
  print('==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
  local size = trainLoader:size()
  local N, err_vae_encoder_total, err_vae_decoder_total, err_gan_total = 0, 0.0, 0.0, 0.0
  local tic = torch.tic()
  local dataTimer = torch.Timer()
  for t, sample in trainLoader:run() do
    N = N + 1
    local timer = torch.Timer()
    local dataTime = dataTimer:time().real

    -- load data and augmentation (horizontal flip)
    local input_im, input_attr = sample.input:cuda(), sample.target:cuda()
    local inputs = {input_attr:cuda(), input_im:cuda()}
    collectgarbage()
  --]]

  epoch = epoch or 1
  print('==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
  local indices = torch.randperm(data.train_im:size(1)):long():split(opt.batchSize)
  indices[#indices] = nil
  local size = #indices
  local tic = torch.tic()
  local err_cvae_encoder_total, err_cvae_decoder_total = 0, 0
  local N, err_vae_encoder_total, err_vae_decoder_total, err_gan_total = 0, 0.0, 0.0, 0.0
  local N = 0
  local reconstruction
  local inputs
  local inputs_im
  local inputs_attr
  for t,v in ipairs(indices) do
    N = N + 1
    local timer = torch.Timer()

    -- load data and augmentation (horizontal flip)
    local inputs_im_original = data.train_im:index(1,v)
    local inputs_attr_original = data.train_attr:index(1,v)

    inputs_im = torch.Tensor(inputs_im_original:size(1), 3, opt.scales[1], opt.scales[1])
    inputs_attr = torch.Tensor(inputs_attr_original:size(1), inputs_attr_original:size(2))

    for i = 1, inputs_im_original:size(1) do
      inputs_im[i] = opt.preprocess_train(image.scale(inputs_im_original[i], opt.scales[1], opt.scales[1]))
      inputs_attr[i] = inputs_attr_original[i]
    end

    inputs_im = inputs_im:cuda()
    inputs_attr = inputs_attr:cuda()

    inputs = {inputs_attr, inputs_im}

    --[[  update from reconstruction
          forward pass: cvae_encoder -> sampling_z -> cvae_decoder 
          backward pass: data recon -> sampling_z + KLD -> cvae_encoder
    --]]

    -- encoder > sampling > decoder
    local output_mean_log_var = cvae_encoder:forward({inputs[1], inputs[2]});
    local latent_z = sampling_z:forward(output_mean_log_var):clone()
    local reconstruction = cvae_decoder:forward({inputs[1], latent_z}):clone()


    -- prior > KL divergence
    local output_prior = prior:forward(inputs[1])
    KLDerr = KLD:forward(output_mean_log_var, output_prior)
    dKLD_dtheta = KLD:backward(output_mean_log_var, output_prior)
    for j = 1,4 do dKLD_dtheta[j]:mul(opt.alpha) end
    
    -- loss & backprop & optimize
    local Dislikerr = ReconCriterion:forward(reconstruction, inputs[2])
    local df_do = ReconCriterion:backward(reconstruction, inputs[2])
    local df_ddecoder = cvae_decoder:updateGradInput({inputs[1], latent_z}, df_do)
    local df_dsampler = sampling_z:updateGradInput(output_mean_log_var, df_ddecoder[2])
    cvae_encoder:zeroGradParameters()
    cvae_encoder:backward({inputs[1], inputs[2]}, {df_dsampler[1] + dKLD_dtheta[1], df_dsampler[2] + dKLD_dtheta[2]})
    optim[opt.optimization](f_vae_encoder, parameterscvae_encoder, optimStatecvae_encoder)
    local err_vae_encoder = Dislikerr + KLDerr * opt.alpha
    err_vae_encoder_total = err_vae_encoder_total + err_vae_encoder

    -- update prior and optimize 
    prior:zeroGradParameters()
    prior:backward(inputs[1], {dKLD_dtheta[3], dKLD_dtheta[4]}) 
    optim[opt.optimization](f_prior, parametersPrior, optimState_prior)
    local err_prior = Dislikerr + KLDerr * (opt.alpha) * 0.5

    -- update cvae_decoder and optimize
    cvae_decoder:zeroGradParameters();
    cvae_decoder:backward({inputs[1], latent_z}, df_do)
    optim[opt.optimization](f_vae_decoder, parameterscvae_decoder, optimStatecvae_decoder)
    err_vae_decoder = Dislikerr 
    err_vae_decoder_total = err_vae_decoder_total + err_vae_decoder

    -- print logs
    if t % print_freq == 0 or t == size then
      print((' | Train: [%d][%d/%d]    Time %.3f (%.3f)  encoder %7.3f (%7.3f)  decoder %7.3f (%7.3f)'):format(
         epoch, t, size, timer:time().real, dataTime, err_vae_encoder, err_vae_encoder_total/N, err_vae_decoder, err_vae_decoder_total/N))
    end
    latent_z, df_do = nil, nil
    timer:reset()
    dataTimer:reset()
    collectgarbage()
  end
  print(('Train loss (vae encoder, vae decoder: %.3f, %.3f'):format(err_vae_encoder_total/N, err_vae_decoder_total/N))
end

function val()
  cvae_encoder:evaluate(); cvae_decoder:evaluate(); prior:evaluate();

  --[[
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
  --]]
  local val_attr = data.val_attr[{{1, 128}}]
  local val_im_original = data.val_im[{{1, 128}}]
  local val_attr = torch.Tensor(val_attr:size(1), val_attr:size(2))
  local val_im = torch.Tensor(val_im_original:size(1), 3, opt.scales[1], opt.scales[1])
  for i = 1, val_im_original:size(1) do
    val_im[i] = opt.preprocess_test(image.scale(val_im_original[i], opt.scales[1], opt.scales[1]))
    val_attr[i] = val_attr[i]
  end

  val_attr = val_attr:cuda()
  val_im = val_im:cuda()

  if epoch == 1 then
    image.save(opt.save .. 'original.png', image.toDisplayTensor(val_im_original:float():add(1):mul(0.5)))
  end

  --(1) test reconstruction 
  cvae_encoder:forward({val_attr,val_im})
  local val_latent_z = sampling_z:forward(cvae_encoder.output)
  local reconstruction_save = cvae_decoder:forward({val_attr,val_latent_z})
  image.save(opt.save .. 'recon_' .. epoch .. '_LR_' .. opt.LR .. '_alphas_' .. opt.alpha .. '.png', image.toDisplayTensor(reconstruction_save:float():add(1):mul(0.5)))

  --(2) test generation
  val_prior = prior:forward(val_attr)
  local val_latent_z = sampling_z:forward(val_prior)
  local generation_val = cvae_decoder:forward({val_attr,val_latent_z})
  image.save( opt.save .. 'gen_' .. epoch .. '_LR_' .. opt.LR .. '_alphas_' .. opt.alpha ..  '.png', image.toDisplayTensor(generation_val:float():add(1):mul(0.5)))
  parameterscvae_encoder, gradParameterscvae_encoder    = nil, nil
  parameterscvae_decoder, gradParameterscvae_decoder    = nil, nil
  parametersPrior,        gradParametersPrior           = nil, nil
  collectgarbage()
  if epoch % opt.epochStep == 0 then
     torch.save(opt.save .. 'models_' .. epoch .. '.t7', {cvae_encoder:clearState(),  cvae_decoder:clearState(), prior:clearState()})
     torch.save(opt.save .. 'states_' .. epoch .. '.t7', {optimStatecvae_encoder, optimStatecvae_decoder, optimStatePrior})
  end
  if epoch % opt.step == 0 then
    optimStatecvae_encoder.learningRate  = optimStatecvae_encoder.learningRate*opt.decayLR
    optimStatecvae_decoder.learningRate  = optimStatecvae_decoder.learningRate*opt.decayLR
    optimStatePrior.learningRate        = optimStatePrior.learningRate*opt.decayLR
  end
  parameterscvae_encoder, gradParameterscvae_encoder = cvae_encoder:getParameters()
  parameterscvae_decoder, gradParameterscvae_decoder = cvae_decoder:getParameters()
  parametersPrior,        gradParametersPrior        = prior:getParameters()
  print('Saved image to: ' .. opt.save)
end
