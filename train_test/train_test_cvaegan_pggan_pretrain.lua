optim = require 'optim'
require 'utils.adam_gan'

-- load models and training criterions
local vae_encoder, from_rgb_encoder, vae_decoder, to_rgb, prior, sampling_z, from_rgb, disc = table.unpack(models)
local KLD, ReconCriterion, BCECriterion = table.unpack(criterions)
local sampling_z2 = sampling_z:clone()
local optimState_enc, optimState_dec, optimState_prior, optimState_disc, optimState_from_rgb_enc, optimState_to_rgb, optimState_from_rgb

-- reload if previous checkpoint exists
-- otherwise initialize optimStates
if opt.retrain ~= 'none' and opt.optimState ~= 'none' then
  local models_resume = torch.load(opt.retrain)
  local states_resume = torch.load(opt.optimState)
  vae_encoder, from_rgb_encoder, vae_decoder, to_rgb, prior, from_rgb, disc = nil, nil, nil, nil, nil, nil, nil
  vae_encoder, from_rgb_encoder, vae_decoder, to_rgb, prior, from_rgb, disc = table.unpack(models_resume)
  optimState_enc, optimState_from_rgb_enc, optimState_dec, optimState_to_rgb, optimState_prior, optimState_from_rgb, optimState_disc = table.unpack(states_resume)
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

  epoch = epoch or 1
  print_freq = opt.print_freq or 1
  print('==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
  local indices = torch.randperm(data.train_im:size(1)):long():split(opt.batchSize)
  indices[#indices] = nil
  local size = #indices
  local N, err_vae_encoder_total, err_vae_decoder_total, err_gan_total = 0, 0.0, 0.0, 0.0
  local reconstruction, inputs, input_im, input_attr
  local gan_update_rate, gan_error_rate, iteration = 0.0, 0.0, 0
  local label_recon, label_sample, label_gan = torch.ones(opt.batchSize):cuda(), torch.ones(opt.batchSize):cuda(), torch.ones(opt.batchSize):cuda()
  local tic = torch.tic()
  local dataTimer = torch.Timer()
  for t, sample in trainLoader:run() do
    N = N + 1
    local timer = torch.Timer()
    local dataTime = dataTimer:time().real

    --[[
          load data (horizontal flip) 
    --]]

    -- load data and augmentation (horizontal flip)
    local input_im, input_attr = sample.input:cuda(), sample.target:cuda()
    local inputs = {input_attr:cuda(), input_im:cuda()}
    collectgarbage()
    
    local reconstruction_sample, z_sample, latent_z, df_do

    --[[
          inference and backprop
          >> from_rgb_encoder > vae_encoder > sampling_z > vae_decoder > to_rgb
          >> prior > KLD
          >> from_rgb > disc
    --]]

    -- encoder > sampling > decoder
    from_rgb_encoder:forward(inputs[2])
    local output_mean_log_var = vae_encoder:forward({inputs[1], from_rgb_encoder.output});
    latent_z = sampling_z(output_mean_log_var):clone()
    vae_decoder:forward({inputs[1], latent_z})
    reconstruction = to_rgb:forward(vae_decoder.output):clone()

    -- prior > KL divergence
    local output_prior = prior:forward(inputs[1])
    KLDerr = KLD:forward(output_mean_log_var, output_prior)
    local dKLD_dtheta = KLD:backward(output_mean_log_var, output_prior)
    for j = 1,4 do dKLD_dtheta[j]:mul(opt.alpha) end
    
    -- loss & backprop
    local Dislikerr = ReconCriterion:forward(reconstruction, inputs[2])
    local df_do = ReconCriterion:backward(reconstruction, inputs[2])

    to_rgb:updateGradInput(vae_decoder.output, df_do)
    vae_decoder:updateGradInput({inputs[1], latent_z}, to_rgb.gradInput)
    local df_dsampler = sampling_z:updateGradInput(output_mean_log_var, vae_decoder.gradInput[2])

    
    -- update vae_encoder, from_rgb_encoder
    vae_encoder:zeroGradParameters(); from_rgb_encoder:zeroGradParameters();
    vae_encoder:backward({inputs[1], from_rgb_encoder.output}, {df_dsampler[1] + dKLD_dtheta[1], df_dsampler[2] + dKLD_dtheta[2]})
    from_rgb_encoder:backward(inputs[2], vae_encoder.gradInput[2])
    
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
    disc:forward({inputs[1], from_rgb.output})
    local gan_err = BCECriterion:forward(disc.output, label_recon)
    BCECriterion:backward(disc.output, label_recon)
    disc:updateGradInput({inputs[1], from_rgb.output}, BCECriterion.gradInput)
    from_rgb:updateGradInput(reconstruction, disc.gradInput[2])
    to_rgb:backward(vae_decoder.output, from_rgb.gradInput * opt.beta + df_do)
    vae_decoder:backward({inputs[1], latent_z}, to_rgb.gradInput)

    -- generate images from random sample
    local output_prior = prior:forward(inputs[1])
    local z_sample = sampling_z:forward(output_prior)
    vae_decoder:forward({inputs[1], z_sample})
    reconstruction_sample = to_rgb:forward(vae_decoder.output):clone()

    -- GAN for sampled images
    from_rgb:forward(reconstruction_sample)
    disc:forward({inputs[1], from_rgb.output})
    local gan_err_sample = BCECriterion:forward(disc.output, label_sample)
    BCECriterion:backward(disc.output, label_sample)
    disc:updateGradInput({inputs[1], from_rgb.output}, BCECriterion.gradInput)
    from_rgb:updateGradInput(reconstruction_sample, disc.gradInput[2])
    to_rgb:backward(vae_decoder.output, from_rgb.gradInput * opt.beta)
    vae_decoder:backward({inputs[1], z_sample}, to_rgb.gradInput)
    
    -- optimize & loss
    optim[opt.optimization](f_vae_decoder, params_dec, optimState_dec)
    optim[opt.optimization](f_to_rgb, params_to_rgb, optimState_to_rgb)
    err_vae_decoder = Dislikerr + gan_err * opt.beta
    err_vae_decoder_total = err_vae_decoder_total + err_vae_decoder

    
    -- from_rgb > disc
    local input_im_gan, label_gan = generate_discriminator_batch(inputs[2], label_gan:fill(1), reconstruction, reconstruction_sample, opt.batchSize, opt.fakeLabel)
    from_rgb:forward(input_im_gan)
    local outputs_gan = disc:forward({inputs[1], from_rgb.output})
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
      disc:backward({inputs[1], from_rgb.output}, BCECriterion.gradInput)
      from_rgb:backward(input_im_gan, disc.gradInput[2])
    end
    optim[opt.optimization .. '_gan'](f_gan, params_disc, optimState_disc)
    optim[opt.optimization .. '_gan'](f_from_rgb, params_from_rgb, optimState_from_rgb)
    err_gan_total = err_gan_total + gan_err
    

    -- print scores
    iteration = iteration + 1
    if t % print_freq == 0 or t == size then
      -- print only every 10 epochs
      print((' | Train: [%d][%d/%d]    Time %.3f  encoder %7.3f (%7.3f)  decoder %7.3f (%7.3f)  gan %7.3f (%7.3f)  gan update rate %.3f (erate %.3f)'):format(
         epoch, t, size, timer:time().real,  err_vae_encoder, err_vae_encoder_total/N, err_vae_decoder, err_vae_decoder_total/N, gan_err, err_gan_total/N, gan_update_rate/iteration, gan_error_rate/iteration))
      gan_update_rate, gan_error_rate, iteration = 0.0, 0.0, 0.0
    end
    reconstruction_sample, z_sample, latent_z, df_do, Dislikerr = nil, nil, nil, nil, nil
    timer:reset()
    collectgarbage()
  end

  print(('Train loss (vae encoder, vae decoder, gan: '..'%.2f ,'..'%.2f ,'..'%.2f '):format(err_vae_encoder_total/N, err_vae_decoder_total/N, err_gan_total/N))
end

function val()
    vae_encoder:evaluate(); vae_decoder:evaluate(); prior:evaluate(); disc:evaluate();
    from_rgb_encoder:evaluate(); to_rgb:evaluate(); from_rgb:evaluate();
    
    local val_attr = data.val_attr[{{1, 128}}]
    local val_im_original = data.val_im[{{1, 128}}]
    local val_attr_tensor = torch.Tensor(val_attr:size(1), val_attr:size(2))
    local val_im = torch.Tensor(val_im_original:size(1), 3, opt.scales[1], opt.scales[1])
    for i = 1, val_im_original:size(1) do
      val_im[i] = opt.preprocess_train(image.scale(val_im_original[i], opt.scales[1], opt.scales[1]))
      val_attr_tensor[i] = val_attr[i]
    end

    val_attr_tensor = val_attr_tensor:cuda()
    val_im = val_im:cuda()

    if epoch == 1 then
      image.save(opt.save .. 'original.png', image.toDisplayTensor(val_im_original:float():add(1):mul(0.5)))
    end
    --(1) test reconstruction 
    from_rgb_encoder:forward(val_im)
    vae_encoder:forward({val_attr_tensor, from_rgb_encoder.output})
    local val_latent_z = sampling_z:forward(vae_encoder.output)
    vae_decoder:forward({val_attr_tensor,val_latent_z})
    local reconstruction_save = to_rgb:forward(vae_decoder.output)
    image.save(opt.save .. 'recon_' .. epoch .. '_LR_' .. opt.LR .. '_alphas_' .. opt.alpha .. '.png', image.toDisplayTensor(reconstruction_save:float():add(1):mul(0.5)))

    --(2) test generation
    val_prior = prior:forward(val_attr_tensor)
    local val_latent_z = sampling_z:forward(val_prior)
    vae_decoder:forward({val_attr_tensor,val_latent_z})
    local generation_val = to_rgb:forward(vae_decoder.output)
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
       torch.save(opt.save .. 'models_' .. epoch .. '.t7', {vae_encoder:clearState(), from_rgb_encoder:clearState(), vae_decoder:clearState(), to_rgb:clearState(), prior:clearState(), from_rgb:clearState(), disc:clearState()})
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
