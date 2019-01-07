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
local vae_encoder, encoder_conv, from_rgb_encoder_layer, vae_decoder, decoder_conv, to_rgb_layer, var_encoder, var_decoder, prior, from_rgb_layer, disc_conv, disc_feature, recon, disc, sampling_z = table.unpack(models)
local KLD, ReconCriterion, BCECriterion, ReconZCriterion = table.unpack(criterions)
local sampling_z2 = sampling_z:clone()
local latent_division = opt.nf/opt.timeStep
local optimState_enc, optimState_from_rgb_enc, optimState_dec, optimState_to_rgb, optimState_var_enc, optimState_var_dec, optimState_prior, optimState_from_rgb, optimState_disc, optimState_recon, optimState_feature

-- reload if previous checkpoint exists
-- otherwise initialize optimStates
if opt.init_weight_from ~= 'none' then
  local models_resume = torch.load(opt.init_weight_from)
  vae_encoder, from_rgb_encoder_layer, vae_decoder, to_rgb_layer, var_encoder, var_decoder, prior, from_rgb_layer, disc_feature, recon, disc = table.unpack(models_resume)
  optimState_enc          = { learningRate = opt.LR_mult*opt.LR, optimize = true, numUpdates = 0}
  optimState_dec          = { learningRate = opt.LR_mult*opt.LR, optimize = true, numUpdates = 0}
  optimState_var_enc      = { learningRate = opt.LR_mult*opt.LR/opt.timeStep, optimize = true, numUpdates = 0}
  optimState_var_dec      = { learningRate = opt.LR_mult*opt.LR/opt.timeStep, optimize = true, numUpdates = 0}
  optimState_prior        = { learningRate = opt.LR_mult*opt.LR, optimize = true, numUpdates = 0}
  optimState_disc         = { learningRate = opt.LR_mult*opt.LR, optimize = true, numUpdates = 0, step = opt.discRatio}
  optimState_recon        = { learningRate = opt.LR_mult*opt.LR, optimize = true, numUpdates = 0}
  optimState_feature      = { learningRate = opt.LR_mult*opt.LR/2, optimize = true, numUpdates = 0}
  optimState_from_rgb     = { learningRate = opt.LR/2, beta1 = opt.beta1, optimize = true, numUpdates = 0}
  optimState_to_rgb       = { learningRate = opt.LR, optimize = true, numUpdates = 0}
  optimState_from_rgb_enc = { learningRate = opt.LR, optimize = true, numUpdates = 0}
elseif opt.retrain ~= 'none' then
  local models_resume = torch.load(opt.retrain)
  if opt.optimState ~= 'none' then
    states_resume = torch.load(opt.optimState)
    optimState_enc, optimState_from_rgb_enc, optimState_dec, optimState_to_rgb, optimState_var_enc, optimState_var_dec, optimState_prior, optimState_from_rgb, optimState_disc, optimState_recon, optimState_feature = table.unpack(states_resume)
  else
    optimState_enc          = { learningRate = opt.LR, optimize = true, numUpdates = 0}
    optimState_dec          = { learningRate = opt.LR, optimize = true, numUpdates = 0}
    optimState_var_enc      = { learningRate = opt.LR/opt.timeStep, optimize = true, numUpdates = 0}
    optimState_var_dec      = { learningRate = opt.LR/opt.timeStep, optimize = true, numUpdates = 0}
    optimState_prior        = { learningRate = opt.LR, optimize = true, numUpdates = 0}
    optimState_disc         = { learningRate = opt.LR/2, beta1 = opt.beta1, optimize = true, numUpdates = 0, step = opt.discRatio}
    optimState_recon        = { learningRate = opt.LR/2, beta1 = opt.beta1, optimize = true, numUpdates = 0, step = opt.discRatio}
    optimState_feature      = { learningRate = opt.LR/2, beta1 = opt.beta1, optimize = true, numUpdates = 0, step = opt.discRatio}
    optimState_from_rgb     = { learningRate = opt.LR/2, beta1 = opt.beta1, optimize = true, numUpdates = 0, step = opt.discRatio}    optimState_to_rgb       = { learningRate = opt.LR, optimize = true, numUpdates = 0}
    optimState_from_rgb_enc = { learningRate = opt.LR, optimize = true, numUpdates = 0}
  end
  vae_encoder, encoder_conv, from_rgb_encoder_layer, vae_decoder, decoder_conv, to_rgb_layer, var_encoder, var_decoder, prior, from_rgb_layer, disc_conv, disc_feature, recon, disc = nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil
  vae_encoder, encoder_conv, from_rgb_encoder_layer, vae_decoder, decoder_conv, to_rgb_layer, var_encoder, var_decoder, prior, from_rgb_layer, disc_conv, disc_feature, recon, disc = table.unpack(models_resume)
  collectgarbage()
else
  optimState_enc          = { learningRate = opt.LR, optimize = true, numUpdates = 0}
  optimState_dec          = { learningRate = opt.LR, optimize = true, numUpdates = 0}
  optimState_var_enc      = { learningRate = opt.LR/opt.timeStep, optimize = true, numUpdates = 0}
  optimState_var_dec      = { learningRate = opt.LR/opt.timeStep, optimize = true, numUpdates = 0}
  optimState_prior        = { learningRate = opt.LR, optimize = true, numUpdates = 0}
  optimState_disc         = { learningRate = opt.LR, optimize = true, numUpdates = 0, step = opt.discRatio}
  optimState_recon        = { learningRate = opt.LR, optimize = true, numUpdates = 0}
  optimState_feature      = { learningRate = opt.LR/2, optimize = true, numUpdates = 0}
  optimState_from_rgb     = { learningRate = opt.LR/2, beta1 = opt.beta1, optimize = true, numUpdates = 0}
  optimState_to_rgb       = { learningRate = opt.LR, optimize = true, numUpdates = 0}
  optimState_from_rgb_enc = { learningRate = opt.LR, optimize = true, numUpdates = 0}
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

-- model params and gradient params
params_enc,           gradParams_enc          = vae_encoder:getParameters()
params_dec,           gradParams_dec          = vae_decoder:getParameters()
params_var_enc,       gradParams_var_enc      = var_encoder:getParameters()
params_var_dec,       gradParams_var_dec      = var_decoder:getParameters()
params_prior,         gradParams_prior        = prior:getParameters()
params_disc,          gradParams_disc         = disc:getParameters()
params_recon,         gradParams_recon        = recon:getParameters()
params_feature,       gradParams_feature      = disc_feature:getParameters()
params_from_rgb,      gradParams_from_rgb     = from_rgb:getParameters()
params_to_rgb,        gradParams_to_rgb       = to_rgb:getParameters()
params_from_rgb_enc,  gradParams_from_rgb_enc = from_rgb_encoder:getParameters()

-- train crVAE-GAN
function train(opt)
  local function f_from_rgb_encoder()       return 0.0, gradParams_from_rgb_enc; end
  local function f_vae_encoder()            return 0.0, gradParams_enc; end
  local function f_var_encoder()            return 0.0, gradParams_var_enc; end
  local function f_prior()                  return 0.0, gradParams_prior; end
  local function f_vae_decoder()            return 0.0, gradParams_dec; end
  local function f_vae_decoder_sample()     return 0.0, gradParams_dec; end
  local function f_var_decoder()            return 0.0, gradParams_var_dec; end
  local function f_var_decoder_sample()     return 0.0, gradParams_var_dec; end
  local function f_to_rgb()                 return 0.0, gradParams_to_rgb; end
  local function f_to_rgb_sample()          return 0.0, gradParams_to_rgb; end
  local function f_disc()                   return 0.0, gradParams_disc; end
  local function f_recon_feature()          return 0.0, gradParams_feature; end
  local function f_recon_feature_sample()   return 0.0, gradParams_feature; end
  local function f_disc_feature()           return 0.0, gradParams_feature; end
  local function f_recon_from_rgb()         return 0.0, gradParams_from_rgb; end
  local function f_recon_from_rgb_sample()  return 0.0, gradParams_from_rgb; end
  local function f_from_rgb()               return 0.0, gradParams_from_rgb; end
  local function f_recon_z()                return 0.0, gradParams_recon; end
  local function f_recon_z_sample()         return 0.0, gradParams_recon; end

  vae_encoder:training(); var_encoder:training(); prior:training();
  vae_decoder:training(); var_decoder:training();
  disc:training(); recon:training(); disc_feature:training();
  from_rgb:training(); to_rgb:training(); from_rgb_encoder:training();
  
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

  local function weight_gradKLD(dKLD_dtheta, latent_division, timeStep, alpha1, alpha2)
    assert(timeStep == 8 or timeStep == 2 or timeStep == 4, 'currently only support time step 8 or 2 or 4')
    if timeStep == 8 then
      dKLD_dtheta[1][{{},{1, latent_division*3},{},{}}]:mul(alpha1)
      dKLD_dtheta[2][{{},{1, latent_division*3},{},{}}]:mul(alpha1)
      dKLD_dtheta[1][{{},{latent_division*3+1, latent_division*8},{},{}}]:mul(alpha2)
      dKLD_dtheta[2][{{},{latent_division*3+1, latent_division*8},{},{}}]:mul(alpha2)
      dKLD_dtheta[3][{{},{1, latent_division*3},{},{}}]:mul(alpha1)
      dKLD_dtheta[4][{{},{1, latent_division*3},{},{}}]:mul(alpha1)
      dKLD_dtheta[3][{{},{latent_division*3+1, latent_division*8},{},{}}]:mul(alpha2)
      dKLD_dtheta[4][{{},{latent_division*3+1, latent_division*8},{},{}}]:mul(alpha2)
    elseif timeStep == 2 then
      dKLD_dtheta[1][{{},{1, latent_division},{},{}}]:mul(alpha1)
      dKLD_dtheta[2][{{},{1, latent_division},{},{}}]:mul(alpha1)
      dKLD_dtheta[1][{{},{latent_division+1, latent_division*2},{},{}}]:mul(alpha2)
      dKLD_dtheta[2][{{},{latent_division+1, latent_division*2},{},{}}]:mul(alpha2)
      dKLD_dtheta[3][{{},{1, latent_division},{},{}}]:mul(alpha1)
      dKLD_dtheta[4][{{},{1, latent_division},{},{}}]:mul(alpha1)
      dKLD_dtheta[3][{{},{latent_division+1, latent_division*2},{},{}}]:mul(alpha2)
      dKLD_dtheta[4][{{},{latent_division+1, latent_division*2},{},{}}]:mul(alpha2)
    else
      dKLD_dtheta[1][{{},{1, latent_division*2},{},{}}]:mul(alpha1)
      dKLD_dtheta[2][{{},{1, latent_division*2},{},{}}]:mul(alpha1)
      dKLD_dtheta[1][{{},{latent_division*2+1, latent_division*4},{},{}}]:mul(alpha2)
      dKLD_dtheta[2][{{},{latent_division*2+1, latent_division*4},{},{}}]:mul(alpha2)
      dKLD_dtheta[3][{{},{1, latent_division*2},{},{}}]:mul(alpha1)
      dKLD_dtheta[4][{{},{1, latent_division*2},{},{}}]:mul(alpha1)
      dKLD_dtheta[3][{{},{latent_division*2+1, latent_division*4},{},{}}]:mul(alpha2)
      dKLD_dtheta[4][{{},{latent_division*2+1, latent_division*4},{},{}}]:mul(alpha2)       
    end
    return dKLD_dtheta
  end

  local function weight_gradReconZ(df_dz_recon, latent_division, timeStep)
    if timeStep == 8 then
      df_dz_recon[{{},{1, latent_division},{},{}}]:mul(0.1);
      df_dz_recon[{{},{latent_division+1, latent_division*3},{},{}}]:mul(1.5);
      df_dz_recon[{{},{latent_division*3+1, latent_division*4},{},{}}]:mul(1.25);
      df_dz_recon[{{},{latent_division*4+1, latent_division*6},{},{}}]:mul(0.1);
      df_dz_recon[{{},{latent_division*6+1, latent_division*8},{},{}}]:mul(1);
    elseif timeStep == 4 then
      df_dz_recon[{{},{1, latent_division},{},{}}]:mul(0.1);
      df_dz_recon[{{},{latent_division+1, latent_division*2},{},{}}]:mul(1);
      df_dz_recon[{{},{latent_division*2+1, latent_division*3},{},{}}]:mul(0.1);
      df_dz_recon[{{},{latent_division*3+1, latent_division*4},{},{}}]:mul(1);
    end
    return df_dz_recon
  end

  local function do_grad_clip(gradParams, grad_clip)
    if grad_clip > 0 then
       gradParams:clamp(-grad_clip, grad_clip)
    end
    return gradParams
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
  local N, KLD_total, Recon_total, ReconZ_total, err_gan_total = 0, 0.0, 0.0, 0.0, 0.0
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

    local input_im, input_attr = sample.input:cuda(), sample.target:cuda()
    local inputs = {input_attr, input_im}
    collectgarbage()


    --[[
          forward for reconstruction
          >> vae_encoder > var_encoder > sampling_z > var_decoder > vae_decoder
          >> prior > KLD
          >> reconstruction > disc_feature > recon, disc
    --]]

    -- encoder (vae, var) > sampling > decoder (var, vae)
    from_rgb_encoder:forward(inputs[2])
    merge_from_rgb_encoder:forward(from_rgb_encoder.output)
    local output_mean_log_var_before = vae_encoder:forward({inputs[1], merge_from_rgb_encoder.output});
    local output_mean_log_var = var_encoder:forward(output_mean_log_var_before)
    local latent_z = sampling_z:forward(output_mean_log_var)
    local output_decoder_before = var_decoder:forward({inputs[1], latent_z})
    vae_decoder:forward(output_decoder_before)
    to_rgb:forward(vae_decoder.output)
    local reconstruction = merge_to_rgb:forward(to_rgb.output):clone()

    -- prior > KL divergence
    local output_prior = prior:forward(inputs[1])
    local KLDerr = KLD:forward(output_mean_log_var, output_prior)
    local dKLD_dtheta = KLD:backward(output_mean_log_var,output_prior)
    dKLD_dtheta = weight_gradKLD(dKLD_dtheta, latent_division, opt.timeStep, opt.alpha1, opt.alpha2)
    KLD_total = KLD_total + KLDerr

    -- loss & backprop
    local Dislikerr = ReconCriterion:forward(reconstruction, inputs[2])
    local df_do = ReconCriterion:backward(reconstruction, inputs[2])
    Recon_total = Recon_total + Dislikerr

    -- feature reconstruction
    from_rgb:forward(reconstruction)
    merge_from_rgb:forward(from_rgb.output)
    local feature = disc_feature:forward({inputs[1], merge_from_rgb.output})
    local z_recon = recon:forward(feature)
    local ZDislikerr = ReconZCriterion:forward(z_recon, output_decoder_before)
    local df_dz_recon = ReconZCriterion:backward(z_recon, output_decoder_before)
    ReconZ_total = ReconZ_total + ZDislikerr*0.5


    --[[
          backward from reconstruction
    --]]

    -- backward (recon, disc_feature)
    df_dz_recon = weight_gradReconZ(df_dz_recon, latent_division, opt.timeStep)
    
    recon:zeroGradParameters(); disc_feature:zeroGradParameters(); from_rgb:zeroGradParameters();
    recon:backward(feature, df_dz_recon)
    disc_feature:backward({inputs[1], merge_from_rgb.output}, recon.gradInput)
    merge_from_rgb:backward(from_rgb.output, disc_feature.gradInput[2])
    from_rgb:backward(reconstruction, merge_from_rgb.gradInput)
    local df_recon_feature = from_rgb.gradInput:clone()

    optim[opt.optimization](f_recon_z, params_recon, optimState_recon)
    optim[opt.optimization](f_recon_feature, params_feature, optimState_feature)
    optim[opt.optimization](f_recon_from_rgb, params_from_rgb, optimState_from_rgb)

    -- backward (vae_decoder, var_decoder)
    to_rgb:zeroGradParameters(); vae_decoder:zeroGradParameters(); var_decoder:zeroGradParameters();

    -- local feature = disc_feature:forward({inputs[1], reconstruction})
    disc:forward(feature)
    local gan_err = BCECriterion:forward(disc.output, label_recon)
    BCECriterion:backward(disc.output, label_recon)
    disc:updateGradInput(disc_feature.output, BCECriterion.gradInput)
    disc_feature:updateGradInput({inputs[1], merge_from_rgb.output}, disc.gradInput)
    merge_from_rgb:updateGradInput(from_rgb.output, disc_feature.gradInput[2])
    from_rgb:updateGradInput(reconstruction, merge_from_rgb.gradInput)
    merge_to_rgb:backward(to_rgb.output, df_do + df_recon_feature*opt.kappa + from_rgb.gradInput*opt.beta)
    to_rgb:backward(vae_decoder.output, merge_to_rgb.gradInput)
    vae_decoder:backward(var_decoder.output, to_rgb.gradInput);
    var_decoder:backward({inputs[1], latent_z}, vae_decoder.gradInput);
    gradParams_var_dec = do_grad_clip(gradParams_var_dec, opt.grad_clip)
    
    optim[opt.optimization](f_to_rgb, params_to_rgb, optimState_to_rgb)
    optim[opt.optimization](f_vae_decoder, params_dec, optimState_dec)
    optim[opt.optimization](f_var_decoder, params_var_dec, optimState_var_dec)

    -- backward (prior, var_encoder, vae_encoder)
    prior:zeroGradParameters(); var_encoder:zeroGradParameters(); vae_encoder:zeroGradParameters(); from_rgb_encoder:zeroGradParameters();
    local df_dsampler = sampling_z:updateGradInput(output_mean_log_var, var_decoder.gradInput[2])
    prior:backward(inputs[1], {dKLD_dtheta[3], dKLD_dtheta[4]})
    var_encoder:backward(output_mean_log_var_before, {dKLD_dtheta[1] + df_dsampler[1], dKLD_dtheta[2] + df_dsampler[2]})
    gradParams_var_enc = do_grad_clip(gradParams_var_enc, opt.grad_clip)
    vae_encoder:backward({inputs[1], merge_from_rgb_encoder.output}, var_encoder.gradInput);
    merge_from_rgb_encoder:backward(from_rgb_encoder.output, vae_encoder.gradInput[2])
    from_rgb_encoder:backward(inputs[2], merge_from_rgb_encoder.gradInput)
    
    optim[opt.optimization](f_prior, params_prior, optimState_prior)
    optim[opt.optimization](f_var_encoder, params_var_enc, optimState_var_enc)
    optim[opt.optimization](f_vae_encoder, params_enc, optimState_enc)
    optim[opt.optimization](f_from_rgb_encoder, params_from_rgb_enc, optimState_from_rgb_enc)


    --[[
          forward to generate random samples
          >> prior > sampling_z > var_decoder > vae_decoder
          >> reconstruction_sample > disc_feature > recon, disc
    --]]
    
    local output_prior_sample = prior:forward(inputs[1])
    local latent_z_sample = sampling_z2:forward(output_prior_sample)
    local output_decoder_before_sample = var_decoder:forward({inputs[1], latent_z_sample})
    vae_decoder:forward(output_decoder_before_sample)
    to_rgb:forward(vae_decoder.output)
    local reconstruction_sample = merge_to_rgb:forward(to_rgb.output):clone()

    -- feature reconstruction
    from_rgb:forward(reconstruction_sample)
    merge_from_rgb:forward(from_rgb.output)
    local feature_sample = disc_feature:forward({inputs[1], merge_from_rgb.output})
    local z_recon_sample = recon:forward(feature_sample)
    local ZDislikerr_sample = ReconZCriterion:forward(z_recon_sample, output_decoder_before_sample)
    local df_dz_recon_sample = ReconZCriterion:backward(z_recon_sample, output_decoder_before_sample)
    ReconZ_total = ReconZ_total + ZDislikerr_sample*0.5


    --[[
          backward from generation
    --]]

    -- backward (recon, disc_feature)
    df_dz_recon_sample = weight_gradReconZ(df_dz_recon_sample, latent_division, opt.timeStep)

    recon:zeroGradParameters(); disc_feature:zeroGradParameters(); from_rgb:zeroGradParameters();
    recon:backward(feature_sample, df_dz_recon_sample)
    disc_feature:backward({inputs[1], merge_from_rgb.output}, recon.gradInput)
    merge_from_rgb:backward(from_rgb.output, disc_feature.gradInput[2])
    from_rgb:backward(reconstruction_sample, merge_from_rgb.gradInput)
    local df_recon_feature_sample = from_rgb.gradInput:clone()

    optim[opt.optimization](f_recon_z_sample, params_recon, optimState_recon)
    optim[opt.optimization](f_recon_feature_sample, params_feature, optimState_feature)
    optim[opt.optimization](f_recon_from_rgb_sample, params_from_rgb, optimState_from_rgb)
    
    -- backward (vae_decoder, var_decoder)
    to_rgb:zeroGradParameters(); vae_decoder:zeroGradParameters(); var_decoder:zeroGradParameters();

    -- local feature_sample = disc_feature:forward({inputs[1], reconstruction_sample})
    disc:forward(feature_sample)
    local gan_err = BCECriterion:forward(disc.output, label_sample)
    BCECriterion:backward(disc.output, label_sample)
    disc:updateGradInput(disc_feature.output, BCECriterion.gradInput)
    disc_feature:updateGradInput({inputs[1], merge_from_rgb.output}, disc.gradInput)
    merge_from_rgb:updateGradInput(from_rgb.output, disc_feature.gradInput[2])
    from_rgb:updateGradInput(reconstruction_sample, merge_from_rgb.gradInput)
    merge_to_rgb:backward(to_rgb.output, df_recon_feature_sample*opt.kappa + from_rgb.gradInput*opt.beta)
    to_rgb:backward(vae_decoder.output, merge_to_rgb.gradInput);
    vae_decoder:backward(var_decoder.output, to_rgb.gradInput)
    var_decoder:backward({inputs[1], latent_z_sample}, vae_decoder.gradInput);
    gradParams_var_dec = do_grad_clip(gradParams_var_dec, opt.grad_clip)

    optim[opt.optimization](f_to_rgb_sample, params_to_rgb, optimState_to_rgb)
    optim[opt.optimization](f_vae_decoder_sample, params_dec, optimState_dec)
    optim[opt.optimization](f_var_decoder_sample, params_var_dec, optimState_var_dec)

    
    --[[
          forward to update discriminator
          >> reconstruction, reconstruction_sample, mismatch, real > disc_feature > disc
    --]]

    -- update discriminator
    local input_im_gan, label_gan = generate_discriminator_batch(inputs[2], label_gan:fill(1), reconstruction, reconstruction_sample, opt.batchSize, opt.fakeLabel)
    from_rgb:forward(input_im_gan)
    merge_from_rgb:forward(from_rgb.output)
    local outputs_feature = disc_feature:forward({inputs[1], merge_from_rgb.output})
    local outputs_disc = disc:forward(outputs_feature)
    local gan_err = BCECriterion:forward(outputs_disc, label_gan)
    err_gan_total = err_gan_total + gan_err

    -- compute disc error rate
    local predictions = torch.round(outputs_disc):squeeze()
    local top1 = torch.eq(predictions:float(), label_gan:float()):sum()
    local gan_erate = 1 - top1/outputs_disc:size(1)
    gan_error_rate = gan_error_rate + gan_erate
    optimState_disc.optimize = true    
    if gan_erate < opt.margin then
      optimState_disc.optimize = false
    else
      -- update discriminator, 
      gan_update_rate = gan_update_rate + 1
      disc:zeroGradParameters(); disc_feature:zeroGradParameters(); from_rgb:zeroGradParameters();
      BCECriterion:backward(outputs_disc, label_gan)
      disc:backward(outputs_feature, BCECriterion.gradInput)
      disc_feature:backward({inputs[1], merge_from_rgb.output}, disc.gradInput)
      merge_from_rgb:backward(from_rgb.output, disc_feature.gradInput[2])
      from_rgb:backward(input_im_gan, merge_from_rgb.gradInput)
    end
    optim[opt.optimization .. '_gan'](f_disc, params_disc, optimState_disc)
    optim[opt.optimization](f_disc_feature, params_feature, optimState_feature)
    optim[opt.optimization](f_from_rgb, params_from_rgb, optimState_from_rgb)

    -- print scores
    iteration = iteration + 1
    if t % print_freq == 1 or t == size then
      print((' | Train: [%d][%d/%d]    Time %.3f (%.3f)  KL %7.3f (%7.3f)  recon %7.3f (%7.3f)  recon_Z %7.3f (%7.3f)  gan %7.3f (%7.3f)  gan update rate %.3f (erate %.3f)'):format(
          epoch, t, size, timer:time().real, dataTime, KLDerr, KLD_total/N, Dislikerr, Recon_total/N, ZDislikerr, ReconZ_total/N, gan_err, err_gan_total/N, gan_update_rate/iteration, gan_error_rate/iteration))
      gan_update_rate, gan_error_rate, iteration = 0.0, 0.0, 0.0
    end
    timer:reset()
    dataTimer:reset()
    collectgarbage()
  end

  print(('Train loss (KLD, Recon, ReconZ: %.3f, %.3f, %.3f'):format(KLD_total/N, Recon_total/N, ReconZ_total/N))
end

function val(opt)
  vae_encoder:evaluate(); vae_decoder:evaluate(); var_encoder:evaluate(); var_decoder:evaluate(); prior:evaluate();
  disc:evaluate(); disc_feature:evaluate(); recon:evaluate();
  from_rgb_encoder:evaluate(); from_rgb:evaluate(); to_rgb:evaluate();

  if (val_im == nil) and (val_attr == nil) then
    for n, sample in valLoader:run() do
      if n == 1 then
        val_im, val_attr = sample.input, sample.target
        break;
      end
    end
    val_im, val_attr = val_im:cuda(), val_attr:cuda()
  end

  --(0) save the original image
  if epoch == 1 then
    image.save(opt.save .. 'original.png', image.toDisplayTensor(val_im:add(1):mul(0.5)))
  end
  --(1) test reconstruction 
  from_rgb_encoder:forward(val_im)
  vae_encoder:forward({val_attr, from_rgb_encoder.output[2]})
  var_encoder:forward(vae_encoder.output)
  local val_latent_z = sampling_z:forward(var_encoder.output)
  var_decoder:forward({val_attr,val_latent_z})
  vae_decoder:forward(var_decoder.output)
  reconstruction_val = to_rgb:forward(vae_decoder.output)[2]
  reconstruction_val = reconstruction_val:float()
  image.save(opt.save .. 'recon_' .. epoch .. '_LR_' .. opt.LR .. '_alphas_' .. opt.alpha1 .. '_' .. opt.alpha2 .. '.png', image.toDisplayTensor(reconstruction_val:add(1):mul(0.5)))

  --(2) test generation
  val_prior = prior:forward(val_attr)
  local val_latent_z = sampling_z:forward(val_prior)
  var_decoder:forward({val_attr,val_latent_z})
  vae_decoder:forward(var_decoder.output)
  local generation_val = to_rgb(vae_decoder.output)[2]
  generation_val = generation_val:float()
  generation_val:add(1):mul(0.5)
  image.save( opt.save .. 'gen_' .. epoch .. '_LR_' .. opt.LR .. '_alphas_' .. opt.alpha1 .. '_' .. opt.alpha2 .. '.png', image.toDisplayTensor(generation_val))
  params_enc,           gradParams_enc          = nil, nil
  params_dec,           gradParams_dec          = nil, nil
  params_var_enc,       gradParams_var_enc      = nil, nil
  params_var_dec,       gradParams_var_dec      = nil, nil
  params_prior,         gradParams_prior        = nil, nil
  params_disc,          gradParams_disc         = nil, nil
  params_recon,         gradParams_recon        = nil, nil
  params_feature,       gradParams_feature      = nil, nil
  params_from_rgb_enc,  gradParams_from_rgb_enc = nil, nil
  params_from_rgb,      gradParams_from_rgb     = nil, nil
  params_to_rgb,        gradParams_to_rgb       = nil, nil
  if epoch % opt.epochStep == 0 then
    torch.save(opt.save .. 'models_' .. epoch .. '.t7', {vae_encoder:clearState(), encoder_conv:clearState(), from_rgb_encoder_layer:clearState(), vae_decoder:clearState(), decoder_conv:clearState(), to_rgb_layer:clearState(), var_encoder:clearState(), var_decoder:clearState(), prior:clearState(), from_rgb_layer:clearState(), disc_conv:clearState(), disc_feature:clearState(), recon:clearState(), disc:clearState()})
    torch.save(opt.save .. 'states_' .. epoch .. '.t7', {optimState_enc, optimState_from_rgb_enc, optimState_dec, optimState_to_rgb, optimState_var_enc, optimState_var_dec, optimState_prior, optimState_from_rgb, optimState_disc, optimState_recon, optimState_feature})
  end
  if epoch % opt.step == 0 then
    optimState_enc.learningRate           = optimState_enc.learningRate*opt.decayLR
    optimState_dec.learningRate           = optimState_dec.learningRate*opt.decayLR
    optimState_var_enc.learningRate       = optimState_var_enc.learningRate*opt.decayLR
    optimState_var_dec.learningRate       = optimState_var_dec.learningRate*opt.decayLR
    optimState_prior.learningRate         = optimState_prior.learningRate*opt.decayLR
    optimState_disc.learningRate          = optimState_disc.learningRate*opt.decayLR
    optimState_recon.learningRate         = optimState_recon.learningRate*opt.decayLR
    optimState_feature.learningRate       = optimState_feature.learningRate*opt.decayLR
    optimState_from_rgb.learningRate      = optimState_from_rgb.learningRate*opt.decayLR
    optimState_from_rgb_enc.learningRate  = optimState_from_rgb_enc.learningRate*opt.decayLR
    optimState_to_rgb.learningRate        = optimState_to_rgb.learningRate*opt.decayLR
  end
  params_enc,           gradParams_enc            = vae_encoder:getParameters()
  params_dec,           gradParams_dec            = vae_decoder:getParameters()
  params_var_enc,       gradParams_var_enc        = var_encoder:getParameters()
  params_var_dec,       gradParams_var_dec        = var_decoder:getParameters()
  params_prior,         gradParams_prior          = prior:getParameters()
  params_disc,          gradParams_disc           = disc:getParameters()
  params_recon,         gradParams_recon          = recon:getParameters()
  params_feature,       gradParams_feature        = disc_feature:getParameters()
  params_from_rgb_enc,  gradParams_from_rgb_enc   = from_rgb_encoder:getParameters()
  params_from_rgb,      gradParams_from_rgb       = from_rgb:getParameters()
  params_to_rgb,        gradParams_to_rgb         = to_rgb:getParameters()
  print('Saved image to: ' .. opt.save)
end

function generate(opt)
  print('Saving Generateions to: ' .. opt.testSample)
  os.execute('mkdir -p ' .. opt.testSample)
  vae_encoder:evaluate(); vae_decoder:evaluate(); var_encoder:evaluate(); var_decoder:evaluate(); prior:evaluate();
  to_rgb:evaluate();
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
    var_decoder:forward({inputs_attr,val_latent_z})
    vae_decoder:forward(var_decoder.output)
    upsample:forward(vae_decoder.output)
    decoder_conv:forward(upsample.output)
    local generation_val = to_rgb_layer(decoder_conv.output)
    generation_val = generation_val:float()
    generation_val:add(1):mul(0.5)
    image.save(paths.concat(opt.testSample, t .. '.png'), image.toDisplayTensor(generation_val[1]))
  end
end
