local function ConcatAct()
   local model = nn.ConcatTable()
   model:add(nn.Identity())
   model:add(nn.MulConstant(-1))
   return model
end

local function createModel(opt)
   local from_rgb_encoder = nn.Sequential()
   local from_rgb = nn.Sequential()
   local to_rgb = nn.Sequential()
   local encoder_total = nn.Sequential()
   local decoder = nn.Sequential()
   local var_encoder = nn.ParallelTable()
   local var_decoder = nn.Sequential()
   local gan = nn.Sequential()
   local disc = nn.Sequential()
   local gan_feature = nn.Sequential()
   local recon = nn.Sequential()
   local prior = nn.Sequential()
   
   local baseChannels = opt.baseChannels
   local w = opt.latentDims[1]
   local z = opt.nf
   local eps = opt.eps
   local mom = opt.mom
   local time_step = opt.timeStep
   local attribute_dim = opt.attrDim

   if opt.dataset == 'celeba' then
      -----------------------------------
      -- Encoder (Inference network) ----
      -- convolution net -> LSTM layer --
      -----------------------------------

      from_rgb_encoder:add(ConcatAct())
      from_rgb_encoder:add(nn.JoinTable(2))
      from_rgb_encoder:add(cudnn.ReLU(true))
      from_rgb_encoder:add(cudnn.SpatialConvolution(6, baseChannels, 1, 1, 1, 1, 0, 0))
      from_rgb_encoder:add(nn.SpatialBatchNormalization(baseChannels, eps, mom))
      from_rgb_encoder:add(cudnn.ReLU(false))

      local encoder = nn.Sequential()

      -- conv1: 64 x 64 --> 32 x 32
      encoder:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 5, 5, 2, 2, 2, 2))
      encoder:add(nn.SpatialBatchNormalization(baseChannels, eps, mom))
      encoder:add(ConcatAct())
      encoder:add(nn.JoinTable(2))
      encoder:add(cudnn.ReLU(true))
      baseChannels = baseChannels * 2

      -- conv2: 32 x 32 --> 16 x 16
      encoder:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 3, 3, 2, 2, 1, 1))
      encoder:add(nn.SpatialBatchNormalization(baseChannels, eps, mom))
      encoder:add(ConcatAct())
      encoder:add(nn.JoinTable(2))
      encoder:add(cudnn.ReLU(true))
      baseChannels = baseChannels * 2

      -- conv3-1, conv3-2: 16 x 16 --> 8 x 8
      encoder:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 3, 3, 2, 2, 1, 1))
      encoder:add(nn.SpatialBatchNormalization(baseChannels, eps, mom))
      encoder:add(ConcatAct())
      encoder:add(nn.JoinTable(2))
      encoder:add(cudnn.ReLU(true))
      baseChannels = baseChannels * 2
      encoder:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 3, 3, 1, 1, 1, 1))
      encoder:add(nn.SpatialBatchNormalization(baseChannels, eps, mom))
      encoder:add(nn.LeakyReLU(0.1))

      -- conv4-1, conv4-2: 8 x 8 --> 4 x 4
      encoder:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 3, 3, 2, 2, 1, 1))
      encoder:add(nn.SpatialBatchNormalization(baseChannels, eps, mom))
      encoder:add(ConcatAct())
      encoder:add(nn.JoinTable(2))
      encoder:add(cudnn.ReLU(true))
      encoder:add(cudnn.SpatialConvolution(baseChannels*2, baseChannels, 3, 3, 1, 1, 1, 1))
      encoder:add(nn.SpatialBatchNormalization(baseChannels, eps, mom))
      encoder:add(nn.LeakyReLU(0.1))
      encoder:add(nn.View(time_step, baseChannels/time_step, w, w):setNumInputDims(4))

      local latent_division = z/time_step
      local encoder_attribute = nn.Sequential()
      encoder_attribute:add(nn.Linear(attribute_dim, w*w*latent_division))
      encoder_attribute:add(nn.LeakyReLU(0.1))
      encoder_attribute:add(nn.View(latent_division,w,w):setNumInputDims(2))
      encoder_attribute:add(nn.Replicate(time_step,2))
      local encoder_connection = nn.ParallelTable()
      encoder_connection:add(encoder_attribute)
      encoder_connection:add(encoder)

      encoder_total:add(encoder_connection)
      encoder_total:add(nn.JoinTable(3))
      encoder_total:add(nn.View(z+baseChannels,w,w):setNumInputDims(5))
      -- LSTM layer for Channel-Recurrency
      -- mean path: convolution followed by bias subtraction (mu_0 in figure 2(c))
      local mean_shift = nn.Sequential()
      local add_size = torch.LongStorage(3)
      add_size[1] = z
      add_size[2] = w
      add_size[3] = w
      mean_shift:add(cudnn.SpatialConvolution(baseChannels+z, z, 3, 3, 1, 1, 1, 1))
      mean_shift:add(nn.Add(add_size))
      mean_shift:get(2):reset(0.01)

      local mean_logvar_before = nn.ConcatTable()
      mean_logvar_before:add(mean_shift)
      mean_logvar_before:add(nn.Identity())
      encoder_total:add(mean_logvar_before)

      -- variance path: 
      -- 1. 4 x 4 x baseChannels is divided into time_step blocks of size 4 x 4 x baseChannels/time_step
      -- 2. channel-recurrency via LSTM followed by block-wise FC layer to generate \sigma^{rnn} in figure 2(c)
      local input_division = baseChannels/time_step
      local latent_division = z/time_step
      local var_transform = nn.Sequential()
      local LSTM_encoder = cudnn.LSTM(w*w*input_division+w*w*latent_division, w*w*input_division, 2, true)
      for i = 0, LSTM_encoder.numLayers - 1 do
           local params = getParams(LSTM_encoder, i, 1)
           params.bias:fill(1)
      end
      var_transform:add(nn.View(time_step, w*w*input_division+w*w*latent_division):setNumInputDims(3))
      var_transform:add(nn.Contiguous())
      var_transform:add(LSTM_encoder)
      var_transform:add(nn.Contiguous())
      --var_transform:add(nn.View(opt.batchSize*time_step, w*w*input_division))
      var_transform:add(nn.View(-1, w*w*input_division):setNumInputDims(3))
      var_transform:add(nn.Contiguous())
      var_transform:add(nn.BatchNormalization(w*w*input_division))
      var_transform:add(nn.Linear(w*w*input_division, w*w*latent_division))
      var_transform:add(nn.View(z, w, w):setNumInputDims(2))
      var_transform:add(nn.Contiguous())

      var_encoder:add(nn.Identity())
      var_encoder:add(var_transform)

      -------------------------------------
      -- Decoder (Generation network)    --
      -- LSTM layer -> deconvolution net --
      -------------------------------------
      -- Channel-Recurrent Decoder that transforms z_{i} into u_{i} in figure 2(c)

      local decoder_attribute = nn.Sequential()
      decoder_attribute:add(nn.Linear(attribute_dim, w*w*latent_division))
      decoder_attribute:add(nn.LeakyReLU(0.1))
      decoder_attribute:add(nn.Replicate(time_step,2))
      local decoder_connection = nn.ParallelTable()
      decoder_connection:add(decoder_attribute)
      decoder_connection:add(nn.View(z/latent_division, w * w * latent_division):setNumInputDims(3))

      local LSTM_decoder = cudnn.LSTM(w*w*latent_division*2, w*w*input_division, 2, true)
      for i = 0, LSTM_decoder.numLayers - 1 do
           local params = getParams(LSTM_decoder, i, 1)
           params.bias:fill(1)
      end
      var_decoder:add(decoder_connection)
      var_decoder:add(nn.JoinTable(3))
      var_decoder:add(nn.Contiguous())
      var_decoder:add(LSTM_decoder)
      var_decoder:add(nn.Contiguous())
      var_decoder:add(nn.View(-1, w*w*input_division):setNumInputDims(3))
      var_decoder:add(nn.Contiguous())
      var_decoder:add(nn.Linear(w*w*input_division, w*w*latent_division))
      var_decoder:add(nn.View(-1, z, w, w))
      var_decoder:add(nn.Contiguous())

      -- add bias back (figure 2(c))
      local add_size = torch.LongStorage(3)
      add_size[1] = z
      add_size[2] = w
      add_size[3] = w
      decoder:add(nn.Add(add_size))
      decoder:get(1):reset(0.01)

      -- deconv5: 4 x 4 --> 4 x 4
      decoder:add(cudnn.SpatialConvolution(z, baseChannels, 3, 3, 1, 1, 1, 1))
      decoder:add(nn.SpatialBatchNormalization(baseChannels, eps, mom))

      -- deconv4-1, deconv4-2: 4 x 4 --> 8 x 8
      decoder:add(cudnn.SpatialFullConvolution(baseChannels, baseChannels, 4, 4, 2, 2, 1, 1))
      decoder:add(nn.SpatialBatchNormalization(baseChannels, eps, mom))
      decoder:add(nn.LeakyReLU(0.1))
      decoder:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 3, 3, 1, 1, 1, 1))
      decoder:add(nn.SpatialBatchNormalization(baseChannels, eps, mom))
      decoder:add(nn.LeakyReLU(0.1))
      baseChannels = baseChannels/2

      -- deconv3: 8 x 8 --> 16 x 16
      decoder:add(cudnn.SpatialFullConvolution(baseChannels*2, baseChannels, 4, 4, 2, 2, 1, 1))
      decoder:add(nn.SpatialBatchNormalization(baseChannels, eps, mom))
      decoder:add(nn.LeakyReLU(0.1))
      baseChannels = baseChannels/2

      -- deconv2: 16 x 16 --> 32 x 32
      decoder:add(cudnn.SpatialFullConvolution(baseChannels*2, baseChannels, 4, 4, 2, 2, 1, 1))
      decoder:add(nn.SpatialBatchNormalization(baseChannels, eps, mom))
      decoder:add(nn.LeakyReLU(0.1))

      -- deconv1: 32 x 32 --> 64 x 64
      decoder:add(cudnn.SpatialFullConvolution(baseChannels, baseChannels, 4, 4, 2, 2, 1, 1))
      decoder:add(nn.SpatialBatchNormalization(baseChannels, eps, mom))
      decoder:add(nn.LeakyReLU(0.1))

      -- tanH: 64 x 64 --> 64 x 64
      decoder:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 3, 3, 1, 1, 1, 1))
      decoder:add(nn.SpatialBatchNormalization(baseChannels, eps, mom))
      decoder:add(nn.LeakyReLU(0.1))

      to_rgb:add(cudnn.SpatialConvolution(baseChannels, 3, 1, 1, 1, 1, 0, 0))
      to_rgb:add(nn.Tanh())

      prior:add(nn.Linear(attribute_dim, w*w*z))
      prior:add(nn.LeakyReLU(0.1))
      local mean_logvar = nn.ConcatTable()
      mean_logvar:add(nn.Sequential():add(nn.Linear(w*w*z, w*w*z)):add(nn.View(z, w, w):setNumInputDims(2))) -- mean
      mean_logvar:add(nn.Sequential():add(nn.Linear(w*w*z, w*w*z)):add(nn.View(z, w, w):setNumInputDims(2))) -- variance
      prior:add(mean_logvar)

   
      baseChannels = baseChannels/2
      from_rgb:add(cudnn.SpatialConvolution(3, baseChannels, 1, 1, 1, 1, 0, 0))
      from_rgb:add(cudnn.ReLU(false))

      gan:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 5, 5, 1, 1, 2, 2))
      gan:add(cudnn.SpatialMaxPooling(2, 2))
      gan:add(cudnn.ReLU(true))
      gan:add(nn.SpatialDropout(0.5, true))

      -- conv2: 32 x 32 --> 16 x 16
      gan:add(cudnn.SpatialConvolution(baseChannels, baseChannels*2, 5, 5, 1, 1, 2, 2))
      gan:add(cudnn.SpatialMaxPooling(2, 2))
      gan:add(cudnn.ReLU(true))
      gan:add(nn.SpatialDropout(0.5, true))

      -- conv3: 16 x 16 --> 8 x 8
      gan:add(cudnn.SpatialConvolution(baseChannels*2, baseChannels*4, 5, 5, 1, 1, 2, 2))
      gan:add(cudnn.SpatialMaxPooling(2, 2))
      gan:add(cudnn.ReLU(true))
      gan:add(nn.SpatialDropout(0.5, true))

      -- conv4: 8 x 8 --> 4 x 4
      gan:add(cudnn.SpatialConvolution(baseChannels*4, baseChannels*8, 3, 3, 1, 1, 1, 1))
      gan:add(cudnn.SpatialMaxPooling(2, 2))
      gan:add(cudnn.ReLU(true))
      gan:add(nn.SpatialDropout(0.5, true))
      gan:add(nn.View(time_step, w * w * baseChannels * 8 / time_step):setNumInputDims(3))


      -- Fully-connected: 4 x 4 x (baseChannels x 8) --> 4 x 4 x (baseChannels x 2)
      -- followed by batch discrimination

      local gan_attribute = nn.Sequential()
      gan_attribute:add(nn.Linear(attribute_dim, w*w*latent_division))
      gan_attribute:add(nn.LeakyReLU(0.1))
      gan_attribute:add(nn.Replicate(time_step,2))
      local gan_connection = nn.ParallelTable()
      gan_connection:add(gan_attribute)
      gan_connection:add(gan)

      gan_feature:add(gan_connection)
      gan_feature:add(nn.JoinTable(3))
      gan_feature:add(nn.View(z+baseChannels*8,w,w):setNumInputDims(2))


      recon:add(cudnn.SpatialConvolution(z+baseChannels*8, baseChannels*2, 3, 3, 1, 1, 1, 1))
      recon:add(nn.SpatialBatchNormalization(baseChannels*2, eps, mom))
      recon:add(nn.LeakyReLU(0.2, true))
      recon:add(cudnn.SpatialConvolution(baseChannels*2, z, 1, 1))
      recon:add(nn.SpatialBatchNormalization(z, eps, mom))
      recon:add(nn.LeakyReLU(0.1, true))
      recon:add(nn.View(z*w*w))
      recon:add(nn.Linear(z*w*w, z*w*w))
      recon:add(nn.View(z,w,w))
      local add_size_recon = torch.LongStorage(3)
      add_size_recon[1] = z
      add_size_recon[2] = w
      add_size_recon[3] = w
      recon:add(nn.Add(add_size_recon))
      recon:get(#recon.modules):reset(0.01)

      disc:add(nn.View(w*w*baseChannels*8+w*w*z))
      disc:add(nn.Linear(w*w*baseChannels*8+w*w*z, w*w*baseChannels))
      disc:add(nn.Normalize(2))
      disc:add(nn.Dropout(0.5, true))
      disc:add(cudnn.ReLU(true))
      disc:add(nn.BatchDiscrimination(w*w*baseChannels, 100, 5))
      disc:add(nn.Linear(w*w*baseChannels+100, 1))
      disc:add(nn.Sigmoid())
   else
      error('invalid dataset: ' .. opt.dataset)
   end

   local function ConvInit(name)
      for k,v in pairs(encoder_total:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:uniform(-1*math.sqrt(1/n),math.sqrt(1/n))
         if not opt.bias then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
      for k,v in pairs(decoder:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:uniform(-1*math.sqrt(1/n),math.sqrt(1/n))
         if not opt.bias then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
      for k,v in pairs(var_encoder:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:uniform(-1*math.sqrt(1/n),math.sqrt(1/n))
         if not opt.bias then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
      for k,v in pairs(var_decoder:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:uniform(-1*math.sqrt(1/n),math.sqrt(1/n))
         if not opt.bias then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
      for k, v in pairs(disc:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:uniform(-1*math.sqrt(1/n), math.sqrt(1/n))
         if not opt.bias then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
      for k, v in pairs(from_rgb_encoder:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:uniform(-1*math.sqrt(1/n), math.sqrt(1/n))
         if not opt.bias then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
      for k, v in pairs(to_rgb:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:uniform(-1*math.sqrt(1/n), math.sqrt(1/n))
         if not opt.bias then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
      for k, v in pairs(from_rgb:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:uniform(-1*math.sqrt(1/n), math.sqrt(1/n))
         if not opt.bias then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(encoder_total:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
      for k,v in pairs(decoder:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
      for k, v in pairs(disc:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
      for k, v in pairs(from_rgb_encoder:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('cudnn.SpatialFullConvolution')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')

   for k,v in pairs(encoder_total:findModules('nn.Linear')) do
      v.bias:zero()
   end
   for k,v in pairs(decoder:findModules('nn.Linear')) do
      v.bias:zero()
   end
   --[[ NO DISCRIMINATOR FOR NOW 
   for k,v in pairs(gan:findModules('nn.Linear')) do
      v.bias:zero()
   end
   --]]
   for k,v in pairs(var_encoder:findModules('nn.Linear')) do
      v.bias:zero()
   end
   for k,v in pairs(var_decoder:findModules('nn.Linear')) do
      v.bias:zero()
   end
   for k, v in pairs(disc:findModules('nn.Linear')) do
      v.bias:zero()
   end

   if opt.cudnn == 'deterministic' then
      encoder_total:apply(function(m)
         if m.setMode then m:setMode(1, 1, 1) end
      end)
      var_encoder:apply(function(m)
         if m.setMode then m:setMode(1, 1, 1) end
      end)
      var_decoder:apply(function(m)
         if m.setMode then m:setMode(1, 1, 1) end
      end)
      from_rgb_encoder:apply(function(m)
         if m.setMode then m:setMode(1, 1, 1) end
      end)
      decoder:apply(function(m)
         if m.setMode then m:setMode(1, 1, 1) end
      end)
      to_rgb:apply(function(m)
         if m.setMode then m:setMode(1, 1, 1) end
      end)
      disc:apply(function(m)
         if m.setMode then m:setMode(1, 1, 1) end
      end)
      from_rgb:apply(function(m)
         if m.setMode then m:setMode(1, 1, 1) end
      end)
      prior:apply(function(m)
         if m.setMode then m:setMode(1, 1, 1) end
      end)
      recon:apply(function(m)
         if m.setMode then m:setMode(1, 1, 1) end
      end)
      gan_feature:apply(function(m)
         if m.setMode then m:setMode(1, 1, 1) end
      end)
   end

   sampling_z = nn.Sampler()
   KLD = nn.GKLDCriterion()
   BCECriterion = nn.BCECriterion()
   ReconCriterion = nn.MSECriterion()
   ReconZCriterion = nn.MSECriterion()

   encoder_total:cuda()
   decoder:cuda()
   var_encoder:cuda()
   var_decoder:cuda()
   prior:cuda()
   ReconCriterion:cuda()
   KLD:cuda()
   sampling_z:cuda()
   BCECriterion:cuda()
   disc:cuda()
   gan_feature:cuda()
   recon:cuda()
   ReconZCriterion:cuda()
   from_rgb:cuda()
   to_rgb:cuda()
   from_rgb_encoder:cuda()

   return {encoder_total, from_rgb_encoder, decoder, to_rgb, var_encoder, var_decoder, prior, from_rgb, gan_feature, recon, disc, sampling_z}, {KLD, ReconCriterion, BCECriterion, ReconZCriterion}
end

return createModel