local function ConcatAct()
   local model = nn.ConcatTable()
   model:add(nn.Identity())
   model:add(nn.MulConstant(-1))
   return model
end

local function createModel(opt)
   local encoder = nn.Sequential()
   local from_rgb_encoder = nn.Sequential()
   local decoder = nn.Sequential()
   local to_rgb = nn.Sequential()
   local prior = nn.Sequential()
   local gan = nn.Sequential()
   local from_rgb = nn.Sequential()
   local disc = nn.Sequential()
   local baseChannels = opt.baseChannels
   local w = opt.latentDims[1]
   local z = opt.nf
   local eps = opt.eps
   local mom = opt.mom
   local attribute_dim = opt.attrDim

   if opt.dataset == 'celeba' then
      ---------------------------------
      -- Encoder (Inference network) --
      -- convolution net -> FC layer --
      ---------------------------------
      
      from_rgb_encoder:add(ConcatAct())
      from_rgb_encoder:add(nn.JoinTable(2))
      from_rgb_encoder:add(cudnn.ReLU(true))
      from_rgb_encoder:add(cudnn.SpatialConvolution(6, baseChannels, 1, 1, 1, 1, 0, 0))
      from_rgb_encoder:add(nn.SpatialBatchNormalization(baseChannels, eps, mom))
      from_rgb_encoder:add(cudnn.ReLU(false))

      local encoder_sub = nn.Sequential()

      -- conv1: 64 x 64 --> 32 x 32
      encoder_sub:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 5, 5, 2, 2, 2, 2))
      encoder_sub:add(nn.SpatialBatchNormalization(baseChannels, eps, mom))
      encoder_sub:add(ConcatAct())
      encoder_sub:add(nn.JoinTable(2))
      encoder_sub:add(cudnn.ReLU(true))
      baseChannels = baseChannels * 2

      -- conv2: 32 x 32 --> 16 x 16
      encoder_sub:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 3, 3, 2, 2, 1, 1))
      encoder_sub:add(nn.SpatialBatchNormalization(baseChannels, eps, mom))
      encoder_sub:add(ConcatAct())
      encoder_sub:add(nn.JoinTable(2))
      encoder_sub:add(cudnn.ReLU(true))
      baseChannels = baseChannels * 2

      -- conv3-1, conv3-2: 16 x 16 --> 8 x 8
      encoder_sub:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 3, 3, 2, 2, 1, 1))
      encoder_sub:add(nn.SpatialBatchNormalization(baseChannels, eps, mom))
      encoder_sub:add(ConcatAct())
      encoder_sub:add(nn.JoinTable(2))
      encoder_sub:add(cudnn.ReLU(true))
      baseChannels = baseChannels * 2
      encoder_sub:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 3, 3, 1, 1, 1, 1))
      encoder_sub:add(nn.SpatialBatchNormalization(baseChannels, eps, mom))
      encoder_sub:add(nn.LeakyReLU(0.1))

      -- conv4-1, conv4-2: 8 x 8 --> 4 x 4
      encoder_sub:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 3, 3, 2, 2, 1, 1))
      encoder_sub:add(nn.SpatialBatchNormalization(baseChannels, eps, mom))
      encoder_sub:add(ConcatAct())
      encoder_sub:add(nn.JoinTable(2))
      encoder_sub:add(cudnn.ReLU(true))
      encoder_sub:add(cudnn.SpatialConvolution(baseChannels*2, baseChannels, 3, 3, 1, 1, 1, 1))
      encoder_sub:add(nn.SpatialBatchNormalization(baseChannels, eps, mom))
      encoder_sub:add(nn.LeakyReLU(0.1))

      -- Fully-connected: 4 x 4 x baseChannels --> 4 x 4 x z
      encoder_sub:add(nn.View(w*w*baseChannels):setNumInputDims(3))

      local encoder_attribute = nn.Sequential()
      encoder_attribute:add(nn.Linear(attribute_dim, w*w*z))
      encoder_attribute:add(nn.LeakyReLU(0.1))

      local connect_attribute_vision = nn.ParallelTable()
      connect_attribute_vision:add(encoder_attribute):add(encoder_sub)
      encoder:add(connect_attribute_vision)
      encoder:add(nn.JoinTable(2))
      encoder:add(nn.Linear(w*w*baseChannels+w*w*z, w*w*z))
      encoder:add(nn.LeakyReLU(0.1))


      local mean_logvar = nn.ConcatTable()
      mean_logvar:add(nn.Linear(w*w*z, w*w*z)) -- mean
      mean_logvar:add(nn.Linear(w*w*z, w*w*z)) -- variance
      encoder:add(mean_logvar)

      -----------------------------------
      -- Decoder (Generation network)  --
      -- FC layer -> deconvolution net --
      -----------------------------------
      -- Fully-connected: 4 x 4 x z --> 4 x 4 x baseChannels
      local decoder_attribute = nn.Sequential()
      decoder_attribute:add(nn.Linear(attribute_dim, w*w*z))
      decoder_attribute:add(nn.LeakyReLU(0.1))

      local connect_attribute_vision_decoder = nn.ParallelTable():add(decoder_attribute):add(nn.Identity())
      decoder:add(connect_attribute_vision_decoder)
      decoder:add(nn.JoinTable(2))
      decoder:add(nn.Linear(z*w*w + w*w*z, w*w*baseChannels))
      decoder:add(nn.View(baseChannels, w, w):setNumInputDims(1))

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
      
      --to_rgb:add(cudnn.SpatialConvolution(baseChannels, 3, 3, 3, 1, 1, 1, 1))
      to_rgb:add(cudnn.SpatialConvolution(baseChannels, 3, 1, 1, 1, 1, 0, 0))
      to_rgb:add(nn.Tanh())

      -------------------
      -- Prior --
      -------------------
      -- conv1: 64 x 64 --> 32 x 32
      prior:add(nn.Linear(attribute_dim, w*w*z))
      prior:add(nn.LeakyReLU(0.1))
      prior:get(1).weight:uniform(-0.0001, 0.0001)
      local mean_logvar = nn.ConcatTable()
      mean_logvar:add(nn.Sequential():add(nn.Linear(w*w*z, w*w*z))) -- mean
      mean_logvar:add(nn.Sequential():add(nn.Linear(w*w*z, w*w*z))) -- variance
      prior:add(mean_logvar)
 
      -------------------
      -- Discriminator --
      -------------------
      -- conv1: 64 x 64 --> 32 x 32
      baseChannels = baseChannels/2
      --from_rgb:add(cudnn.SpatialConvolution(3, baseChannels, 5, 5, 1, 1, 2, 2))
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

      gan:add(nn.View(w*w*baseChannels*8))
      gan:add(nn.Linear(w*w*baseChannels*8, w*w*baseChannels))
      gan:add(cudnn.ReLU(true))

      -- Fully-connected: 4 x 4 x (baseChannels x 8) --> 4 x 4 x (baseChannels x 2)
      -- followed by batch discrimination
      local gan_attribute = nn.Sequential()
      gan_attribute:add(nn.Linear(attribute_dim, w*w*z*2))
      gan_attribute:add(nn.LeakyReLU(0.1))

      local connect_attribute_vision = nn.ParallelTable()
      connect_attribute_vision:add(gan_attribute):add(gan)
      disc:add(connect_attribute_vision)
      disc:add(nn.JoinTable(2))
      disc:add(nn.Linear(w*w*baseChannels+2*w*w*z, w*w*baseChannels)) 
      disc:add(nn.Normalize(2))
      disc:add(nn.Dropout(0.5, true))
      disc:add(cudnn.ReLU(true))
      disc:add(nn.BatchDiscrimination(w*w*baseChannels, 100, 5))
      disc:add(nn.Linear(w*w*baseChannels+100, 1)) --+ 100, 1))
      disc:add(nn.Sigmoid())
   else
      error('invalid dataset: ' .. opt.dataset)
   end

   local function ConvInit(name)
      for k, v in pairs(encoder:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:uniform(-1*math.sqrt(1/n), math.sqrt(1/n))
         if not opt.bias then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
      for k, v in pairs(decoder:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:uniform(-1*math.sqrt(1/n), math.sqrt(1/n))
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
      for k, v in pairs(encoder:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
      for k, v in pairs(from_rgb_encoder:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
      for k, v in pairs(decoder:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
      for k, v in pairs(disc:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('cudnn.SpatialFullConvolution')
   ConvInit('nn.SpatialFullConvolution')
   ConvInit('nn.SpatialConvolution')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')

   for k, v in pairs(encoder:findModules('nn.Linear')) do
      v.bias:zero()
   end
   for k, v in pairs(decoder:findModules('nn.Linear')) do
      v.bias:zero()
   end
   for k, v in pairs(disc:findModules('nn.Linear')) do
      v.bias:zero()
   end

   if opt.cudnn == 'deterministic' then
      encoder:apply(function(m)
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
   end

   local sampling_z = nn.Sampler()
   local KLD = nn.GKLDCriterion()
   local BCECriterion = nn.BCECriterion()
   local ReconCriterion = nn.MSECriterion()

   encoder:cuda()
   from_rgb_encoder:cuda()
   decoder:cuda()
   to_rgb:cuda()
   prior:cuda()
   ReconCriterion:cuda()
   KLD:cuda()
   sampling_z:cuda()

   -- GAN-related
   local BCECriterion = nn.BCECriterion():cuda()
   disc:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
   from_rgb:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
   disc:cuda()
   from_rgb:cuda()

   return {encoder, from_rgb_encoder, decoder, to_rgb, prior, sampling_z, from_rgb, disc}, {KLD, ReconCriterion, BCECriterion}
end

return createModel
