local function ConcatAct()
   local model = nn.ConcatTable()
   model:add(nn.Identity())
   model:add(nn.MulConstant(-1))
   return model
end

local function createModel(opt)
   local encoder_total = nn.Sequential()
   local decoder = nn.Sequential()
   local prior = nn.Sequential()
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
      local encoder = nn.Sequential()
      encoder:add(ConcatAct())
      encoder:add(nn.JoinTable(2))
      encoder:add(cudnn.ReLU(true))

      -- conv1: 64 x 64 --> 32 x 32
      encoder:add(cudnn.SpatialConvolution(6, baseChannels, 5, 5, 2, 2, 2, 2))
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

      -- Fully-connected: 4 x 4 x baseChannels --> 4 x 4 x z
      encoder:add(nn.View(w*w*baseChannels):setNumInputDims(3))

      local encoder_attribute = nn.Sequential()
      encoder_attribute:add(nn.Linear(attribute_dim, w*w*z))
      encoder_attribute:add(nn.LeakyReLU(0.1))

      local connect_attribute_vision = nn.ParallelTable()
      connect_attribute_vision:add(encoder_attribute):add(encoder)
      encoder_total:add(connect_attribute_vision)
      encoder_total:add(nn.JoinTable(2))
      encoder_total:add(nn.Linear(w*w*baseChannels+w*w*z, w*w*z))
      encoder_total:add(nn.LeakyReLU(0.1))


      local mean_logvar = nn.ConcatTable()
      mean_logvar:add(nn.Linear(w*w*z, w*w*z)) -- mean
      mean_logvar:add(nn.Linear(w*w*z, w*w*z)) -- variance
      encoder_total:add(mean_logvar)

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
      decoder:add(cudnn.SpatialConvolution(baseChannels, 3, 3, 3, 1, 1, 1, 1))
      decoder:add(nn.Tanh())

      
      -----------------------------------
      -- Prior network  --
      -- FC layer -> prior mean and variance --
      -----------------------------------
      prior:add(nn.Linear(attribute_dim, w*w*z))
      prior:add(nn.LeakyReLU(0.1))
      prior:get(1).weight:uniform(-0.0001, 0.0001)
      local mean_logvar = nn.ConcatTable()
      mean_logvar:add(nn.Sequential():add(nn.Linear(w*w*z, w*w*z))) -- mean
      mean_logvar:add(nn.Sequential():add(nn.Linear(w*w*z, w*w*z))) -- variance
      prior:add(mean_logvar)
   else
      error('invalid dataset: ' .. opt.dataset)
   end

   local function ConvInit(name)
      for k, v in pairs(encoder_total:findModules(name)) do
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
   end
   local function BNInit(name)
      for k, v in pairs(encoder_total:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
      for k, v in pairs(decoder:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end
   for k, v in pairs(encoder_total:findModules('nn.Linear')) do
      v.bias:zero()
   end
   for k, v in pairs(decoder:findModules('nn.Linear')) do
      v.bias:zero()
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('cudnn.SpatialFullConvolution')
   ConvInit('nn.SpatialFullConvolution')
   ConvInit('nn.SpatialConvolution')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')

   if opt.cudnn == 'deterministic' then
      encoder_total:apply(function(m)
         if m.setMode then m:setMode(1, 1, 1) end
      end)
      decoder:apply(function(m)
         if m.setMode then m:setMode(1, 1, 1) end
      end)
   end

   --criterions
   sampling_z = nn.Sampler()
   KLD = nn.GKLDCriterion()
   ReconCriterion = nn.MSECriterion()

   encoder_total:cuda()
   decoder:cuda()
   prior:cuda()
   ReconCriterion:cuda()
   KLD:cuda()
   sampling_z:cuda()

   return {encoder_total, decoder, prior, sampling_z}, {KLD, ReconCriterion}
end

return createModel
