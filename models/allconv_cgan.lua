local function ConcatAct()
   local model = nn.ConcatTable()
   model:add(nn.Identity())
   model:add(nn.MulConstant(-1))
   return model
end

local function createModel(opt)
   local decoder = nn.Sequential()
   local gan = nn.Sequential()
   local gan_total = nn.Sequential()
   local baseChannels = opt.baseChannels
   local w = opt.latentDims[1]
   local z = opt.nf
   local eps = opt.eps
   local mom = opt.mom
   local attribute_dim = opt.attrDim

   if opt.dataset == 'celeba' then
      -----------------------------------
      -- Decoder (Generation network)  --
      -- FC layer -> deconvolution net --
      -----------------------------------

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

      -------------------
      -- Discriminator --
      -------------------
      -- conv1: 64 x 64 --> 32 x 32
      baseChannels = baseChannels/2
      gan:add(cudnn.SpatialConvolution(3, baseChannels, 5, 5, 1, 1, 2, 2))
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
      gan_total:add(connect_attribute_vision)
      gan_total:add(nn.JoinTable(2))
      gan_total:add(nn.Linear(w*w*baseChannels+2*w*w*z, w*w*baseChannels)) 
      gan_total:add(nn.Normalize(2))
      gan_total:add(nn.Dropout(0.5, true))
      gan_total:add(cudnn.ReLU(true))
      gan_total:add(nn.BatchDiscrimination(w*w*baseChannels, 100, 5))
      gan_total:add(nn.Linear(w*w*baseChannels+100, 1)) --+ 100, 1))
      gan_total:add(nn.Sigmoid())

   else
      error('invalid dataset: ' .. opt.dataset)
   end

   local function ConvInit(name)
      for k, v in pairs(gan_total:findModules(name)) do
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
      for k, v in pairs(gan_total:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
      for k, v in pairs(decoder:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end
   for k, v in pairs(gan_total:findModules('nn.Linear')) do
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
      gan_total:apply(function(m)
         if m.setMode then m:setMode(1, 1, 1) end
      end)
      decoder:apply(function(m)
         if m.setMode then m:setMode(1, 1, 1) end
      end)
   end

   BCECriterion = nn.BCECriterion()

   gan_total:cuda()
   decoder:cuda()
   BCECriterion:cuda()

   return {gan_total, decoder}, BCECriterion
end

return createModel
