--Import Libraries
package.path = package.path .. ';?/?.lua;./?.lua'
require 'torch';
require 'nn';
require 'cunn';
require 'optim';
require 'image';
require 'pl';
require 'paths';
require 'cutorch';
require 'cudnn';
require 'stn';

require 'utils.Sampler';
require 'utils.GKLDCriterion';
DataLoader = require 'utils.dataloader';
init = require 'utils.init';
opts = require 'opts';
--require 'utils.L1HingeCriterion';
--require 'layers.LinearMix'
--require 'layers.LinearMix2'
--require 'utils.Reparametrize'
--require 'utils.Uniform';
--require 'utils.KLDCriterion';
require 'utils.RNNinit';


image_utils = require 'utils.image_utils';


--House Keeping
opt = opts.parse(arg)
if opt.cudnn == 'fastest' then
  cudnn.fastest = true
  cudnn.benchmark = true
end
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

--opt.data = '/local/wshang/celeba_align_loose_crop/'

--Saving Dir Name
opt.save = opt.save  .. opt.netType .. '_' .. opt.latentType  .. '_beta_' .. opt.beta .. '_LR_' .. opt.LR
if string.find(opt.latentType, 'pggan') ~= nil and opt.timeStep > 0 then
  opt.save = opt.save .. '_' .. opt.LR_mult
end
if opt.alpha1 and opt.alpha2 then
  opt.save = opt.save  .. '_alpha_' .. opt.alpha1 .. '_' .. opt.alpha2
else
  opt.save = opt.save  .. '_alpha_' .. opt.alpha
end
opt.save = opt.save .. '_beta1_' .. opt.beta1
if (string.find(opt.latentType, 'crvae') ~= nil or string.find(opt.latentType, 'acvae') ~= nil) and opt.timeStep > 0 then
  opt.save = opt.save .. '_ts_' .. opt.timeStep
end
if opt.kappa > 0 then
  opt.save = opt.save .. '_kp_' .. opt.kappa
end
if opt.rho > 0 then
  opt.save = opt.save .. '_rho_' .. opt.rho
end
if opt.rho_entreg > 0 then
  opt.save = opt.save .. '_ent_' .. opt.rho_entreg
end
opt.save = opt.save ..'_seed_' .. opt.manualSeed .. '/'

print('Saving everything to: ' .. opt.save)
os.execute('mkdir -p ' .. opt.save)

init.savetable(opt, opt.save .. 'hyperparameters.txt')

--Data Loading
--opt.scales = {64, 128}
--opt.latentDims = {4}
--opt.latentFilters = {opt.nf}
--trainLoader, valLoader, testLoader = DataLoader.create(opt)

--Data Loading
if opt.dataset == 'celeba' then
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
elseif opt.dataset == 'celeba128' then
    --opt.data = '/local/wshang/celeba_align_loose_crop/'
    opt.scales = {128}
    opt.latentDims = {4}
    opt.latentFilters = {opt.nf}
    trainLoader, valLoader = DataLoader.create(opt)
end


--Import Model, Criterion and Training functions
local createModel = require('models/' .. opt.netType .. '_' .. opt.latentType)
models, criterions = createModel(opt)
require('train_test/train_test_' .. opt.latentType)


--Training
epoch = opt.epochNumber
for i = opt.epochNumber, opt.nEpochs do
   train(opt)
   collectgarbage()
   val(opt)
   collectgarbage()
   epoch = epoch + 1
end
