--  This code is taken from Facebook ResNet code under the following license
--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--

local image = require 'image'
local paths = require 'paths'
local t = require 'datasets/transforms'
local ffi = require 'ffi'

local M = {}
local CelebaDataset = torch.class('rgb.CelebaDataset', M)

function CelebaDataset:__init(imageInfo, opt, split)
   self.imageInfo = imageInfo[split]
   self.opt = opt
   self.split = split
   self.dir = opt.data
   assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

function CelebaDataset:get(i)
   local path = ffi.string(self.imageInfo.imagePath[i]:data())
   local image = self:_loadImage(paths.concat(self.dir, path))
   local class = self.imageInfo.imageClass[i]

   return {
      input = image,
      target = class,
   }
end

function CelebaDataset:_loadImage(path)
   local ok, input = pcall(function()
      return 2.0*image.load(path, 3, 'float') - 1.0
   end)

   -- Sometimes image.load fails because the file extension does not match the
   -- image format. In that case, use image.decompress on a ByteTensor.
   if not ok then
      local f = io.open(path, 'r')
      assert(f, 'Error reading: ' .. tostring(path))
      local data = f:read('*a')
      f:close()

      local b = torch.ByteTensor(string.len(data))
      ffi.copy(b:data(), data, b:size(1))

      input = image.decompress(b, 3, 'float')
   end

   return input
end

function CelebaDataset:size()
   return self.imageInfo.imageClass:size(1)
end

function CelebaDataset:preprocess()
   if self.split == 'train' then
      return t.Compose{
         t.ScaleSquare(self.opt.scales[self.opt.stage]),
         t.HorizontalFlip(0.5),
      }
   elseif self.split == 'val' then
      return t.Compose{
         t.ScaleSquare(self.opt.scales[self.opt.stage]),
      }
   elseif self.split == 'test' then
      return t.Compose{
         t.ScaleSquare(self.opt.scales[self.opt.stage]),
      }
   else
      error('invalid split: ' .. self.split)
   end
end

return M.CelebaDataset
