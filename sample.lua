require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'

require 'LanguageModel'


local cmd = torch.CmdLine()
cmd:option('-checkpoint', 'cv/checkpoint_4000.t7')
cmd:option('-length', 2000)
cmd:option('-start_text', '')
cmd:option('-sample', 1)
cmd:option('-temperature', 1)
cmd:option('-gpu', 0)
cmd:option('-verbose', 0)
local opt = cmd:parse(arg)

cutorch.setDevice(opt.gpu + 1)

local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model

model:evaluate()
model:cuda()

local sample = model:sample(opt)
print(sample)
