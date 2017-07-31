require 'torch'
require 'nn'

require 'LanguageModel'
require 'util.DataLoader'

local utils = require 'util.utils'


local cmd = torch.CmdLine()

cmd:option('-checkpoint', '')
cmd:option('-split', 'val')
cmd:option('-gpu', 0)
cmd:option('-gpu_backend', 'cuda')
local opt = cmd:parse(arg)


-- Set up GPU stuff
local dtype = 'torch.FloatTensor'
if opt.gpu >= 0 and opt.gpu_backend == 'cuda' then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.gpu + 1)
  dtype = 'torch.CudaTensor'
  print(string.format('Running with CUDA on GPU %d', opt.gpu))
elseif opt.gpu >= 0 and opt.gpu_backend == 'opencl' then
  require 'cltorch'
  require 'clnn'
  cltorch.setDevice(opt.gpu + 1)
  dtype = torch.Tensor():cl():type()
  print(string.format('Running with OpenCL on GPU %d', opt.gpu))
else
  -- Memory benchmarking is only supported in CUDA mode
  print 'Running in CPU mode'
end

-- Load the checkpoint and model
local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model
model:type(dtype)
local crit = nn.CrossEntropyCriterion():type(dtype)

-- Load the vocab and data
local loader = DataLoader(checkpoint.opt)
local N, T = checkpoint.opt.batch_size, checkpoint.opt.seq_length

-- Evaluate the model on the specified split
model:evaluate()
model:resetStates()
local num = loader.split_sizes[opt.split]
local loss = 0
for i = 1, num do
  print(string.format('%s batch %d / %d', opt.split, i, num))
  local x, y = loader:nextBatch(opt.split)
  N = x:size(1)
  x = x:type(dtype)
  y = y:type(dtype):view(N * T)
  local scores = model:forward(x):view(N * T, -1)
  loss = loss + crit:forward(scores, y)
end
loss = loss / num
print(string.format('%s loss = %f', opt.split, loss))
