require 'torch'
require 'nn'

require 'LanguageModel'

local turbo = require("turbo")

local cmd = torch.CmdLine()
cmd:option('-checkpoint', 'cv/checkpoint_4000.t7')
cmd:option('-gpu', 0)
cmd:option('-gpu_backend', 'cuda')
cmd:option('-verbose', 0)
cmd:option('-port', 8888) -- http port to listen
local opt = cmd:parse(arg)


local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model

local msg
if opt.gpu >= 0 and opt.gpu_backend == 'cuda' then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.gpu + 1)
  model:cuda()
  msg = string.format('Running with CUDA on GPU %d', opt.gpu)
elseif opt.gpu >= 0 and opt.gpu_backend == 'opencl' then
  require 'cltorch'
  require 'clnn'
  model:cl()
  msg = string.format('Running with OpenCL on GPU %d', opt.gpu)
else
  msg = 'Running in CPU mode'
end
if opt.verbose == 1 then print(msg) end

model:evaluate()


local SampleHandler = class("SampleHandler", turbo.web.RequestHandler)

function SampleHandler:get()
    -- Get the 'name' argument, or use 'Santa Claus' if it does not exist
    opt['length'] = self:get_argument("length", 2000)
    opt['start_text'] = self:get_argument("start_text", "")
    opt['sample'] = self:get_argument("sample", 1)
    opt['temperature'] = self:get_argument("temperature", 1)
    
    local sample = model:sample(opt)
    
    self:write(sample)
end

local app = turbo.web.Application:new({
    {"/sample", SampleHandler}
})

app:listen(opt.port)
turbo.ioloop.instance():start()
