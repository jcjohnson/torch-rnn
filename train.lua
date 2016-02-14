require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'optim'

require 'LanguageModel'
require 'DataLoader'

local utils = require 'utils'


local cmd = torch.CmdLine()

-- Dataset options
cmd:option('-input_h5', 'data/tiny-shakespeare.h5')
cmd:option('-input_json', 'data/tiny-shakespeare.json')

-- Model options
cmd:option('-cell_type', 'lstm')
cmd:option('-wordvec_dim', 64)
cmd:option('-hidden_dim', 128)
cmd:option('-num_layers', 2)

-- Batch options
cmd:option('-batch_size', 50)
cmd:option('-num_timesteps', 50)

-- Optimization options
cmd:option('-num_epochs', 10)
cmd:option('-learning_rate', 2e-3)
cmd:option('-grad_clip', 5)

local opt = cmd:parse(arg)

local loader = DataLoader(opt)
local vocab = utils.read_json(opt.input_json)

local model_kwargs = {
  idx_to_token = vocab.idx_to_token,
  cell_type = opt.cell_type,
  wordvec_dim = opt.wordvec_dim,
  hidden_dim = opt.hidden_dim,
  num_layers = opt.num_layers
}
local model = nn.LanguageModel(model_kwargs):cuda()
local params, grad_params = model:getParameters()
local crit = nn.CrossEntropyCriterion():cuda()


local function decode(data, idx_to_token)
  local N, T = data:size(1), data:size(2)
  local strings = {}
  for i = 1, N do
    local s = ''
    for t = 1, T do
      s = s .. idx_to_token[tostring(data[{i, t}])]
    end
    table.insert(strings, s)
  end
  return strings
end


local N, T = opt.batch_size, opt.num_timesteps
local e, i = 1, 1
local function f(w)
  assert(w == params)
  grad_params:zero()
  local x = loader.x_train[i]:cuda()
  local y = loader.y_train[i]:cuda()
  local scores = model:forward(x)
  

  local scores_view = scores:view(N * T, -1)
  local y_view = y:view(N * T)
  local loss = crit:forward(scores_view, y_view)
  local grad_scores = crit:backward(scores_view, y_view):view(N, T, -1)
  model:backward(x, grad_scores)
  grad_params:clamp(-opt.grad_clip, opt.grad_clip)

  --print(decode(x, vocab.idx_to_token))
  --print(decode(y, vocab.idx_to_token))
  --local _, y_pred = 
  --assert(false)

  if true or i % 10 == 0 then
    print(e, i, loss)
  end

  return loss, grad_params
end

local optim_config = {
  learningRate = opt.learning_rate
}
while e <= opt.num_epochs do
  model:resetStates()
  i = 1
  while i <= loader.x_train:size(1) do
    optim.rmsprop(f, params, optim_config)
    -- local loss, grad_params = feval(grad_params)
    -- print(e, i, loss)
    i = i + 1
  end
  e = e + 1
end
