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
cmd:option('-decay_every', 5)
cmd:option('-decay_factor', 0.2)

local opt = cmd:parse(arg)

local loader = DataLoader(opt)
local vocab = utils.read_json(opt.input_json)
for k, v in pairs(vocab.idx_to_token) do
  vocab.idx_to_token[k] = nil
  vocab.idx_to_token[tonumber(k)] = v
end

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


local N, T = opt.batch_size, opt.num_timesteps
local e, i = 1, 1
local loss_history = {}
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

  table.insert(loss_history, loss)
  if true or i % 10 == 0 then
    print(e, i, loss)
  end

  return loss, grad_params
end

local optim_config = {
  learningRate = opt.learning_rate
}
while e <= opt.num_epochs do
  if e % opt.decay_every == 0 then
    local old_lr = optim_config.learningRate
    local new_lr = opt.decay_factor * old_lr
    print('DECAYING LEARNING RATE! ', old_lr, ' -> ', new_lr)
    optim_config = {learningRate = new_lr}
  end
  model:resetStates()
  i = 1
  while i <= loader.x_train:size(1) do
    optim.adam(f, params, optim_config)
    i = i + 1
  end

  -- Try sampling after each epoch
  local sampled = model:sample(' ', 200)
  print(sampled)
  utils.write_json('losses.json', loss_history)
  e = e + 1
end
