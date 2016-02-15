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
cmd:option('-dropout', 0)

-- Batch options
cmd:option('-batch_size', 50)
cmd:option('-num_timesteps', 50)

-- Optimization options
cmd:option('-max_epochs', 10)
cmd:option('-learning_rate', 2e-3)
cmd:option('-grad_clip', 5)
cmd:option('-lr_decay_every', 5)
cmd:option('-lr_decay_factor', 0.2)

cmd:option('-print_every', 1)
cmd:option('-checkpoint_every', 1000)
cmd:option('-checkpoint_name', 'cv/checkpoint')

cmd:option('-gpu', 0)

local opt = cmd:parse(arg)


cutorch.setDevice(opt.gpu + 1)


-- Initialize the DataLoader and vocabulary
local loader = DataLoader(opt)
local vocab = utils.read_json(opt.input_json)
local idx_to_token = {}
for k, v in pairs(vocab.idx_to_token) do
  idx_to_token[tonumber(k)] = v
end

-- Initialize the model and criterion
local opt_clone = torch.deserialize(torch.serialize(opt))
opt_clone.idx_to_token = idx_to_token
local model = nn.LanguageModel(opt_clone):cuda()
local params, grad_params = model:getParameters()
local crit = nn.CrossEntropyCriterion():cuda()

-- Set up some variables we will use below
local N, T = opt.batch_size, opt.num_timesteps
local train_loss_history = {}
local val_loss_history = {}
local val_loss_history_it = {}

-- Loss function that we pass to an optim method
local function f(w)
  assert(w == params)
  grad_params:zero()

  -- Get a minibatch and run the model forward
  local x, y = loader:nextBatch('train')
  x, y = x:cuda(), y:cuda()
  local scores = model:forward(x)

  -- Use the Criterion to compute loss; we need to reshape the scores to be
  -- two-dimensional before doing so. Annoying.
  local scores_view = scores:view(N * T, -1)
  local y_view = y:view(N * T)
  local loss = crit:forward(scores_view, y_view)

  -- Run the Criterion and model backward to compute gradients
  local grad_scores = crit:backward(scores_view, y_view):view(N, T, -1)
  model:backward(x, grad_scores)

  if opt.grad_clip > 0 then
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  end

  return loss, grad_params
end

-- Train the model!
local optim_config = {learningRate = opt.learning_rate}
local num_train = loader.split_sizes['train']
local num_iterations = opt.max_epochs * num_train
model:training()
for i = 1, num_iterations do
  local epoch = math.floor(i / num_train) + 1

  -- Check if we are at the end of an epoch
  if i % num_train == 0 then
    model:resetStates() -- Reset hidden states

    -- Maybe decay learning rate
    if epoch % opt.lr_decay_every == 0 then
      local old_lr = optim_config.learningRate
      optim_config = {learningRate = old_lr * opt.lr_decay_factor}
    end
  end

  -- Take a gradient step and maybe print
  -- Note that adam returns a singleton array of losses
  local _, loss = optim.adam(f, params, optim_config)
  table.insert(train_loss_history, loss[1])
  if opt.print_every > 0 and i % opt.print_every == 0 then
    local float_epoch = i / num_train + 1
    local msg = 'Epoch %.2f / %d, i = %d / %d, loss = %f'
    local args = {msg, float_epoch, opt.max_epochs, i, num_iterations, loss[1]}
    print(string.format(unpack(args)))
  end

  -- Maybe save a checkpoint
  if i % opt.checkpoint_every == 0 or i == num_iterations then
    -- Evaluate loss on the validation set. Note that we reset the state of
    -- the model; this might happen in the middle of an epoch, but that
    -- shouldn't cause too much trouble.
    model:evaluate()
    model:resetStates()
    local num_val = loader.split_sizes['val']
    local val_loss = 0
    for j = 1, num_val do
      local xv, yv = loader:nextBatch('val')
      xv, yv = xv:cuda(), yv:cuda():view(N * T)
      local scores = model:forward(xv):view(N * T, -1)
      val_loss = val_loss + crit:forward(scores, yv)
    end
    val_loss = val_loss / num_val
    print('val_loss = ', val_loss)
    table.insert(val_loss_history, val_loss)
    table.insert(val_loss_history_it, i)
    model:resetStates()
    model:training()

    -- First save a JSON checkpoint, excluding the model
    local checkpoint = {
      opt = opt,
      train_loss_history = train_loss_history,
      val_loss_history = val_loss_history,
      val_loss_history_it = val_loss_history_it,
    }
    local filename = string.format('%s_%d.json', opt.checkpoint_name, i)
    utils.write_json(filename, checkpoint)

    -- Now save a torch checkpoint with the model
    -- Cast the model to float before saving so it can be used on CPU
    model:float()
    checkpoint.model = model
    local filename = string.format('%s_%d.t7', opt.checkpoint_name, i)
    torch.save(filename, checkpoint)
    model:cuda()
    params, grad_params = model:getParameters()
  end
end

