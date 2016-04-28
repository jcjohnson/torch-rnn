require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'nngraph'

--[[
This file contains a modified version of the LSTM implementation by
Wojciech Zaremba found in https://github.com/wojzaremba/lstm

I've moved all model code to a single file, changed it to use DoubleTensors
rather than CudaTensors, and added annotations to several of the nngraph nodes
so that we can access their weights and activations.

wojzaremba/lstm is released under an Apache license, so this probably counts as
a derivative work, meaning that I'm supposed to redistribute the license; you
can find in in wojzaremba_lstm_license.txt.
--]]


local M = {}


local params = {batch_size=20,
                seq_length=20,
                layers=2,
                decay=2,
                rnn_size=200,
                dropout=0,
                init_weight=0.1,
                lr=1,
                vocab_size=10000,
                max_epoch=4,
                max_max_epoch=13,
                max_grad_norm=5,
              }

local function transfer_data(x)
  return x:double()
  -- return x:cuda()
end


local function g_replace_table(to, from)
  assert(#to == #from)
  for i = 1, #to do
    to[i]:copy(from[i])
  end
end


local function g_cloneManyTimes(net, T)
  local clones = {}
  local params, gradParams = net:parameters()
  if params == nil then
    params = {}
  end
  local paramsNoGrad
  if net.parametersNoGrad then
    paramsNoGrad = net:parametersNoGrad()
  end
  local mem = torch.MemoryFile("w"):binary()
  mem:writeObject(net)
  for t = 1, T do
    -- We need to use a new reader for each clone.
    -- We don't want to use the pointers to already read objects.
    local reader = torch.MemoryFile(mem:storage(), "r"):binary()
    local clone = reader:readObject()
    reader:close()
    local cloneParams, cloneGradParams = clone:parameters()
    local cloneParamsNoGrad
    for i = 1, #params do
      cloneParams[i]:set(params[i])
      cloneGradParams[i]:set(gradParams[i])
    end
    if paramsNoGrad then
      cloneParamsNoGrad = clone:parametersNoGrad()
      for i =1,#paramsNoGrad do
        cloneParamsNoGrad[i]:set(paramsNoGrad[i])
      end
    end
    clones[t] = clone
    collectgarbage()
  end
  mem:close()
  return clones
end


local function lstm(i, prev_c, prev_h, prefix)
  prefix = prefix or ''
  local function new_input_sum(name)
    local i2h            = nn.Linear(params.rnn_size, params.rnn_size)
    local h2h            = nn.Linear(params.rnn_size, params.rnn_size)
    i2h = i2h(i)
    h2h = h2h(prev_h)
    i2h:annotate{name=prefix..'_i2h_'..name}
    h2h:annotate{name=prefix..'_h2h_'..name}
    return nn.CAddTable()({i2h, h2h})
  end
  local in_gate          = nn.Sigmoid()(new_input_sum('i')):annotate{name=prefix..'_i'}
  local forget_gate      = nn.Sigmoid()(new_input_sum('f')):annotate{name=prefix..'_f'}
  local in_gate2         = nn.Tanh()(new_input_sum('g')):annotate{name=prefix..'_g'}
  local next_c           = nn.CAddTable()({
    nn.CMulTable()({forget_gate, prev_c}),
    nn.CMulTable()({in_gate,     in_gate2})
  }):annotate{name=prefix..'_next_c'}
  local out_gate         = nn.Sigmoid()(new_input_sum('o')):annotate{name=prefix..'_o'}
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
  return next_c, next_h
end


local function create_network()
  local x                = nn.Identity()()
  local y                = nn.Identity()()
  local prev_s           = nn.Identity()()
  local i                = {[0] = nn.LookupTable(params.vocab_size,
                                                 params.rnn_size)(x)}
  i[0]:annotate{name='lookup_table'}
  local next_s           = {}
  local split         = {prev_s:split(2 * params.layers)}
  for layer_idx = 1, params.layers do
    local prev_c         = split[2 * layer_idx - 1]
    local prev_h         = split[2 * layer_idx]
    local dropped        = nn.Dropout(params.dropout)(i[layer_idx - 1])
    local prefix = string.format('layer_%d', layer_idx)
    local next_c, next_h = lstm(dropped, prev_c, prev_h, prefix)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    i[layer_idx] = next_h
  end
  local h2y              = nn.Linear(params.rnn_size, params.vocab_size)
  local dropped          = nn.Dropout(params.dropout)(i[params.layers])
  local h2y_gmod = h2y(dropped)
  h2y_gmod:annotate{name='h2y'}
  local pred             = nn.LogSoftMax()(h2y_gmod)
  local err              = nn.ClassNLLCriterion()({pred, y})
  local module           = nn.gModule({x, y, prev_s},
                                      {err, nn.Identity()(next_s)})
  module:getParameters():uniform(-params.init_weight, params.init_weight)
  return transfer_data(module)
end


function M.find_named_modules(gmod)
  local name_to_mods = {}
  for _, node in ipairs(gmod.forwardnodes) do
    if node.data.module then
      local node_name = node.data.annotations.name
      if node_name then
        assert(name_to_mods[node_name] == nil, 'Node names must be unique')
        name_to_mods[node_name] = node.data.module
      end
    end
  end
  return name_to_mods
end


function M.find_modules(model)
  return M.find_named_modules(model.core_network)
end


function M.reset_state(model, state)
  state.pos = 1
  if model ~= nil and model.start_s ~= nil then
    for d = 1, 2 * params.layers do
      model.start_s[d]:zero()
    end
  end
end


function M.getParam(name)
  return params[name]
end



function M.setup()
  local model = {}
  local core_network = create_network()
  local paramx, paramdx = core_network:getParameters()
  model.s = {}
  model.ds = {}
  model.start_s = {}
  for j = 0, params.seq_length do
    model.s[j] = {}
    for d = 1, 2 * params.layers do
      model.s[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    end
  end
  for d = 1, 2 * params.layers do
    model.start_s[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    model.ds[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  end
  model.core_network = core_network
  model.rnns = g_cloneManyTimes(core_network, params.seq_length)
  model.norm_dw = 0
  model.err = transfer_data(torch.zeros(params.seq_length))
  return model, paramx, paramdx
end


function M.fp(model, state)
  g_replace_table(model.s[0], model.start_s)
  if state.pos + params.seq_length > state.data:size(1) then
    M.reset_state(model, state)
  end
  for i = 1, params.seq_length do
    local x = state.data[state.pos]
    local y = state.data[state.pos + 1]
    local s = model.s[i - 1]
    model.err[i], model.s[i] = unpack(model.rnns[i]:forward({x, y, s}))
    state.pos = state.pos + 1
  end
  g_replace_table(model.start_s, model.s[params.seq_length])
  return model.err:mean()
end


return M

