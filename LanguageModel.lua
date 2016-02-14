require 'torch'
require 'nn'

require 'SequenceRNN'
require 'SequenceLSTM'

local utils = require 'utils'


local LM, parent = torch.class('nn.LanguageModel', 'nn.Module')


function LM:__init(kwargs)
  self.idx_to_token = utils.get_kwarg(kwargs, 'idx_to_token')
  self.token_to_idx = {}
  self.vocab_size = 0
  for idx, token in pairs(self.idx_to_token) do
    self.token_to_idx[token] = idx
    self.vocab_size = self.vocab_size + 1
  end

  self.cell_type = utils.get_kwarg(kwargs, 'cell_type', 'lstm')
  self.wordvec_dim = utils.get_kwarg(kwargs, 'wordvec_dim', 128)
  self.hidden_dim = utils.get_kwarg(kwargs, 'hidden_dim', 256)
  self.num_layers = utils.get_kwarg(kwargs, 'num_layers', 1)

  local V, D, H = self.vocab_size, self.wordvec_dim, self.hidden_dim

  self.net = nn.Sequential()
  self.rnns = {}
  self.net:add(nn.LookupTable(V, D))
  for i = 1, self.num_layers do
    local prev_dim = H
    if i == 1 then prev_dim = D end
    local rnn
    if self.cell_type == 'rnn' then
      rnn = nn.SequenceRNN(prev_dim, H)
    elseif self.cell_type == 'lstm' then
      rnn = nn.SequenceLSTM(prev_dim, H)
    end
    rnn.remember_states = true
    table.insert(self.rnns, rnn)
    self.net:add(rnn)
  end

  -- After all the RNNs run, we will have a tensor of shape (N, T, H);
  -- we want to apply a 1D temporal convolution to predict scores for each
  -- vocab element, giving a tensor of shape (N, T, V). Unfortunately
  -- nn.TemporalConvolution is SUPER slow, so instead we will use a pair of
  -- views (N, T, H) -> (NT, H) and (NT, V) -> (N, T, V) with a nn.Linear in
  -- between. Unfortunately N and T can change on every minibatch, so we need
  -- to set them in the forward pass.
  self.view1 = nn.View(1, 1, -1):setNumInputDims(3)
  self.view2 = nn.View(1, -1):setNumInputDims(2)

  self.net:add(self.view1)
  self.net:add(nn.Linear(H, V))
  self.net:add(self.view2)
end


function LM:updateOutput(input)
  local N, T = input:size(1), input:size(2)
  self.view1:resetSize(N * T, -1)
  self.view2:resetSize(N, T, -1)
  return self.net:forward(input)
end


function LM:backward(input, gradOutput, scale)
  return self.net:backward(input, gradOutput, scale)
end


function LM:parameters()
  return self.net:parameters()
end


function LM:resetStates()
  for i, rnn in ipairs(self.rnns) do
    rnn:resetStates()
  end
end


function LM:encode_string(s)
  local encoded = torch.LongTensor(#s)
  for i = 1, #s do
    local token = s:sub(i, i)
    local idx = self.token_to_idx[token]
    assert(idx ~= nil, 'Got invalid idx')
    encoded[i] = idx
  end
  return encoded
end


function LM:decode_string(encoded)
  assert(torch.isTensor(encoded) and encoded:dim() == 1)
  local s = ''
  for i = 1, encoded:size(1) do
    s = s .. self.idx_to_token[encoded[i]]
  end
  return s
end


--[[
Sample from the language model. Note that this will reset the states of the
underlying RNNs.

Inputs:
- init: (1, T0) array of integers
- max_length: Number of characters to sample

Returns:
- sampled: (1, max_length) array of integers, where the first part is init.
--]]
function LM:sample(init, max_length)
  local T0, T = init:size(2), max_length
  local sampled = torch.LongTensor(1, T)
  sampled[{{}, {1, T0}}]:copy(init)
  self:resetStates()
  
  self:resetStates()
  local scores = self:forward(init)[{{}, {T0, T0}}]
  for t = T0 + 1, T do
    local _, next_char = scores:max(3)
    next_char = next_char[{{}, {}, 1}]
    sampled[{{}, {t, t}}]:copy(next_char)
    scores = self:forward(next_char)
  end

  self:resetStates()
  return sampled
end

