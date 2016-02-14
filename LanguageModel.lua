require 'torch'
require 'nn'

require 'SequenceRNN'
require 'SequenceLSTM'

local utils = require 'utils'


local LM, parent = torch.class('nn.LanguageModel', 'nn.Module')


function LM:__init(kwargs)
  self.idx_to_token = utils.get_kwarg(kwargs, 'idx_to_token')
  self.vocab_size = utils.get_size(self.idx_to_token)

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


function LM:clearStates()
  for i, rnn in ipairs(self.rnns) do
    rnn:clearStates()
  end
end

