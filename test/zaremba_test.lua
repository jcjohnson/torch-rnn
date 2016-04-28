require 'torch'
require 'cutorch'

require 'LSTM'
require 'LanguageModel'
local wzlstm = require 'test.wojzaremba_lstm'


--[[
To make sure our LSTM is correct, we compare directly to Wojciech Zaremba's
LSTM implementation found in https://github.com/wojzaremba/lstm.

I've modified his implementation to fit in a single file, found in the file
wojzaremba_lstm.lua.

After constructing a wojzaremba LSTM, we carefully port the weights over to
a torch-rnn LanguageModel. We then run several minibatches of random data
through both, and ensure that they give the same outputs.
--]]


local tests = torch.TestSuite()
local tester = torch.Tester()


function tests.wzForwardTest()
  local model, paramx, paramdx = wzlstm.setup()
  local modules = wzlstm.find_modules(model)
  local rnn_modules = {}
  for i = 1, #model.rnns do
    table.insert(rnn_modules, wzlstm.find_named_modules(model.rnns[i]))
  end

  -- Make sure that we have found all the paramters
  local total_params = 0
  for name, mod in pairs(modules) do
    local s = name
    if mod.weight then
      local num_w = mod.weight:nElement()
      total_params = total_params + num_w
      s = s .. ' ' .. num_w .. ' weights'
    end
    if mod.bias then
      local num_b = mod.bias:nElement()
      total_params = total_params + num_b
      s = s .. ' ' .. num_b .. ' biases'
    end
  end
  assert(total_params == paramx:nElement())

  local N = wzlstm.getParam('batch_size')
  local T = wzlstm.getParam('seq_length')
  local V = wzlstm.getParam('vocab_size')
  local H = wzlstm.getParam('rnn_size')

  -- Construct my LanguageModel
  local idx_to_token = {}
  for i = 1, V do idx_to_token[i] = i end
  local lm = nn.LanguageModel{
                  idx_to_token=idx_to_token,
                  model_type='lstm',
                  wordvec_size=H,
                  rnn_size=H,
                  num_layers=2,
                  dropout=0,
                  batchnorm=0
             }:double()

  -- Copy weights and biases from the wojzaremba LSTM to my language model
  lm.net:get(1).weight:copy(modules.lookup_table.weight)

  lm.rnns[1].weight[{{1, H}, {1, H}}]:copy(            modules.layer_1_i2h_i.weight:t())
  lm.rnns[1].weight[{{1, H}, {H + 1, 2 * H}}]:copy(    modules.layer_1_i2h_f.weight:t())
  lm.rnns[1].weight[{{1, H}, {2 * H + 1, 3 * H}}]:copy(modules.layer_1_i2h_o.weight:t())
  lm.rnns[1].weight[{{1, H}, {3 * H + 1, 4 * H}}]:copy(modules.layer_1_i2h_g.weight:t())
  lm.rnns[1].weight[{{H + 1, 2 * H}, {1, H}}]:copy(            modules.layer_1_h2h_i.weight:t())
  lm.rnns[1].weight[{{H + 1, 2 * H}, {H + 1, 2 * H}}]:copy(    modules.layer_1_h2h_f.weight:t())
  lm.rnns[1].weight[{{H + 1, 2 * H}, {2 * H + 1, 3 * H}}]:copy(modules.layer_1_h2h_o.weight:t())
  lm.rnns[1].weight[{{H + 1, 2 * H}, {3 * H + 1, 4 * H}}]:copy(modules.layer_1_h2h_g.weight:t())

  lm.rnns[1].bias[{{1, H}}]:copy(modules.layer_1_i2h_i.bias)
  lm.rnns[1].bias[{{1, H}}]:add( modules.layer_1_h2h_i.bias)
  lm.rnns[1].bias[{{H + 1, 2 * H}}]:copy(modules.layer_1_i2h_f.bias)
  lm.rnns[1].bias[{{H + 1, 2 * H}}]:add( modules.layer_1_h2h_f.bias)
  lm.rnns[1].bias[{{2 * H + 1, 3 * H}}]:copy(modules.layer_1_i2h_o.bias)
  lm.rnns[1].bias[{{2 * H + 1, 3 * H}}]:add( modules.layer_1_h2h_o.bias)
  lm.rnns[1].bias[{{3 * H + 1, 4 * H}}]:copy(modules.layer_1_i2h_g.bias)
  lm.rnns[1].bias[{{3 * H + 1, 4 * H}}]:add( modules.layer_1_h2h_g.bias)

  local w1 = {}
  w1.Wxi = lm.rnns[1].weight[{{1, H}, {1, H}}]:clone()
  w1.Wxf = lm.rnns[1].weight[{{1, H}, {1, H}}]:clone()
  w1.Wxo = lm.rnns[1].weight[{{1, H}, {1, H}}]:clone()
  w1.Wxg = lm.rnns[1].weight[{{1, H}, {1, H}}]:clone()

  lm.rnns[2].weight[{{1, H}, {1, H}}]:copy(            modules.layer_2_i2h_i.weight:t())
  lm.rnns[2].weight[{{1, H}, {H + 1, 2 * H}}]:copy(    modules.layer_2_i2h_f.weight:t())
  lm.rnns[2].weight[{{1, H}, {2 * H + 1, 3 * H}}]:copy(modules.layer_2_i2h_o.weight:t())
  lm.rnns[2].weight[{{1, H}, {3 * H + 1, 4 * H}}]:copy(modules.layer_2_i2h_g.weight:t())
  lm.rnns[2].weight[{{H + 1, 2 * H}, {1, H}}]:copy(            modules.layer_2_h2h_i.weight:t())
  lm.rnns[2].weight[{{H + 1, 2 * H}, {H + 1, 2 * H}}]:copy(    modules.layer_2_h2h_f.weight:t())
  lm.rnns[2].weight[{{H + 1, 2 * H}, {2 * H + 1, 3 * H}}]:copy(modules.layer_2_h2h_o.weight:t())
  lm.rnns[2].weight[{{H + 1, 2 * H}, {3 * H + 1, 4 * H}}]:copy(modules.layer_2_h2h_g.weight:t())

  lm.rnns[2].bias[{{1, H}}]:copy(modules.layer_2_i2h_i.bias)
  lm.rnns[2].bias[{{1, H}}]:add(modules.layer_2_h2h_i.bias)
  lm.rnns[2].bias[{{H + 1, 2 * H}}]:copy(modules.layer_2_i2h_f.bias)
  lm.rnns[2].bias[{{H + 1, 2 * H}}]:add(modules.layer_2_h2h_f.bias)
  lm.rnns[2].bias[{{2 * H + 1, 3 * H}}]:copy(modules.layer_2_i2h_o.bias)
  lm.rnns[2].bias[{{2 * H + 1, 3 * H}}]:add(modules.layer_2_h2h_o.bias)
  lm.rnns[2].bias[{{3 * H + 1, 4 * H}}]:copy(modules.layer_2_i2h_g.bias)
  lm.rnns[2].bias[{{3 * H + 1, 4 * H}}]:add(modules.layer_2_h2h_g.bias)
  
  local lm_vocab_linear = lm.net:get(#lm.net - 1)
  lm_vocab_linear.weight:copy(modules.h2y.weight)
  lm_vocab_linear.bias:copy(modules.h2y.bias)
  
  local data = torch.LongTensor(100, N):random(V)

  local state = {data=data}
  wzlstm.reset_state(model, state)

  local crit = nn.CrossEntropyCriterion()

  for i = 1, 4 do
    -- Run Zaremba LSTM forward
    local wz_err = wzlstm.fp(model, state)

    -- Run my LSTM forward
    local t0 = (i - 1) * T + 1
    local t1 = i * T
    local x = data[{{t0, t1}}]:transpose(1, 2):clone()
    local y_gt = data[{{t0 + 1, t1 + 1}}]:transpose(1, 2):clone()

    local y_pred = lm:forward(x)
    local jj_err = crit:forward(y_pred:view(N * T, -1), y_gt:view(N * T, -1))

    -- The outputs should match almost exactly
    local diff = math.abs(wz_err - jj_err)
    tester:assert(diff < 1e-12)
  end
end

tester:add(tests)
tester:run()

