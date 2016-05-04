package = "torch-rnn"
version = "scm-1"
source = {
  url = "git://github.com/jcjohnson/torch-rnn.git",
}
description = {
  summary = "Efficient, reusable RNNs and LSTMs for Torch.",
  detailed = [[
    torch-rnn provides efficient torch/nn modules implementing LSTMs and RNNs.
  ]],
  homepage = "https://github.com/jcjohnson/torch-rnn",
  license = "MIT"
}
dependencies = {
   "torch >= 7.0",
   "nn >= 1.0",
}
build = {
  type = "builtin",
  modules = {
    ["torch-rnn.init"] = "init.lua",
    ["torch-rnn.LSTM"] = "LSTM.lua",
    ["torch-rnn.VanillaRNN"] = "VanillaRNN.lua",
    ["torch-rnn.TemporalCrossEntropyCriterion"] = "TemporalCrossEntropyCriterion.lua",
  }
}