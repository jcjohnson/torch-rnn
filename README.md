# torch-rnn
torch-rnn provides high-performance, reusable RNN and LSTM modules for torch7, and uses these modules for character-level
language modeling similar to [char-rnn](https://github.com/karpathy/char-rnn). You can find documentation for the modules
[here](modules.md).

Compared to char-rnn, torch-rnn is up to **1.9x faster** and uses up to **7x less memory**. For more details see 
the [Benchmark](#Benchmarks) section below.


# TODOs
- CPU support
- OpenCL support
- Documentation
  - Dependencies / installation
  - VanillaRNN
  - LSTM
  - LanguageModel
  - preprocess.py
  - train.lua
  - sample.lua

# Setup

# Usage

# Benchmarks

<img src='imgs/lstm_time_benchmark.png' width="400px">
<img src='imgs/lstm_memory_benchmark.png' width="400px">
