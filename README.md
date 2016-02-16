# torch-rnn
torch-rnn provides high-performance, reusable RNN and LSTM modules for torch7, and uses these modules for character-level
language modeling similar to [char-rnn](https://github.com/karpathy/char-rnn).

You can find documentation for the RNN and LSTM modules [here](modules.md); they have no dependencies other than `torch`
and `nn`, so they should be easy to integrate into existing projects.

Compared to char-rnn, torch-rnn is up to **1.9x faster** and uses up to **7x less memory**. For more details see 
the [Benchmark](#benchmarks) section below.


# TODOs
- CPU support
- OpenCL support?
- Get rid of Python / JSON / HDF5 dependencies?
- Documentation
  - Dependencies / installation
  - VanillaRNN
  - LSTM
  - LanguageModel
  - preprocess.py
  - train.lua
  - sample.lua

# Installation
## Python setup
The preprocessing script is written in Python 2.7; its dependencies are in the file `requirements.txt`.
You can install these dependencies in a virtual environment like this:

```bash
virtualenv .env                  # Create the virtual environment
source .env/bin/activate         # Activate the virtual environment
pip install -r requirements.txt  # Install Python dependencies
# Work for a while ...
deactivate                       # Exit the virtual environment
```

## Lua setup
The main modeling code is written in Lua using [torch](http://torch.ch); you can find installation instructions
[here](http://torch.ch/docs/getting-started.html#_). You'll need the following Lua packages:

- [torch/torch7](https://github.com/torch/torch7)
- [torch/nn](https://github.com/torch/nn)
- [torch/optim](https://github.com/torch/optim)
- [lua-cjson](https://luarocks.org/modules/luarocks/lua-cjson)
- [torch-hdf5](https://github.com/deepmind/torch-hdf5)

After installing torch, you can install / update these packages by running the following:

```bash
# Install most things using luarocks
luarocks install torch
luarocks install nn
luarocks install optim
luarocks install lua-cjson

# We need to install torch-hdf5 from GitHub
git clone git@github.com:deepmind/torch-hdf5.git
cd torch-hdf5
luarocks make hdf5-0-0.rockspec
```

### CUDA support
To enable GPU acceleration with CUDA, you'll need to install CUDA 6.5 or higher and the following Lua packages:
- [torch/cutorch](https://github.com/torch/cutorch)
- [torch/cunn](https://github.com/torch/cunn)

You can install / update them by running:

```bash
luarocks install cutorch
luarocks install cunn
```

# Usage
To train a model and use it to generate new text, you'll need to follow three simple steps:

## Step 1: Preprocess the data
You can use any text file for training models. Before training, you'll need to preprocess the data using the script
`scripts/preprocess.py`; this will generate an HDF5 file and JSON file containing a preprocessed version of the data.
You can run the script like this:

```bash
python scripts/preprocess.py \
  --input_txt my_data.txt \
  --output_h5 my_data.h5 \
  --output_json my_data.json
```

The preprocessing script accepts the following command line arguments:

- `--input_txt`: Path to the text file to be used for training. Default is the `tiny-shakespeare.txt` dataset.
- `--output_h5`: Path to the HDF5 file where preprocessed data should be written.
- `--output_json`: Path to the JSON file where preprocessed data should be written.
- `--val_frac`: What fraction of the data to use as a validation set; default is `0.1`.
- `--test_frac`: What fraction of the data to use as a test set; default is `0.1`.
- `--quiet`: If you pass this flag then no output will be printed to the console.

## Step 2: Train the model
After preprocessing the data, you'll need to train the model using the `train.lua` script. This will be the slowest step.
You can run the training script like this:

```bash
th train.lua --input_h5 my_data.h5 --input_json my_data.json
```

You can configure the behavior of the training script with the following flags:

**Data options**:
- `-input_h5`, `-input_json`: Paths to the HDF5 and JSON files output from the preprocessing script.
- `-batch_size`: Number of sequences to use in a minibatch; default is 50.
- `-seq_length`: Number of timesteps for which the recurrent network is unrolled for backpropagation through time.

**Model options**:
- `-model_type`: The type of recurrent network to use; either `lstm` (default) or `rnn`. `lstm` is slower but better.
- `-wordvec_size`: Dimension of learned word vector embeddings; default is 64. You probably won't need to change this.
- `-rnn_size`: The number of hidden units in the RNN; default is 128. Larger values (256 or 512) are commonly used to learn more powerful models and for bigger datasets, but this will significantly slow down computation.
- `-dropout`: Amount of dropout regularization to apply after each RNN layer; must be in the range `0 <= droput < 1`. Setting `dropout` to 0 disables dropout, and higher numbers give a stronger regularizing effect.

**Optimization options**:
- `-max_epochs`: How many training epochs to use for optimization. Default is 50.
- `-learning_rate`: Learning rate for optimization. Default is `2e-3`.
- `-lr_decay_every`: How often to decay the learning rate, in epochs; default is 5.
- `-lr_decay_factor`: How much to decay the learning rate. After every `lr_decay_every` epochs, the learning rate will be multiplied by the `lr_decay_factor`; default is 0.5.

**Output options**:
- `-print_every`: How often to print status message, in iterations. Default is 1.
- `-checkpoint_name`: Base filename for saving checkpoints; default is `cv/checkpoint`. This will create checkpoints named - `cv/checkpoint_1000.t7`, `cv/checkpoint_1000.json`, etc.
- `-checkpoint_every`: How often to save intermediate checkpoints. Default is 1000; set to 0 to disable intermediate checkpointing. Note that we always save a checkpoint on the final iteration of training.

**Benchmark options**:
- `-speed_benchmark`: Set this to 1 to test the speed of the model at every iteration. This is disabled by default because it requires synchronizing the GPU at every iteration, which incurs a performance overhead. Speed benchmarking results will be printed and also stored in saved checkpoints.
- `-memory_benchmark`: Set this to 1 to test the GPU memory usage at every iteration. This is disabled by default because like speed benchmarking it requires GPU synchronization. Memory benchmarking results will be printed and also stored in saved checkpoints. Only available when running in GPU mode.

**Backend options**:
- `-gpu`: The ID of the GPU to use (zero-indexed). Default is 0. Set this to -1 to run in CPU-only mode [NOT YET IMPLEMENTED]

## Step 3: Sample from the model
After training a model, you can generate new text by sampling from it using the script `sample.lua`. You'll typically run
it like this:

```bash
th sample.lua -checkpoint cv/checkpoint_10000.t7 -length 2000
```

This will load the trained checkpoint `cv/checkpoint_10000.t7` from the previous step, sample 2000 characters from it,
and print the results to the console.

The sampling script accepts the following flags:
- `-checkpoint`: Path to a `.t7` checkpoint file from `train.lua`
- `-length`: The length of the generated text, in characters.
- `-start_text`: You can optionally start off the generation process with a string; if this is provided the start text will be processed by the trained network before we start sampling. Without this flag, the first character is chosen randomly.
- `-sample`: Set this to 1 to sample from the next-character distribution at each timestep; set to 0 to instead just pick the argmax at every timestep. Sampling tends to produce more interesting results.
- `-temperature`: Softmax temperature to use when sampling; default is 1. Higher temperatures give noiser samples. Not used when using argmax sampling (`sample` set to 0).
- `-gpu`: The ID of the GPU to use (zero-indexed). Default is 0. Set this to -1 to run in CPU-only mode. [NOT IMPLEMENTED].
- `-verbose`: By default just the sampled text is printed to the console. Set this to 1 to also print some diagnostic information.

# Benchmarks

<img src='imgs/lstm_time_benchmark.png' width="400px">
<img src='imgs/lstm_memory_benchmark.png' width="400px">
