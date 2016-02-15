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

Basic options:
- `-input_h5`, `-input_json`: Paths to the HDF5 and JSON files output from the preprocessing script.

Model options:
- `-model_type`: The type of recurrent network to use; choices are `lstm` or `rnn`. Default is `lstm`, which is slower but tends to perform much better.
- `-wordvec_size`: The dimension of learned word vector embeddings; default is 64. You probably won't need to change this.
- `-rnn_size`: The number of hidden units in the RNN; default is 128. Larger values (256 or 512) are commonly used to learn more powerful models and for bigger datasets, but this will significantly slow down computation.
- `-dropout`: Amount of dropout regularization to apply after each RNN layer; must be in the range `0 <= droput < 1`. Setting `dropout` to 0 disables dropout, and higher numbers give a stronger regularizing effect.

- `-batch_size`: Number of sequences to use in a minibatch; default is 50.
- `-seq_length`: Number of timesteps for which the recurrent network is unrolled for backpropagation through time.


# Benchmarks

<img src='imgs/lstm_time_benchmark.png' width="400px">
<img src='imgs/lstm_memory_benchmark.png' width="400px">
