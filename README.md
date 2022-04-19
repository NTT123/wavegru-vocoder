# wavegru-vocoder
WaveGRU vocoder for TTS.

**Visit [this](https://huggingface.co/spaces/ntt123/WaveGRU-Text-To-Speech) hugging-face-space for a live demo.**


## Introduction

This repo implements a WaveRNN vocoder from the paper [Efficient Neural Audio Synthesis](https://arxiv.org/abs/1802.08435).

- We predict a 8-bit mu-transformed signal instead of the 16-bit signal in the paper.
  + We use a pre-emphasis filter with `coef=0.86` to achieve good speech quality with only 8-bit signal (as used by [LPCNet](https://jmvalin.ca/papers/lpcnet_icassp2019.pdf)).
- We use the upsampling network from [Lyra](https://github.com/google/lyra).
- We follow the prunning procedure in the WaveRNN paper.
  + However, only the WaveRNN network is prunned to 95% sparsity. The upsampling network is not prunned.
- We use Lyra sparse matmul library for fast inference on CPU for the live demo. Visit [here](https://huggingface.co/spaces/ntt123/WaveGRU-Text-To-Speech/tree/main) for the source code of the live demo.


## Instructions

Step 1: download data

```sh
python ljs.py
```


Step 2: extract mel features and mu waveform

```sh
python extract_mel_mu.py <wav_dir> <ft_dir>
```

Step 3: prepare tf dataset

```sh
python tf_data.py <ft_dir>
```

Step 4: train wavegru vocoder

```sh
python train_on_tpu.py
```
