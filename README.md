# wavegru-vocoder
WaveGRU vocoder


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
python tf_data.py <ft_dir> <tf_data_dir>
```

Step 4: train wavegru

```sh
python train.py
```