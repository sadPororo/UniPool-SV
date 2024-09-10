# Universal Pooling Method for Speaker Verification Utilizing Pre-trained Multi-layer Features
We share a PyTorch implementation of our experiments here.

The figure below overviews the proposed framework to pool speaker embedding from the multi-layered nature of pre-trained models.
<p align="center">
<img src="/img/Fig-Overall_framework_v0.png" width="900" height="290">
</p>

## Environment Supports & Python Requirements
![Ubuntu](https://img.shields.io/badge/Ubuntu-18.04+-E95420?style=for-the-badge&logo=ubuntu&logoColor=E95420)
![Python](https://img.shields.io/badge/Python-3.8.8-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12.0-%23EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=%23EE4C2C)   
* We recommend you to visit [Previous Versions (v1.12.0)](https://pytorch.org/get-started/previous-versions/#v1120) for **PyTorch** installation including torchaudio==0.12.0.

Use the [requirements.txt](/requirements.txt) to install the rest of the Python dependencies.   
**Ubuntu-Soundfile** and **conda-ffmpeg** packages would be required for downloading and preprocessing data, and you can install them as below.

```bash
$ pip install -r requirements.txt
$ apt-get install python3-soundfile
$ conda install -c conda-forge ffmpeg
```

## 1. Dataset Preparation

The datasets can be downloaded from here:

* [**VCTK CSTR Corpus**](https://doi.org/10.7488/ds/2645)

* [**LibriSpeech**](https://www.openslr.org/12)

* **VoxCeleb 1 & 2**  
  We use [clovaai/voxceleb_trainer](https://github.com/clovaai/voxceleb_trainer) to download VoxCeleb datasets, released under the MIT licence.  
  Follow the data preparation script, till you convert the VoxCeleb 2 audio format ```aac(.m4a)``` into ```.wav``` file.


## 2. Data Preprocessing & Evaluation Split
The following scripts are to preprocess audio data and build evaluation trials from each dataset.
* _However, you can skip the **set split** part, since we have uploaded the ready-made splits and trials used for our experiments in the file tree._  
  Please check ([data/VCTK-Corpus](data/VCTK-Corpus/preprocess) ; [data/LibriSpeech](data/LibriSpeech/preprocess) ; [data/VoxCeleb](data/VoxCeleb/preprocess))

#### VCTK CSTR Corpus  
```bash
# preprocessing
$ python ./src/preprocess/process-VCTK.py --read_path SRC_PATH
```
>Remove speaker [p280, p315] of risen technical issues.  
>Drop samples (no.000~no.024), where the same transcript is used for each number.  
>Resample audio sources to meet the sample rate in common (48K &rarr; 16K).

```bash
# set split
$ python ./src/preprocess/split-VCTK-0-speakers.py
$ python ./src/preprocess/split-VCTK-1-rawtrials.py
$ python ./src/preprocess/split-VCTK-2-balancedtrials.py
```
>Subset the total speaker pool into train, validation, and test speaker subsets.  
>Check the match of speaker meta-info (Gender | Age | Accents | Region | Label) given the total combination.  
>Sample the trials with a balance to the label distribution and meta-info matches.

#### LibriSpeech
```bash
# preprocessing
$ python ./src/preprocess/process-LibriSpeech.py --read_path SRC_PATH
```
>Convert audio format ```.flac``` to ```.wav``` file.

```bash
# set split
$ python ./src/preprocess/split-LibriSpeech-1-rawtrials.py
$ python ./src/preprocess/split-LibriSpeech-2-balancedtrials.py
```
>Check the match of speaker meta-info (Gender(SEX) | Label) given the total combination of samples.  
>Sample the trials with a balance to the label distribution and meta-info matches.

#### VoxCeleb 1 & 2  
```bash
$ mv ./data/VoxCeleb/*_wav/ ./data/VoxCeleb/preprocess/
```
>No special data preprocessing required.

```bash
# set split
$ python ./src/preprocess/split-VoxCeleb-0-speakers.py
$ python ./src/preprocess/split-VoxCeleb-2-balancedtrials.py
```
>List up the speakers in each subsets, and convert '[Vox1-O](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html)' evaluation path file format ```.txt``` to ```.csv```.  
>Sample the trials with a balance to the label distribution.

## 3. Run Experiments
```bash
$ python ./src/main.py -h
usage: main.py [1-action] [2-data] [3-model] [-h]

positional arguments (required):
  [1] action:  {train,eval}
  [2] data  :  {VCTK,LibriSpeech,Vox1-Base,Vox2-Base}
  [3] model :  {X-vector,ECAPA-TDNN,SincNet,ExploreWV2,FinetuneWV2,UniPool}

optional arguments in general:
  -h, --help                 show this help message and exit
  --quickrun                 quick check for the running experiment on the modification, set as True if given
  --skiptest                 skip evaluation for the testset during training, set as True if given
  --neptune                  log experiment with neptune logger, set as True if given
  --workers     WORKERS      the number of cpu-cores to use from dataloader (per each device), defaults to: 4
  --device     [DEVICE0,]    specify the list of index numbers to use the cuda devices, defaults to: [0]
  --seed        SEED         integer value of random number initiation, defaults to: 42
  --eval_path   EVAL_PATH    result path to load model on the "action" given as {eval}, defaults to: None
  --description DESCRIPTION  user parameter for specifying certain version, Defaults to "Untitled".

keyword arguments:
  --kwarg KWARG              dynamically modifies any of the hyperparameters declared in ../configs/.../...*.yaml or ./benchmarks/...*.yaml
  (e.g.) --lr 0.001 --batch_size 64 --nb_total_step 25000 ...
```

## License [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
This repository is released under the [MIT](https://choosealicense.com/licenses/mit/) license.
