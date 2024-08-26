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


## 2. Data Preprocessing & Evaluation Set Split
#### 2-1. VCTK CSTR Corpus  
```bash
$ python ./src/preprocess/process-VCTK.py
```
* Preprocessing includes,
    - remove speaker [p280, p315] of risen technical issues.  
    - drop samples (no.000~no.024), where the same transcript is used for each number.  
    - resample audio sources to meet the sample rate in common (48K &rarr; 16K).

* Split includes,  
    - aaaaa

#### 2-2. LibriSpeech
```bash
$ python ./src/preprocess/process-LibriSpeech.py
```
* Preprocessing includes,
    - convert audio format ```.flac``` into ```.wav``` file.
* Split includes,  
    - aaaaa

#### 2-3. VoxCeleb 1 & 2  
```bash
$ python ./src/preprocess/process-VoxCeleb.py
```
* Preprocessing includes,  
    - convert audio format ```.flac``` into ```.wav``` file.
* Split includes,  
    - aaaaa


## License [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
This repository is released under the [MIT](https://choosealicense.com/licenses/mit/) license.
