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

```bash
$ pip install -r requirements.txt
$ apt-get install python3-soundfile
$ conda install -c conda-forge ffmpeg
```


## License [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
This repository is released under the [MIT](https://choosealicense.com/licenses/mit/) license.
