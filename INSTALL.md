# Installation

## conda setup

Download miniconda, and download  version for python 3.6

```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

Run miniconda installation:
```
bash Miniconda3-latest-Linux-x86_64.sh
```

Add conda to your path:

```
source bashrc
```

To create a new environment for task name it gsv:

```
conda create -n gsv anaconda python=3
```

To activate a conda env

```
source activate gsv
```

use `source deactivate` to deactivate

## Install libraries

Install base libraries

```
conda install numpy mkl scipy scikit-learn nose sphinx tqdm Flask
```

Install deep learning framework pytorch

for the gpu:
```
conda install pytorch torchvision cuda80 -c soumith
```

for cpu only:
```
conda install pytorch torchvision -c soumith
```


## OSX install

```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
```

Similar instructions as for linux, but use the cpu only pytorch install.





