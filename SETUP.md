# Setup on Ubuntu 18.04

## Prerequsities

Install Anaconda 3 from https://www.anaconda.com.

Install the `build-essential` package. // TODO install gcc 6 instead of 7

## NVIDIA prep

Follow instructions for installation of CUDA 9, the 17.04 network version. 
Install `cuda-9-0` instead of just `cuda`. You'll
run into a problem with one package, `nvidia-390`, which wants to overwrite a 
file from another package, `libglx-mesa0`. See https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-390/+bug/1753796 
for details. Execute 

    $ sudo apt-get -o Dpkg::Options::="--force-overwrite" install --fix-broken

to strongarm it through.

You also have to join the NVIDIA Developer program and download the **cuDNN** runtime 
library from https://developer.nvidia.com/rdp/cudnn-download.

## Set up Anaconda environment

    $ conda create --name sg1 python=3.6
    $ conda activate sg1
    $ pip install pillow typeguard requests tensorflow-gpu

## Download pretrained model

Download the model data from https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ and put the
file in the `models/` subdirectory of this project.

