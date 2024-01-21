#!/bin/bash -e

# to check jetpack version
# sudo apt-cache show nvidia-jetpack
# if not in latest 4.6.3 jetpack
# edit etc/apt/sources.list.d/nvidia-l4t-apt-source.list to point to the 32.7 repo (just change the version to r32.7 in both lines)
# sudo apt update
# sudo apt dist-upgrade

#password='jetson'
# Get the full dir name of this script
#DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Keep updating the existing sudo time stamp
#sudo -v
#while true; do sudo -n true; sleep 120; kill -0 "$$" || exit; done 2>/dev/null &

########################################
# Install DonkeyCar
########################################
# Install ubuntu modules
sudo apt-get update -y
sudo apt-get upgrade -y
# Do not restart docker daemon auto
sudo apt-get install -y git nano
sudo apt-get install -y libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
sudo apt-get install -y python3-dev python3-pip
sudo apt-get install -y openmpi-doc openmpi-bin libopenmpi-dev libopenblas-dev
sudo apt-get install -y libxslt1-dev libxml2-dev libffi-dev libcurl4-openssl-dev libssl-dev libpng-dev libopenblas-dev
sudo apt-get install python3-libnvinfer python3-libnvinfer-dev

# Install python modules
sudo pip3 install virtualenv
python3 -m virtualenv venvs/donkeycar --system-site-packages
source ~/venvs/donkeycar/bin/activate
echo "source ~/venvs/donkeycar/bin/activate" >> ~/.bashrc

python3 -m pip install pip testresources setuptools
# sudo ln -s /usr/include/locale.h /usr/include/xlocale.h
# python3 -m pip install cython-0.29.34 futures-2.2.0 protobuf-3.19.6 pybind11-2.10.4 pyserial-3.5 
python3 -m pip install cython futures protobuf pybind11 pyserial 
# python3 -m pip install numpy==1.19.5 future mock h5py keras==2.6.0 keras_preprocessing keras_applications gast==0.2.1 
python3 -m pip install future mock h5py keras==2.6.0 keras_preprocessing keras_applications gast 
# python3 -m pip install grpcio absl-py py-cpuinfo psutil portpicker wrapt==1.12.1 six requests 
python3 -m pip install grpcio absl-py py-cpuinfo psutil portpicker wrapt six requests 
python3 -m pip install astor termcolor google-pasta scipy pandas gdown pkgconfig packaging
python3 -m pip install depthai depthai-sdk
python3 -m pip install onnx==1.12.0


# upgrade default versions in 
python3 -m pip install scipy --upgrade
python3 -m pip install pandas --upgrade


python3 -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v502 tensorflow==2.10.1+nv22.12

python3 -m pip install pycuda

python3 -m pip install numpy==1.23 --upgrade


# python3 -m pip install Jetson.GPIO
#cd ~
#sudo cp robocar/lib/python3.6/site-packages/Jetson/GPIO/99-gpio.rules /etc/udev/rules.d/99-gpio.rules
#sudo udevadm control --reload-rules && sudo udevadm trigger
#sudo reboot now

# Install tensorflow
# python3 -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v461 tensorflow
# or upgrade
# python3 -m pip install --upgrade --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v461 tensorflow

echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger

sudo systemctl stop nvgetty
sudo systemctl disable nvgetty
sudo usermod -aG dialout $USER


# configure jetson clocks in a service
sudo systemctl enable jetson-clocks.service
sudo systemctl start jetson-clocks.service

# https://github.com/keras-team/keras-tuner/issues/317
echo "export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1" >> ~/.bashrc
#export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

echo "export PATH=/usr/local/cuda-11.4/bin:$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc

echo "export OPENBLAS_CORETYPE=ARMV8" >> ~/.bashrc

########################################
# Install PyTorch v1.7 - torchvision v0.8.1
# pytorch 1.6.0-rc2
# https://forums.developer.nvidia.com/t/pytorch-for-jetson-nano-version-1-5-0-now-available/72048/392
#wget https://nvidia.box.com/shared/static/wa34qwrwtk9njtyarwt5nvo6imenfy26.whl -O torch-1.7.0-cp36-cp36m-linux_aarch64.whl
#sudo -H pip3 install ./torch-1.7.0-cp36-cp36m-linux_aarch64.whl

# Install PyTorch Vision
#sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
#mkdir -p ~/projects; cd ~/projects
#git clone --branch v0.8.1 https://github.com/pytorch/vision torchvision
#cd torchvision
#export BUILD_VERSION=0.8.1
#sudo python3 setup.py install


# Create virtual enviroment
# pip3 install virtualenv
# python3 -m virtualenv -p python3 ~/.virtualenv/donkeycar --system-site-packages
# echo "source ~/.virtualenv/donkeycar/bin/activate" >> ~/.bashrc
# "source ~/.virtualenv/donkeycar/bin/activate" in the shell script
# . ~/.virtualenv/donkeycar/bin/activate


# Install DonkeyCar as user package
mkdir ~/github
cd ~/github
git clone https://github.com/roboracingleague/donkeycar
cd donkeycar
git checkout main
python3 -m pip install -e .\[nano\]



#export PATH=/usr/local/cuda-10.2/bin:$PATH
#export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH

#export PATH=/usr/local/cuda-11.4/bin:$PATH
#export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH



#sudo apt-get install python3-libnvinfer python3-libnvinfer-dev

# python3 -m pip install pyzmq-static

# python3 -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v502 tensorflow==2.10.1+nv22.12
# https://developer.download.nvidia.com/compute/redist/jp/v502/tensorflow/tensorflow-2.10.1+nv22.12-cp38-cp38-linux_aarch64.whl

donkey createcar --template=complete --path ~/mycar
cd ~/mycar
python manage.py drive

# add this line to /etc/fstab
tmpfs /home/drobee/mycar/data tmpfs defaults,size=2g 0 0
