# dealer_deepface

API para la comparacion de rostros

## DOCKER
docker build -t deepface .

## Instalacion
Pasos para instalar desde la linea de comandos de linux / ubuntu

```shell
sudo apt update && sudo apt upgrade
sudo apt install git
sudo apt install vim

sudo apt install python3-pip
sudo apt install cmake

export PATH="/home/usuario/.local/bin:$PATH"
source ~/.bashrc

sudo reboot

sudo apt install -y libgl1-mesa-glx
sudo ldconfig

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
export PATH="$HOME/miniconda3/bin:$PATH"
bash Miniconda3-latest-Linux-x86_64.sh
rm Miniconda3-latest-Linux-x86_64.sh
sudo reboot

conda --version
conda install -c conda-forge deepface

sudo ufw allow 5000/tcp

git clone https://sk-software1-admin:HF2gAgnsvKNrRUuDyWtB@bitbucket.org/sk-software1/dealer_deepface.git

cd dealer_deepface

pip install -r requirements.txt
pip install -r requirements_additional.txt

cd api
python3 api.py
```


## Mas info y referencias
Link repositorio
[serengil/deepface: A Lightweight Face Recognition and Facial Attribute Analysis (Age, Gender, Emotion and Race) Library for Python (github.com)](https://github.com/serengil/deepface)

Ejemplo de como implementar una API
[DeepFace API for Face Recognition and Facial Attribute Analysis - YouTube](https://www.youtube.com/watch?v=HeKCQ6U9XmI&feature=youtu.be)

Como instalar TensorFlow en ubuntu
[Install and Use TensorFlow on Ubuntu 22.04 – Its Linux FOSS](https://itslinuxfoss.com/install-use-tensorflow-ubuntu-22-04/)