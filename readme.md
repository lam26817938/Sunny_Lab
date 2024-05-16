## Install VS code:
```bash
sudo apt install code
```
## Install python packages:

Note: please either creat a virtual environment or add --break-system-package to the command
```bash
pip install -r requirements.txt
or
pip install opencv-python
pip install tensorflow
```
## if you face issue with h5py, please run the following command:
```bash
sudo apt-get install libhdf5-dev
```

# Some useful commands:

## list the camera devices:
```bash
sudo apt install v4l-utils
v4l2-ctl --list-devices
```

## update the tools:
```bash
sudo apt update
python -m pip install --upgrade pip
pip install --upgrade setuptools wheel
```