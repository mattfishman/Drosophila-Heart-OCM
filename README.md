# Drosophila-Heart-OCM
This repository contains the code used to train an LSTM convolutional neural network to perform fly heart segmentation. Additional details can be found within the maniscript, this readme will focus on running FlyNet models and predictions using this repository. 

## Requirements
The code was tested on a system with the following specifications:
* Windows 10
* NVIDIA GeForce RTX 3090
* AMD Ryzen 9 5950X
* 128 GB RAM

The following software is required to run the code:
* Python 3.9
* Tensorflow 2.10
* CUDA 11.2
* cuDNN 8.1.0
* scikit-image
* opencv-python

## Installation
Full tutorial for installing TensorFlow for Windows Native can be found [here](https://www.tensorflow.org/install/pip#windows-native). A brief summary is provided below for convenience.

```bash
conda create --name Drosophila_OCM python=3.9
conda activate Drosophila_OCM
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install --upgrade pip
pip install "tensorflow<2.11"
pip install scikit-image opencv-python
```

## Usage
### Training
To train a model, open the LSTM_scientific_data.ipynb notebook and run the cells in order. You will first need to download the data from Figshare and then update the data paths in the notebook. The notebook will train a model and save the weights to the output folder that also needs to be updated.

The notebook is configured for additional hyperparameter tuning to make it easier to run on different systems. Additionally, the model being published can be loaded in and fine tuned using the notebook rather than starting from scratch.

### Predictions
For predictions we created a simple program predict_video.py that can be run from the command line to predict on a video. The program takes a path to a 128 x 128 video and outputs a 128 x 128 mask in the same location as the original video. The program can be run as follows:

```bash
python predict_video.py --video_path="path/to/video"
```

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Acknowledgements
The code for the loss_funcs.py file was adapted from the following repository:
[SemSegLoss](https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions/tree/master#semantic-segmentation-loss-functions-semsegloss)