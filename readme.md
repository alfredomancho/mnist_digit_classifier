# Digit Recognition Demo

## Overview
This project trains a small neural network to classify images from the MNIST handwritten digit dataset. It demonstrates:

- Data loading & normalization in Python  
- Building and compiling a Keras model  
- Saving the trained model
- Training with validation split  
- Evaluating model accuracy  
- Visualizing loss curves
- Testing the trained model on user generated test images (PNG) of handwritten digits of any size and colour

## Structure
- **model.py**: Complete script for data prep, model creation, training, evaluation, and plotting.  Saves trained model to .h5 file.
- **load_model_predict_one_digit.py** Script for preprocessing the user generated test images and making the prediction using the trained model.
- **loss_curves.png**: Sample plot showing training and validation loss over epochs.
- **my_digit6_rotated_lightb.png**: User-generated test digit image
## How to Run
1. **Install dependencies**  
   ```bash
   pip install tensorflow matplotlib opencv-python

