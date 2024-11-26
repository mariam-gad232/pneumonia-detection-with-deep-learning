
# Pneumonia Detection Using Deep Learning 🩺🧠

This project demonstrates the use of transfer learning with the **VGG16** model to detect pneumonia from chest X-ray images. The model classifies X-ray images into two categories: **Normal** and **Pneumonia**.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Evaluation](#evaluation)
- [How to Use](#how-to-use)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Acknowledgements](#acknowledgements)

---

## Overview

Pneumonia is a serious respiratory condition that can be detected using chest X-rays. This project applies **deep learning** to automate the classification of chest X-rays into two categories:
1. **Normal** 
2. **Pneumonia**

By leveraging **transfer learning** with the **VGG16 model**, this approach achieves high accuracy while minimizing training time.

---

## Dataset

The dataset used in this project is publicly available and contains chest X-ray images categorized into:
- **Training Set**: 5,216 images
- **Test Set**: 624 images

Dataset Structure:
```
data/chest_xray/
│
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
└── val/
    ├── NORMAL/
    └── PNEUMONIA/
```

---

## Model Architecture

The project uses the **VGG16** model pre-trained on ImageNet as the base. The top layers are replaced with:
1. **Flatten Layer**
2. **Dense Layer (64 neurons, ReLU activation)**
3. **Dense Layer (32 neurons, ReLU activation)**
4. **Output Layer (2 neurons, Softmax activation)**

The VGG16 layers are frozen to utilize the pre-trained features while training only the added layers.

---

## Preprocessing

### **Data Augmentation**
- Rescaling pixel values to [0, 1]
- Random shear transformations
- Random zoom
- Horizontal flipping

### **Input Size**
- Images are resized to **224x224x3** to match the VGG16 input requirement.

---

## Training

The model is compiled with:
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy

### **Training Parameters**
- **Batch Size**: 10
- **Epochs**: 1 (for demonstration purposes, can be increased for better results)
- **Steps Per Epoch**: Number of training batches
- **Validation Steps**: Number of test batches

---

## Evaluation

### **Performance Metrics**
After 1 epoch:
- **Training Accuracy**: ~89.61%
- **Validation Accuracy**: ~88.30%
- **Validation Loss**: 0.2961

---

## How to Use

### **Requirements**
- Python 3.x
- TensorFlow/Keras
- NumPy
- Matplotlib

### Clone the Repository

To get a copy of the project on your local machine, run the following command:

```bash
git clone https://github.com/mariam-gad232/pneumonia-detection-with-deep-learning.git
```

## Results

The model predicts whether the person is affected by pneumonia or not based on chest X-ray images. For example:
- Input: X-ray image
- Output: `Result is Normal` or `Person is Affected By PNEUMONIA`

---

## Technologies Used

- **TensorFlow/Keras**: Deep learning framework
- **VGG16**: Pre-trained model for transfer learning
- **NumPy**: Numerical computations
- **Matplotlib**: Visualization

---

## Acknowledgements

- Dataset from [Chest X-Ray Images](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).
- VGG16 pre-trained weights from Keras.

---

## Future Work

- Increase the number of epochs for better performance.
- Experiment with other pre-trained models like ResNet50 or InceptionV3.
- Add explainability techniques like Grad-CAM to visualize model focus areas.

------
Feel free to raise an issue or contribute to this project! 🚀

