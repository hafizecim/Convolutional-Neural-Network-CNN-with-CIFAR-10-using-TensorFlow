---

# Convolutional Neural Network (CNN) with CIFAR-10 using TensorFlow

This project demonstrates how to build and train a **Convolutional Neural Network (CNN)** using **TensorFlow/Keras** on the **CIFAR-10** dataset.
The goal is to understand deep learning basics, data preprocessing, CNN architecture design, training, evaluation, and visualization.

---

## 🔍 Project Overview

* **Dataset:** CIFAR-10 (60,000 images, 10 classes)

  * Training: 50,000 images
  * Test: 10,000 images
* **Model:** Convolutional Neural Network (Sequential)
* **Framework:** TensorFlow / Keras
* **Task:** Image Classification (10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

---

## 🚀 Features

* Load and preprocess CIFAR-10 dataset
* Normalize image pixels (0–1 scaling)
* Visualize the first 25 training images in a 5x5 grid
* Build a CNN with:

  * Conv2D + ReLU
  * MaxPooling2D
  * Flatten
  * Dense layers
* Compile model using **Adam optimizer** and **SparseCategoricalCrossentropy**
* Train and validate the model for **10 epochs**
* Plot **accuracy** and **loss curves**
* Evaluate on test data
* Generate **confusion matrix** to visualize predictions

---

## 🧱 Model Architecture

The CNN architecture is as follows:

| Layer (Type)   | Output Shape | Param # | Description                           |
| -------------- | ------------ | ------- | ------------------------------------- |
| Conv2D         | (30, 30, 32) | 896     | 32 filters, 3x3 kernel, ReLU          |
| MaxPooling2D   | (15, 15, 32) | 0       | 2x2 pooling                           |
| Conv2D         | (13, 13, 64) | 18,496  | 64 filters, 3x3 kernel, ReLU          |
| MaxPooling2D   | (6, 6, 64)   | 0       | 2x2 pooling                           |
| Conv2D         | (4, 4, 64)   | 36,928  | 64 filters, 3x3 kernel, ReLU          |
| Flatten        | (1024)       | 0       | Converts 2D feature maps to 1D vector |
| Dense          | (64)         | 65,600  | Fully connected layer, ReLU           |
| Dense (Output) | (10)         | 650     | Output layer for 10 classes           |

**Total trainable parameters:** 122,570

---

## 📊 Training & Results

* **Epochs:** 10
* **Training Accuracy:** ~0.7516
* **Test Accuracy:** ~0.6574

### Accuracy & Loss Curves

*(Add matplotlib plots here from Colab)*

### Confusion Matrix

*(Add seaborn heatmap here from Colab)*

---

## ▶️ How to Run

1. Open the notebook in Google Colab
2. Run all cells in order
3. Visualize training progress with accuracy & loss plots
4. Evaluate final model on test data

---

## 📚 Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Seaborn
* Google Colab

---

## 👤 Author

Created by **Hafize Şenyıl** using Google Colab.
Feel free to use, fork, or improve this project.

---

## ⭐ Future Improvements

* Add **Dropout** & **Batch Normalization** layers
* Use **Data Augmentation** to improve generalization
* Experiment with deeper architectures (e.g., VGG-like)
* Deploy as a **web application** or **API**

---
