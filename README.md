# MNIST Classification with ML and DL Models

This repository contains a notebook that explores various **Machine Learning (ML)** and **Deep Learning (DL)** models for classifying handwritten digits from the **MNIST** dataset. The goal is to gain familiarity with the practical application of ML and DL algorithms, comparing their performance on the MNIST dataset.

## 🚀 **Models Explored**

The notebook explores four different models:

1. **Logistic Regression** - A simple linear model for classification.
2. **Random Forest** - A popular non-linear model using ensemble learning.
3. **Fully Connected Neural Network (FCNN)** - A basic neural network consisting of fully connected layers.
4. **Convolutional Neural Network (CNN)** - A more advanced neural network designed for image data, utilizing convolutional and pooling layers.

### Performance Metrics

The models are evaluated using the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

These metrics are essential for multiclass classification tasks, which are particularly relevant for the MNIST dataset, where each image is a handwritten digit belonging to one of 10 classes.

## 📊 **Results & Insights**

- **Logistic Regression** showed the lowest performance with an accuracy of approximately **92%**.
- **Random Forest** and **FCNN** achieved **97%** accuracy, with **Random Forest** being faster to train.
- **CNN** outperformed all other models with an impressive **98.6%** accuracy, demonstrating the power of **Deep Learning** for image classification tasks.

Additionally, a **simple CNN** achieved **99.3%** accuracy after adding a single convolutional layer followed by pooling, further proving the effectiveness of convolutional layers for feature extraction from images.

## 🔄 **Learning Rate Experimentation**

The notebook also explores the impact of different learning rates on model convergence. The findings include:
- For **$lr=0.1$**, the loss stabilized without converging.
- Convergence was achieved for lower learning rates, with **$lr=0.01$** and **$lr=0.001$** offering the fastest convergence. However, smaller learning rates like **$lr=1e-4$** and **$lr=1e-5$** slowed down the convergence process.
- **$lr=1e-3$** was found to be the optimal learning rate for this setup.

## 📈 **Objective**

The objective of this project is to:
- Gain hands-on experience with classical ML and DL algorithms.
- Understand the strengths and weaknesses of models like **Logistic Regression**, **Random Forest**, and **Neural Networks**.
- Explore how **CNNs** can outperform **FCNNs** in image-related tasks.

## 🛠 **Technologies Used**
- **PyTorch** for implementing the neural networks.
- **scikit-learn** for the Random Forest model.
- **MNIST Dataset** for classification.
