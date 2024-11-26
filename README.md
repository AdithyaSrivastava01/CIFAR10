# CIFAR-10 Classification with ResNet9

---

## Project Overview
This project implements a deep learning pipeline to classify images from the CIFAR-10 dataset using a custom ResNet9 architecture. The model is trained using state-of-the-art optimization techniques to achieve robust and efficient performance. The project is implemented in PyTorch and incorporates advanced methods such as One-Cycle Learning Rate Scheduling and Gradient Clipping to enhance training.

---

## Dataset Description
The **CIFAR-10 dataset** is a collection of 60,000 color images in 10 classes, with 6,000 images per class. Each image is 32x32 pixels and contains one of the following categories:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

### Data Split
- **Training Set**: 50,000 images
- **Test Set**: 10,000 images

### Preprocessing
- The dataset is normalized to zero mean and unit variance for all three RGB channels.
- Data augmentation techniques such as random cropping and flipping may be applied to enhance model generalization.

---

## Model Architecture
The project employs a **ResNet9-inspired architecture**, which balances model depth and computational efficiency. It incorporates residual connections to allow deeper networks to train effectively by mitigating vanishing gradient issues.

### Architecture Details
1. **Convolutional Blocks**:
   - Feature extraction using 3x3 convolutions, batch normalization, and ReLU activation.
   - Downsampling with MaxPooling layers where needed.

2. **Residual Layers**:
   - Two residual blocks are included, each comprising two convolutional layers with a skip connection to add the input to the output.

3. **Classifier**:
   - A global max-pooling layer reduces the feature map size.
   - A fully connected layer outputs the class probabilities.

### Flow of Layers
| **Layer**            | **Output Shape**        | **Description**                              |
|----------------------|-------------------------|----------------------------------------------|
| Input                | (3, 32, 32)            | RGB image input                              |
| `conv1`              | (64, 32, 32)           | Convolutional block with 64 channels         |
| `conv2`              | (128, 16, 16)          | Convolutional block with pooling             |
| `res1`               | (128, 16, 16)          | Residual block with 128 channels             |
| `conv3`              | (256, 8, 8)            | Convolutional block with pooling             |
| `conv4`              | (512, 4, 4)            | Convolutional block with pooling             |
| `res2`               | (512, 4, 4)            | Residual block with 512 channels             |
| Classifier           | (10)                   | Fully connected layer with softmax activation|

**Key Features**:
- **Residual Connections**: Improves gradient flow and allows deeper networks to be trained effectively.
- **Dropout**: A dropout layer (20%) in the classifier reduces overfitting.
- **Batch Normalization**: Speeds up convergence and stabilizes training.

---

## Training Configuration
The model training process leverages advanced optimization techniques to ensure efficiency and robustness:

### Training Components
1. **Optimizer**:
   - **Adam** optimizer is used, which adjusts learning rates dynamically for each parameter.
   - **Weight Decay**: Regularization term (`1e-4`) penalizes large weights to reduce overfitting.

2. **Loss Function**:
   - **Cross-Entropy Loss** is used for multi-class classification, measuring the difference between predicted and true class probabilities.

3. **Learning Rate Scheduler**:
   - **One-Cycle Learning Rate Policy** dynamically adjusts the learning rate:
     - Increases it linearly to the peak value (`max_lr = 0.01`) during the first half of training.
     - Decreases it exponentially in the second half.
   - **Benefits**: Faster convergence and better generalization.

4. **Gradient Clipping**:
   - Gradients are clipped (`grad_clip = 0.1`) to prevent instability caused by exploding gradients in deep networks.

5. **Epochs**:
   - The model is trained for `8 epochs`, iterating over the entire training dataset.

---

## Summary of Custom Training Techniques

| **Technique**               | **Purpose**                                                                                     | **Advantages**                                                                                                   |
|------------------------------|-------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| **Adam Optimizer**           | Optimizes model weights with an adaptive learning rate for each parameter.                     | Ensures faster convergence and effective handling of sparse gradients.                                          |
| **Weight Decay**             | Regularizes model weights to prevent overfitting.                                              | Promotes simpler and more generalizable models.                                                                 |
| **One-Cycle Learning Rate**  | Dynamically adjusts learning rate during training.                                             | Improves convergence speed and enhances generalization.                                                         |
| **Gradient Clipping**        | Caps gradients during backpropagation to prevent instability.                                  | Stabilizes training by preventing exploding gradients.                                                          |
| **GPU Acceleration**         | Moves computations to the GPU for increased speed.                                             | Drastically reduces training time for large datasets and deep networks.                                         |
| **Batch-wise Training**      | Processes data in mini-batches.                                                               | Efficient memory utilization and faster parameter updates.                                                      |
| **Epoch Logging**            | Tracks the progress of training after each epoch.                                              | Helps monitor performance and detect training issues early.                                                     |
