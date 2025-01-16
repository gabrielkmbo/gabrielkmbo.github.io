---
title: "Convolutional Neural Network Bird Classifier"
excerpt: "Led the development of a custom bird classification model in PyTorch, achieving 87% accuracy on a dataset of 3,113 bird images. Utilized data augmentation and GAN-based image generation techniques to reduce overfitting by 20%, enhancing model generalization. This project demonstrated proficiency in deep learning and image classification, addressing challenges in dataset variability.






 <br/><img src='/images/bird.png'>"
collection: portfolio
---
[Visit the GitHub Repository](https://github.com/gabrielkmbo/bird-classifier)

## Stack

Here are the technologies used in this project:

<p>
  <img src="/images/aws.png" alt="AWS" title="AWS" width="40" height="40" />
  <img src="/images/gcp.png" alt="GCP" title="GCP" width="40" height="40" />
  <img src="/images/git.png" alt="Git" title="Git" width="40" height="40" />
  <img src="/images/firebase.png" alt="Firebase" title="Firebase" width="40" height="40" />
  <img src="/images/python.png" alt="Python" title="Python" width="40" height="40" />
  <img src="/images/github.png" alt="Github" title="Github" width="40" height="40" />
  <img src="/images/pytorch.png" alt="Pytorch" title="Pytorch" width="40" height="40" />
  <img src="/images/tensorflow.png" alt="Tensorflow" title="Tensorflow" width="40" height="40" />
  <img src="/images/hugging-face.png" alt="Hugging Face" title="Hugging Face" width="40" height="40" />
</p>

## Overview
- Developed a custom bird classification model using PyTorch with an accuracy of **87%** across 20 bird species from a dataset of 3,113 images.
- Focused on optimizing generalization and reducing overfitting, achieving improved performance on unseen test data.

## Model Architecture
- **Architecture Selection**:
  - Experimented with **Variational Autoencoder (VAE)** and **Generative Adversarial Network (GAN)**.
  - Chose a **Convolutional Neural Network (CNN)** due to its:
    - Superior feature detection capabilities.
    - Efficiency in training.
    - Proven effectiveness in handling image data.

### Sample Code of the Model
```python
class BirdModel(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        # input = 224 x 224 x 3
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(20 * 53 * 53, 50)
        self.fc2 = nn.Linear(50, 20)

    def forward(self, x):
        self.to(device)
        x = self.pool1(nn.functional.relu(self.conv1(x)))
        x = self.pool2(nn.functional.relu(self.conv2(x)))
        x = x.reshape(-1, 20 * 53 * 53)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
```

## Training Process
- **Data Augmentation Techniques**:
  - Implemented:
    - Image rotation.
    - Color fixation changes.
    - Random erasing.
    - Horizontal and vertical flips.
  - Designed to simulate diverse real-world scenarios and improve model robustness.
- **Hyperparameter Tuning**:
  - Extensive tuning led to a **15% improvement** in model precision and recall.
- **Loss Functions**:
  - Optimized loss functions tailored for classification tasks to enhance performance.

## Validation and Overfitting
- **Dataset Split**:
  - Used a **70-30 split** for training and validation.
  - **Limitation**: Absence of a distinct test set reduces ability to evaluate generalization on unseen data.
- **Overfitting Strategies**:
  - Implemented data augmentation techniques (e.g., random flips, color jittering).
  - Acknowledged the potential for advanced strategies like **dropout** or **weight decay**.

## Error Analysis
- Limited error analysis due to the lack of a test set.
- Recognized the importance of a comprehensive error analysis to:
  - Understand misclassifications and biases.
  - Identify patterns in errors.

## Future Improvements
- **Incorporate a Test Set**:
  - Evaluate performance on completely unseen data for better generalization.
- **Conduct Detailed Error Analysis**:
  - Hypothesize causes of errors and refine the model to enhance robustness.
- **Explore Advanced Techniques**:
  - Introduce regularization and advanced data augmentation to further mitigate overfitting.

