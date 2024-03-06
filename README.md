# Project Title: FrameFlow
FrameFlow: Video Classification with Pre-trained CNNs
This project, titled 'FrameFlow', utilizes the power of convolutional neural networks (CNNs) pre-trained on large datasets to perform video classification tasks. The focus of FrameFlow is to apply transfer learning techniques to adapt these sophisticated models to the domain of action recognition in video data.

Overview
FrameFlow is a video classification project that leverages the EfficientNetB0 pre-trained model and adapts it for action recognition on the UCF101 dataset. The key components of this project include:

Dataset Preparation: Organize and prepare video data, extracting frames and corresponding labels.
Model Selection: Employ the EfficientNetB0 model, a pre-trained model known for its efficiency and accuracy.
Transfer Learning: Fine-tune the pre-trained model on a subset of the UCF101 dataset, targeting action recognition.
Evaluation: Assess model performance using metrics such as accuracy, F1 score, and a confusion matrix.

## NON-TECHNICAL EXPLANATION OF YOUR PROJECT

FrameFlow is a project that aims to classify videos into different categories based on the actions or events happening in them. It leverages the power of deep learning techniques and pre-trained models to achieve this task efficiently. The project uses a technique called "transfer learning," where a pre-trained model (a model that has already learned to recognize visual patterns from a large dataset) is fine-tuned on a specific video dataset to adapt it for video classification.

## DATA

The project uses the UCF101 dataset, which is a widely used benchmark for video classification tasks. The dataset consists of 13,320 videos from 101 different action categories, including human actions like "Basketball Shooting," "Bowling," "Horse Riding," and many more. The videos are split into training, validation, and testing sets to train and evaluate the model effectively. More information is on the data sheet on this project and on https://www.crcv.ucf.edu/research/data-sets/ucf101/

![UCF action categores](UCF101.jpg)

## HYPERPARAMETER OPTIMSATION

One of the critical aspects of training deep learning models is finding the optimal hyperparameters, such as learning rate, batch size, and number of epochs. In this project, various techniques were employed to optimize these hyperparameters, including:

Early Stopping: A technique to prevent overfitting by stopping the training process when the model's performance on the validation set stops improving. This was observed in the main project where early stopping occurred after 7 epochs - indicating that the model had converged.

Bayesian Optimization: A method that uses probabilistic models to search for the optimal set of hyperparameters efficiently. Trying different exploration/exploitation methods, different kernels and varying hyper paramaters there were a number of issues where often

Grid Search: An exhaustive search approach where a range of hyperparameter values is specified, and the model is trained and evaluated for each combination.

## RESULTS

The FrameFlow project achieved promising results on the UCF101 dataset. The trained model, using the EfficientNetB0 pre-trained model and custom layers for video classification, achieved a test accuracy of approximately 92.76%. This means that the model correctly classified nearly 93% of the videos in the test set, which was not used during training.

Additionally, the project provides visualizations and metrics to evaluate the model's performance in-depth, including:

Confusion Matrix: A visual representation of the model's performance, showing the number of correct and incorrect predictions for each class.
F1 Scores: A metric that combines precision and recall, providing insights into the model's performance for each class.
Training and Validation Curves: Plots displaying the model's loss and accuracy during the training process, helping identify potential overfitting or underfitting issues.
These results demonstrate the effectiveness of the FrameFlow approach in leveraging pre-trained models and transfer learning for video classification tasks.

## OTHER MODELS
Initially, I explored using a ResNet model on a subset of the UCF101 dataset, following a TensorFlow documented example. However, I found this approach to be computationally challenging and underperforming on my local machine.
