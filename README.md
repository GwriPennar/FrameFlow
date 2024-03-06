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

 - In the context of the FrameFlow project, which involves training a deep neural network on a large video dataset, early stopping was a good idea for the following reasons:
 - Prevent Overfitting: As the model trains for more epochs, it may start to memorize the training data instead of learning the underlying patterns. Early stopping helps to stop the training process before the model overfits, ensuring that it generalizes well to new video samples.
 - Efficient Training: Training deep learning models can be computationally expensive, especially when working with large datasets like UCF101. Early stopping allows the training process to terminate automatically when the model's performance stops improving on the validation set, saving computational resources and time.
 - Avoid Degradation: In some cases, continuing to train a model beyond a certain point can lead to a degradation in performance on both the training and validation sets. Early stopping prevents this degradation by stopping the training process at the optimal point.
- Hyperparameter Tuning: Early stopping can be used in conjunction with other hyperparameter tuning techniques, such as Bayesian optimization and grid search. By incorporating early stopping, the hyperparameter search can be more efficient and effective, as the training process terminates automatically when the optimal performance is achieved.

In the FrameFlow project, early stopping was implemented using a callback function provided by TensorFlow/Keras. This callback monitored the validation loss during training and stopped the training process if the validation loss did not improve for a specified number of epochs (in this case, 2 epochs). This approach helped to prevent overfitting, optimize computational resources, and ensure that the model's performance was maximized on the validation set, which is a good indicator of its generalization capability.

Bayesian Optimization: A method that uses probabilistic models to search for the optimal set of hyperparameters efficiently. There were a number of challenges and the focus was on the learning rate for the main training loop. 

 - Complexity and Sensitivity: Bayesian optimization, while powerful, can be quite complex and sensitive to the choice of kernel, acquisition function, and their hyperparameters. In our case, we experimented with different kernels (RBF, Matern) and adjusted various parameters (length_scale, noise_level, beta for exploration-exploitation balance), but still faced issues with convergence and the learning rate getting stuck at suboptimal values.

 - Narrow Search Space: Given the performance of the default learning rate in TensorFlow's Adam optimizer, our search space was relatively narrow, centered around this default value. Bayesian optimization is generally more advantageous in larger, high-dimensional search spaces where random or exhaustive searches are impractical. In our narrow search space, the added complexity of Bayesian optimization did not provide a significant advantage over simpler methods.

 - Repeatability and Stability: The stochastic nature of Bayesian optimization, coupled with the non-convexity of neural network training, led to less repeatable and stable outcomes. This variability made it difficult to confidently identify the optimal learning rate.

- Performance Overhead: Bayesian optimization involves fitting a probabilistic model (Gaussian Process in my case) to the observed outcomes, which introduces additional computational overhead. For a relatively simple hyperparameter like the learning rate and a narrow search space, this overhead might not be justified, especially when simpler methods can achieve comparable results with less complexity.

 - Model Specificity: The effectiveness of Bayesian optimization can be highly dependent on the specific model and task. In our case, with a pre-trained EfficientNetB0 and a focus on fine-tuning with additional custom layers for video classification, the nuances of this setup might not have aligned well with the assumptions and strengths of Bayesian

Grid Search: An exhaustive search approach where a range of hyperparameter values is specified, and the model is trained and evaluated for each combination.

 - The best learning rate found was 0.0015, with the corresponding lowest validation loss being approximately 0.0791. This suggests that among the explored learning rates, 0.0015 was the most effective for training the model, balancing the speed of learning and the model's ability to generalize well to new data. The grid search was straightforward and provided a clear outcome, making it a suitable choice for optimizing the learning rate in this scenario.

Batch Size: This was set to 2 to manage the model's memory usage effectively and maintain consistency with previous parameter settings. This small batch size facilitated more frequent model updates, potentially leading to faster convergence. However, it's important to consider that the optimal batch size can vary based on the task, model architecture, and available computational resources.

## RESULTS

Training/Validation/Test Scores:

- The model achieved a test accuracy of approximately 92.76% on the UCF101 dataset.
- The training accuracy reached 97.54%, and the validation accuracy was 92.04% after 50 epochs.

F1 Scores:
- The project provides visualizations of the F1 scores for each class.
- The average F1 score across all classes was 0.927.

![F1 Scores](Final_F1_Scores.png)

Confusion Matrix:

- The confusion matrix visualizations show the number of correct and incorrect predictions for each class.
- These matrices help identify classes that the model may have struggled with or performed well on.

Precision and Recall:

- Precision by Class: This chart displays the precision for each class in your dataset. Precision is a measure of how many of the items identified as belonging to a class actually do belong to that class. A high precision score indicates that when the model predicts a class, it is very likely to be correct. You'll want to look for classes with lower precision which might indicate a higher number of false positives for those classes. For example, if your model is predicting that a video contains a specific activity, but it doesn't, that would lower the precision.

- Recall by Class: The recall chart shows how many of the actual items of a class were correctly identified. It measures the model's ability to detect all relevant instances in a dataset. A high recall score means the model is good at detecting the class, but it doesn't tell you how many other classes were incorrectly labeled as belonging to the class in question. For classes with lower recall, it means there are more false negatives – instances that belong to a class which the model failed to recognize.

Training and Validation Curves:

These curves help identify potential overfitting or underfitting issues and monitor the model's performance over time.
Overall, the results demonstrate the effectiveness of the FrameFlow approach in leveraging pre-trained models and transfer learning for video classification tasks. The model achieved high accuracy on the test set, and the provided visualizations and metrics offer insights into the model's performance across different classes and during the training process.

![Accuracy over Epochs](Final_Accuracy.png)

Additionally, the project provides visualizations and metrics to evaluate the model's performance in-depth, including:

Confusion Matrix: A visual representation of the model's performance, showing the number of correct and incorrect predictions for each class.
F1 Scores: A metric that combines precision and recall, providing insights into the model's performance for each class.
Training and Validation Curves: Plots displaying the model's loss and accuracy during the training process, helping identify potential overfitting or underfitting issues.
These results demonstrate the effectiveness of the FrameFlow approach in leveraging pre-trained models and transfer learning for video classification tasks.

## OTHER MODELS
Initially, I explored using a ResNet model on a subset of the UCF101 dataset, following a TensorFlow documented example. However, I found this approach to be computationally challenging and underperforming on my local machine.
