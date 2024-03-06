# Model Card
Model Card for EfficientNetB0 with Custom Layers for UCF101 Video Classification.  

## Model Description
Goal: To create a cutting-edge video classification system capable of recognizing a broad range of human actions within the UCF101 dataset.

Relevance: Video classification has vital applications in security, entertainment, healthcare, and sports analytics, where it can aid in surveillance, content curation, patient monitoring, and performance analysis.

Challenges: This project addresses high-dimensional data management, the extraction of temporal dynamics from actions, and achieving model robustness under varying video conditions.

Innovative Approach: Utilizing 3D CNNs to process spatial-temporal data, complemented by transfer learning strategies, the project aims to set a new benchmark in action recognition.

Dataset Overview: The UCF101 dataset, with 101 action categories from various real-world videos, serves as the foundation for this research.

Development Journey:

Initial explorations utilized TensorFlow/Keras frameworks.
Training a multi-layer CNN proved computationally intensive.
Implemented GPU acceleration for improved computational efficiency.
Discovered EfficientNetB0 with custom layers to be highly efficient for training and development.
Compared ResNet and EfficientNetB0 on the full dataset; EfficientNetB0 exhibited superior performance.
Employed Bayesian optimization and grid search to identify the optimal learning rate.

**Input:** Describe the inputs of your model 

**Output:** Describe the output(s) of your model

**Model Architecture:** Describe the model architecture you’ve used
Model Architecture: The architecture is a modified EfficientNetB0, optimized for video data, integrating custom layers designed to capture temporal information and to address the specific challenges of video classification.

Architecture Overview:
The model is an innovative adaptation of the EfficientNetB0 architecture, designed to harness its efficiency and accuracy for video classification tasks. The architecture is augmented with custom layers, making it adept at handling the spatial-temporal dynamics present in video data.

Input Layer:
The input to the model is a sequence of video frames, each with the dimension of 224x224 pixels with 3 color channels (RGB). These are processed as batches, with the first dimension representing the batch size and the second dimension representing the sequence length or the number of frames in the video.

Preprocessing Layer:
A rescaling layer is applied to each frame to normalize pixel values, ensuring that the input data is standardized before entering the core model.

TimeDistributed Layer:
The core of the model employs the TimeDistributed wrapper around the EfficientNetB0 network. This allows the EfficientNetB0 architecture to process each frame individually while sharing the weights across the temporal dimension, thereby capturing spatial features within each frame.

Dense Layer:
Following the EfficientNetB0, a dense layer with 101 units corresponds to the number of action classes in the UCF101 dataset. This layer is crucial for learning the discriminative features that differentiate between various actions.

Global Average Pooling Layer:
A global average pooling layer follows the dense layer, aggregating the temporal information by averaging over the time dimension. This step condenses the spatial-temporal features extracted by previous layers into a singular feature vector.

Output:
The final output of the model is a probability distribution over the 101 action classes, providing a prediction for the video's classified action.

Customization for Temporal Dynamics:
The custom layers added to the EfficientNetB0 are specifically tailored to capture the temporal information inherent in video sequences, which is crucial for accurately classifying dynamic actions. By combining the powerful feature extraction capabilities of EfficientNetB0 with these custom temporal layers, the model achieves a nuanced understanding of both spatial and temporal aspects of the data.

Additional information about EfficientNetB0:

EfficientNetB0 is a convolutional neural network that's part of the EfficientNet family, which was introduced in the paper "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks". The EfficientNets are known for their efficiency in terms of both accuracy and computational resources. This efficiency is achieved through a systematic study of model scaling, leading to a family of models (B0 through B7) that are scaled up starting from a baseline model (B0).

The main features of EfficientNetB0 include:

Depthwise Separable Convolutions: These are used to reduce the number of parameters and computational cost compared to regular convolutions.

MBConv Blocks: An inverted residual structure where the input and output are connected through a shortcut with lightweight depthwise convolutions.

Compound Scaling Method: EfficientNets use a compound coefficient to uniformly scale the network’s width, depth, and resolution.

Squeeze-and-Excitation Blocks: These are included to recalibrate the feature channels, emphasizing important features selectively.

In terms of architecture, EfficientNetB0 includes multiple stages with MBConv blocks, starting with a stem convolutional layer, and ending with a top classifier that typically consists of a pooling layer and a fully connected layer.


## Performance

Give a summary graph or metrics of how the model performs. Remember to include how you are measuring the performance and what data you analysed it on. 


## Limitations

While the model performs robustly on the UCF101 dataset, limitations arise in scenarios with low-light conditions, subtle movements, and actions outside the scope of the dataset. It may also exhibit biases inherent in the dataset, reflecting the demographic and action-type representation within the training data.

## Trade-offs

The model design considers a trade-off between computational efficiency and accuracy. EfficientNetB0 offers an optimal balance for rapid training while maintaining high accuracy. Nonetheless, certain edge cases or nuanced actions may challenge the model, indicating a need for further fine-tuning or additional training data.
