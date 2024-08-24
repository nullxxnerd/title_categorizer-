Overview

This project aims to develop a product categorization system using deep learning. The input to the model is a text description of a product, and the output is the predicted category. The model uses a CNN-based architecture, which is effective in capturing local patterns in the text data, making it suitable for this classification task.
Model Architecture

The model is built using the Sequential API from TensorFlow/Keras and consists of the following layers:

    Embedding Layer: Transforms the input text into dense vectors of fixed size.
        Input: Vocabulary size (input_dim=vocab_size)
        Output: Embedding dimension (output_dim=embedding_dim)

    1D Convolutional Layer: Applies a 1D convolution to capture local features in the text.
        Filters: Number of filters (filters=num_filters)
        Kernel Size: Size of the convolutional window (kernel_size)
        Activation: ReLU activation function (activation='relu')

    Global Max Pooling Layer: Reduces each feature map to a single value by taking the maximum value.

    Dense Layer: Fully connected layer with ReLU activation.
        Units: 128

    Dropout Layer: Regularization layer to prevent overfitting.
        Dropout Rate: Configurable dropout rate (dropout_rate)

    Output Layer: Fully connected layer with softmax activation for multi-class classification.
        Units: Number of categories (num_categories)
        Activation: Softmax (activation='softmax')
