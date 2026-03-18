import keras


def load_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize the input data - MNIST data is pixel arrays, so divide by max pixel value 255
    x_train = x_train/255.0
    x_test = x_test/255.0

    # Output is categorical - map from digit target to vector (e.g. 2 -> [0,0,1,0,0,0,0,0,0,0])
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    return x_train, y_train, x_test, y_test


def build_model(cnn=True):

    model = keras.Sequential()

    # Input is 28x28 image, single channel (grayscale)
    model.add(keras.Input(shape=(28, 28, 1)))

    if not cnn:

        ###  Fully connected neural network ###

        # Input is multidimensional, flattened to single dimension
        model.add(keras.layers.Flatten())
        # Add a hidden layer - units is number of neurons/layer width
        model.add(keras.layers.Dense(units=128, activation="relu"))
        model.add(keras.layers.Dense(units=64, activation="relu"))
        model.add(keras.layers.Dropout(0.2))  # Dropout layer to reduce overfitting by randomly setting input units to 0 with a frequency of 0.2 at each step during training time
        model.add(keras.layers.Dense(units=32, activation="relu"))
        model.add(keras.layers.Dropout(0.2))
    else:

        ###  Convolutional neural network  ###

        # Add convolutional layer - filters is depth of layer output and kernel_size the convolution window
        model.add(keras.layers.Conv2D(filters=8, kernel_size=(2, 2), activation="relu", padding="same"))
        # Add pooling layer to downscale (MaxPooling downscales by returning the maximum value in each input window)
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(keras.layers.Conv2D(filters=16, kernel_size=(2, 2), activation="relu", padding="same"))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        # TODO add more layers and/or experiment with different number of filters, different kernel_size or pool_size

        # Flatten internal dimensions before output - additional dense layers could also be included after this line
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=128, activation="relu"))
        model.add(keras.layers.Dropout(0.3))  

    # Final model layer - the same for all model architectures
    # Activation is softmax
    model.add(keras.layers.Dense(units=10, activation="softmax"))

    return model



if __name__ == "__main__":

    # TODO try different values for epochs and learning_rate to improve model performance
    epochs = 10
    learning_rate = 0.1

    x_train, y_train, x_test, y_test = load_mnist()
    
    # Train and evaluate both MLP and CNN models
    for model_type in [False, True]:  # False = MLP, True = CNN
        model_name = "CNN" if model_type else "MLP"
        print(f"\n{'='*60}")
        print(f"Training {model_name} model")
        print(f"{'='*60}\n")
        
        model = build_model(cnn=model_type)

        # Compile model - Stochastic gradient descent is chosen for the optimizer and categorical cross entropy for the
        # loss calculation
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])

        # Show model architecture details and compare parameter counts
        model.summary()

        # Train the model on training data
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=128, verbose=1, validation_split=0.1)

        # Evaluate the model on test data
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
        
        print(f"\n{model_name} Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
