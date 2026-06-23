import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

def main():
    print("Loading MNIST dataset...")
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

    print("Preprocessing data...")
    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Reshape data to include the channel dimension (28, 28, 1)
    train_images = np.expand_dims(train_images, axis=-1)
    test_images = np.expand_dims(test_images, axis=-1)

    print("Building lightweight CNN model...")
    model = models.Sequential()

    # First Convolutional Block
    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))

    # Second Convolutional Block
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Dense Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3)) # Prevent overfitting
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("Training model...")
    epochs = 5
    model.fit(train_images, train_labels, epochs=epochs, 
              validation_data=(test_images, test_labels))

    print("Evaluating model...")
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print(f'Test accuracy: {test_acc:.4f}')

    print("Saving model...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'mnist_cnn.keras')
    model.save(model_path)
    print(f"Model saved in '{model_path}'.")

if __name__ == "__main__":
    main()
