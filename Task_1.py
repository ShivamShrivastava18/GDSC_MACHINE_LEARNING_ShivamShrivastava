import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),   
    layers.Dense(128, activation='relu'),    
    layers.Dropout(0.2),                     
    layers.Dense(10, activation='softmax')   
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'\nTest accuracy: {test_acc}')

# Make predictions
predictions = model.predict(x_test[:5])
predicted_labels = tf.argmax(predictions, axis=1)

# Display the predictions and true labels
print(f'Predicted labels: {predicted_labels.numpy()}')
print(f'True labels:      {y_test[:5]}')

for i in range(5):
    print(predictions[i], " => ", y_test[i])
    plt.imshow(x_test[i], cmap="Greys")
    plt.show()
