import tensorflow as tf
import matplotlib.pyplot as plt

# print Tensorflow version
print(tf.__version__)

# 1. Load & normalize the MNIST data
(train_images, train_labels), (eval_images, eval_labels) = tf.keras.datasets.mnist.load_data()
train_images, eval_images = train_images.astype('float32') / 255.0, eval_images.astype('float32') / 255.0

# 2. Build the network
digit_classifier = tf.keras.Sequential([
    # small CNN network for better generalization
    tf.keras.layers.Reshape((28,28,1), input_shape=(28,28)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])

# 3. Compile the model
digit_classifier.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Train the model
history = digit_classifier.fit(
    train_images,
    train_labels,
    epochs=7,
    validation_split=0.1,
    verbose=2
)

# 5. Evaluate on test set
test_loss, test_acc = digit_classifier.evaluate(eval_images, eval_labels, verbose=0)
print(f"Evaluation Accuracy: {test_acc:.3f}")

# 6. Plot & save loss curves
plt.figure(figsize=(6,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training vs. Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('loss_curves.png')

# save trained model to file
digit_classifier.save('mnist_digit_classifier.h5')  