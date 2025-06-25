import time
import tensorflow as tf
from tensorflow.keras import layers, models

# === CONFIGURATION ===
EPOCHS = 5
BATCH_SIZE = 128

# === MODEL DEFINITION ===
def get_model():
    model = models.Sequential([
        layers.Reshape((28,28,1), input_shape=(28,28)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# === DATA LOADING ===
def load_data():
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    return x_train, y_train

# === GPU TRAINING ONLY ===
def run_gpu_training(x, y):
    # Check that at least one GPU is available:
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError("No GPU found. Check your CUDA/cuDNN setup.")
    print("Found GPUs:", gpus)
    # (No calls to set_visible_devices here—defaults will include your GPU)
    
    model = get_model()
    print("Starting GPU training for", EPOCHS, "epochs…")
    start = time.time()
    model.fit(
        x, y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1
    )
    total_time = time.time() - start
    print(f"\nTotal training time on GPU: {total_time:.2f} seconds")
    return total_time

if __name__ == "__main__":
    print("TensorFlow version:", tf.__version__)
    x_train, y_train = load_data()
    run_gpu_training(x_train, y_train)
