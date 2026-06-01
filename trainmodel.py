import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================
# LOAD DATASET
# =========================

print("Loading Dataset...")

max_words = 5000
max_len = 100

(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=max_words
)

print("Dataset Loaded")

# =========================
# PAD SEQUENCES
# =========================

X_train = pad_sequences(
    X_train,
    maxlen=max_len
)

X_test = pad_sequences(
    X_test,
    maxlen=max_len
)

print("Padding Complete")

# =========================
# BUILD LSTM MODEL
# =========================

model = tf.keras.Sequential([

    tf.keras.layers.Embedding(
        input_dim=max_words,
        output_dim=64
    ),

    tf.keras.layers.LSTM(32),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(
        1,
        activation='sigmoid'
    )
])

# =========================
# COMPILE MODEL
# =========================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =========================
# MODEL SUMMARY
# =========================

model.summary()

# =========================
# TRAIN MODEL
# =========================

print("Starting Training...")

history = model.fit(
    X_train[:5000],
    y_train[:5000],
    epochs=2,
    batch_size=32,
    validation_split=0.2,
    verbose=2
)

print("Training Finished")

# =========================
# EVALUATE MODEL
# =========================

loss, accuracy = model.evaluate(
    X_test[:1000],
    y_test[:1000],
    verbose=0
)

print(f"\nTest Accuracy: {accuracy:.2f}")

# =========================
# SAVE MODEL
# =========================

model.save("sentiment_model.h5")

print("\nModel Saved Successfully")