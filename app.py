import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================
# LOAD MODEL
# =========================

model = tf.keras.models.load_model(
    "sentiment_model.h5"
)

# =========================
# LOAD WORD INDEX
# =========================

word_index = imdb.get_word_index()

# Reverse mapping
reverse_word_index = {
    value: key for key, value in word_index.items()
}

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🎬"
)

# =========================
# TITLE
# =========================

st.title("🎬 Movie Review Sentiment Analysis")

st.write("Enter a movie review below.")

# =========================
# USER INPUT
# =========================

review = st.text_area("Movie Review")

# =========================
# SIMPLE TOKENIZER
# =========================

def encode_review(text):

    words = text.lower().split()

    encoded = []

    for word in words:

        if word in word_index:
            encoded.append(word_index[word])

    return encoded

# =========================
# PREDICTION
# =========================

if st.button("Predict"):

    encoded_review = encode_review(review)

    padded_review = pad_sequences(
        [encoded_review],
        maxlen=100
    )

    prediction = model.predict(
        padded_review
    )

    confidence = prediction[0][0]

    if confidence > 0.5:
        st.success("✅ Positive Review")
    else:
        st.error("❌ Negative Review")

    st.write(f"Confidence: {confidence:.2f}")