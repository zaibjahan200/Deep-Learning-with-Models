import streamlit as st
import plotly.express as px
import numpy as np
import pickle
import re
from tensorflow.keras.models import load_model
from nltk.tokenize import TweetTokenizer
from urlextract import URLExtract

# ============================================================================
# 1. LOAD ALL ARTIFACTS (cached for performance)
# ============================================================================
@st.cache_resource
def load_artifacts():
    model = load_model('tweet_v1_better.keras')
    with open('train_vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    with open('config.pkl', 'rb') as f:
        config = pickle.load(f)
    with open('best_thresholds.pkl', 'rb') as f:
        thresholds = pickle.load(f)
    with open('kept_classes.pkl', 'rb') as f:
        class_names = pickle.load(f)
    return model, vocab, config, thresholds, class_names

# ============================================================================
# 2. PREPROCESSING FUNCTIONS (must match training EXACTLY)
# ============================================================================
extractor = URLExtract()
tknzr = TweetTokenizer(preserve_case=False, strip_handles=False, reduce_len=True)

def format_tweet(tweet):
    urls = extractor.find_urls(tweet)
    for url in urls:
        tweet = tweet.replace(url, "{{URL}}")
    tweet = re.sub(r"\b(\s*)(@[\S]+)\b", r'\1{\2@}', tweet)
    return tweet

def tokenizer(text):
    return tknzr.tokenize(text)

# ============================================================================
# 3. PREDICTION FUNCTION
# ============================================================================
def predict_tweet(text, model, vocab, config, thresholds, class_names):
    max_len = config['max_len']
    
    # Preprocess
    cleaned = format_tweet(text)
    tokens = tokenizer(cleaned)
    
    # Encode
    sequence = [vocab.get(word, 1) for word in tokens]  # 1 = <UNK>
    
    # Pad/truncate
    if len(sequence) < max_len:
        sequence = sequence + [0] * (max_len - len(sequence))
    else:
        sequence = sequence[:max_len]
    
    # Predict probabilities
    proba = model.predict(np.array([sequence]), verbose=0)[0]
    
    # Apply thresholds
    threshold_list = [thresholds[i] for i in range(len(class_names))]
    predictions = (proba > threshold_list).astype(int)
    
    # Get class names
    predicted_classes = [class_names[i] for i, val in enumerate(predictions) if val == 1]
    
    return predicted_classes, proba

# ============================================================================
# 4. STREAMLIT APP
# ============================================================================
st.set_page_config(page_title="Tweet Topic Classifier", layout="centered")
st.title("🐦 Tweet Topic Classifier")
st.markdown("Enter a tweet to see its predicted topics.")

# Load model
with st.spinner("Loading model..."):
    model, vocab, config, thresholds, class_names = load_artifacts()

# User input
user_input = st.text_area("Write your tweet here:", height=100)

if st.button("Classify", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        # Predict
        predicted_classes, probabilities = predict_tweet(
            user_input, model, vocab, config, thresholds, class_names
        )
        
        # --- Display Predicted Classes ---
        if predicted_classes:
            st.success(f"**Predicted Topics:** {', '.join(predicted_classes)}")
        else:
            st.info("No topics were confidently predicted (all probabilities below thresholds).")
        
        # --- Create DataFrame for top 3 probabilities ---
        # Sort probabilities descending and get top 3
        sorted_indices = np.argsort(probabilities)[::-1]
        top_3 = [(class_names[i], probabilities[i]) for i in sorted_indices[:3]]
        
        # For the plot, separate into x and y
        labels = [item[0] for item in top_3]
        probs = [item[1] for item in top_3]
        
        # --- Plotly Bar Chart ---
        fig = px.bar(
            x=labels,
            y=probs,
            text=probs,
            labels={'x': 'Topic', 'y': 'Probability'},
            title="Top 3 Topic Probabilities",
            color=probs,
            color_continuous_scale='Blues',
            range_y=[0, 1]
        )
        fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        fig.update_layout(showlegend=False, height=400)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # --- Optional: Show all probabilities in a table ---
        with st.expander("View all probabilities"):
            st.dataframe(
                {
                    "Topic": class_names,
                    "Probability": [f"{prob:.4f}" for prob in probabilities],
                    "Predicted": [(prob > thresholds[i]) for i, prob in enumerate(probabilities)]
                }
            )