import streamlit as st
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Initialize preprocessing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load saved models
@st.cache_resource
def load_models():
    """Load the trained model and vectorizer"""
    model_path = 'models/naive_bayes_model.pkl'
    vectorizer_path = 'models/tfidf_vectorizer.pkl'
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    return model, vectorizer

# Preprocessing function
def preprocess_text(text):
    """Complete preprocessing pipeline"""
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

# Streamlit app
st.set_page_config(page_title="IMDB Sentiment Classifier", layout="wide")

st.title("🎬 IMDB Sentiment Analysis")
st.markdown("### Text Classification using TF-IDF and Naive Bayes")

# Load models
try:
    model, vectorizer = load_models()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Prediction", "About", "Model Info"])
    
    with tab1:
        st.subheader("Enter Movie Review")
        
        # Text input
        user_input = st.text_area(
            "Enter your movie review:",
            placeholder="Type a movie review here...",
            height=150
        )
        
        if st.button("Analyze Sentiment", key="predict"):
            if user_input.strip():
                # Preprocess the input
                processed_text = preprocess_text(user_input)
                
                # Vectorize the text
                text_tfidf = vectorizer.transform([processed_text])
                
                # Make prediction
                prediction = model.predict(text_tfidf)[0]
                probability = model.predict_proba(text_tfidf)[0]
                
                # Display results
                st.divider()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.success("### 😊 Positive Sentiment")
                        sentiment_label = "POSITIVE"
                        confidence = probability[1] * 100
                    else:
                        st.error("### 😞 Negative Sentiment")
                        sentiment_label = "NEGATIVE"
                        confidence = probability[0] * 100
                
                with col2:
                    st.metric("Confidence", f"{confidence:.2f}%")
                
                # Display probabilities
                st.subheader("Sentiment Probabilities")
                prob_col1, prob_col2 = st.columns(2)
                
                with prob_col1:
                    st.metric("Negative Probability", f"{probability[0]*100:.2f}%")
                
                with prob_col2:
                    st.metric("Positive Probability", f"{probability[1]*100:.2f}%")
                
                # Show preprocessed text
                with st.expander("View Preprocessed Text"):
                    st.write(processed_text)
            else:
                st.warning("Please enter a review to analyze.")
    
    with tab2:
        st.subheader("About This Application")
        st.markdown("""
        This application provides a compact, practical implementation of a text analytics
        pipeline for sentiment classification on IMDB movie reviews. It includes:
        - Text preprocessing (tokenization, stopword removal, lemmatization)
        - Feature extraction (TF-IDF)
        - Classification (Multinomial Naive Bayes)
        - A Streamlit interface for interactive predictions
        """)

        st.markdown("### Academic Submission")
        st.markdown("""
        **ACADEMIC TASK-2**  
        CSI-323 (Text Analytics)  
        Department of Computer Science and Engineering  
        
        **Submitted by:**  
        Nitesh Dwivedi  
        Registration Number: 12317648  
        Roll No.: 29  
        Section: 223BD  
        
        **Submitted to:**  
        Mr. Dharmendrer  
        Lovely Professional University
        """)

        # GitHub link
        st.markdown("**GitHub Repository:** [https://github.com/niteshdwivedi](https://github.com/niteshdwivedi)")

        # Provide notebook download/view option
        nb_path = 'Text_Analytics_Pipeline.ipynb'
        if os.path.exists(nb_path):
            with open(nb_path, 'rb') as f:
                nb_bytes = f.read()
            st.download_button("Download Notebook (View Code)", nb_bytes, file_name='Text_Analytics_Pipeline.ipynb', mime='application/octet-stream')
            st.markdown("You can download the notebook to view the full pipeline and code used to train the model.")
        else:
            st.info("Notebook file not found locally. You can view the notebook on the GitHub repository linked above.")
    
    with tab3:
        st.subheader("Model Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Algorithm", "Multinomial Naive Bayes")
        
        with col2:
            st.metric("Features", "5000 TF-IDF")
        
        with col3:
            st.metric("N-grams", "1-2 grams")
        
        st.markdown("""
        **Training Configuration:**
        - Max features: 5000
        - Min document frequency: 2
        - Max document frequency: 0.8
        - N-gram range: (1, 2)
        - Train-test split: 80-20
        - Random state: 42
        """)

except FileNotFoundError:
    st.error("⚠️ Model files not found! Please run the Jupyter notebook first to train and save the models.")
    st.info("Steps to fix:\n1. Run the Jupyter notebook (Text_Analytics_Pipeline.ipynb)\n2. Ensure models are saved to 'models/' directory")
