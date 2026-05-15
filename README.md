# IMDB Sentiment Analysis - Text Analytics Pipeline

## Overview

This project implements a complete **ML pipeline for text analytics** on IMDB movie reviews, demonstrating practical application of concepts from **CSI324: Text Analytics** course.

## Course Mapping

### Covered Course Outcomes

- **CO1**: Explains fundamentals of text analytics, NLP, and linguistic preprocessing
- **CO2**: Demonstrates tokenization, stemming, lemmatization, and preprocessing techniques
- **CO3**: Applies feature extraction methods (TF-IDF)
- **CO4**: Analyzes models for text classification and sentiment analysis
- **CO5**: Utilizes Python tools (NLTK, scikit-learn) for text analytics
- **CO6**: Compares solutions and explores real-world applications

## Project Structure

```
ca2/
├── Text_Analytics_Pipeline.ipynb      # Main ML pipeline notebook
├── app.py                              # Streamlit deployment app
├── requirements.txt                    # Project dependencies
├── README.md                           # This file
├── models/
│   ├── naive_bayes_model.pkl          # Trained model
│   └── tfidf_vectorizer.pkl           # TF-IDF vectorizer
└── IMDB Dataset.csv                   # Dataset
```

## Pipeline Architecture

### 1. **Text Preprocessing (Unit I)**

- Tokenization using NLTK
- Stopword removal
- Lemmatization using WordNetLemmatizer
- Text normalization (lowercase, special character handling)

### 2. **Feature Engineering (Unit III)**

- TF-IDF Vectorization with sklearn
- Feature dimension: 5000
- N-gram range: 1-2 grams
- Dimensionality optimization

### 3. **Text Classification (Unit IV)**

- Algorithm: Multinomial Naive Bayes
- Train-test split: 80-20
- Evaluation metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix

### 4. **Deployment**

- Streamlit web interface
- Interactive text input and prediction
- Real-time sentiment analysis

## Installation & Setup

### Local Development

1. **Clone the repository**

   ```bash
   git clone https://github.com/YOUR_USERNAME/ca2-text-analytics.git
   cd ca2-text-analytics
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Jupyter notebook** (To train models)

   ```bash
   jupyter notebook Text_Analytics_Pipeline.ipynb
   ```

   - Execute all cells to train the model and generate `models/` directory

5. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

   - Opens at `http://localhost:8501`

## Usage

### Training the Model

1. Open `Text_Analytics_Pipeline.ipynb`
2. Run cells sequentially to:
   - Load and explore IMDB dataset
   - Preprocess text data
   - Extract TF-IDF features
   - Train Naive Bayes classifier
   - Evaluate model performance
   - Save trained models

### Using the Streamlit App

1. Launch: `streamlit run app.py`
2. Enter a movie review in the text area
3. Click "Analyze Sentiment"
4. View prediction with confidence score

## Model Performance

The trained model achieves:

- **Accuracy**: ~88-90% on test set
- **Precision**: ~88-90%
- **Recall**: ~85-90%
- **F1-Score**: ~87-90%

Metrics vary based on the dataset size used for training.

## Deployment on Streamlit Cloud

### Prerequisites

- GitHub account
- Repository with code pushed to GitHub

### Steps

1. **Push to GitHub**

   ```bash
   git init
   git add .
   git commit -m "Initial commit: Text analytics pipeline"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/ca2-text-analytics.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [streamlit.io/cloud](https://streamlit.io/cloud)
   - Click "New app"
   - Select your GitHub repository
   - Set main file to `app.py`
   - Deploy

3. **Access the app**
   - Get public URL from Streamlit Cloud
   - Share with others

## Technologies Used

- **Python 3.9+**
- **NLTK**: Natural Language Toolkit (tokenization, lemmatization)
- **scikit-learn**: TF-IDF vectorization, Naive Bayes classifier
- **Streamlit**: Web deployment
- **joblib**: Model serialization
- **pandas, numpy**: Data processing
- **matplotlib, seaborn**: Visualization

## Practical Exercises Covered

✅ Text preprocessing using Python (tokenization, stemming, lemmatization)
✅ Build TF-IDF vectors and analyze document similarity
✅ Build a text classification model using Naive Bayes
✅ Perform sentiment analysis on reviews
✅ Evaluate models using precision, recall, F1-score, confusion matrix
✅ Deploy ML model using Streamlit

## References

1. Text Analytics with Python - DIPANJAN SARKAR, APRESS
2. Natural Language Processing with Transformers - LEWIS TUNSTALL, LEANDRO VON WERRA, THOMAS WOLF, O'REILLY
3. Speech and Language Processing - DANIEL JURAFSKY AND JAMES H. MARTIN
4. Natural Language Processing with Python - STEVEN BIRD, EWAN KLEIN, EDWARD LOPER

## Course Information

- **Course Code**: CSI324
- **Course Title**: Text Analytics
- **Credits**: 3
- **Theory**: 2 hrs/week
- **Practical**: 2 hrs/week

## Author

[Your Name/Student ID]

## License

This project is for educational purposes as part of CSI324 course assessment.

## Troubleshooting

### Model Files Not Found

- Ensure `Text_Analytics_Pipeline.ipynb` has been run completely
- Check if `models/` directory exists with `.pkl` files

### NLTK Data Missing

- The app automatically downloads required NLTK data on first run
- If issues persist: `python -m nltk.downloader punkt stopwords wordnet`

### Memory Issues

- Reduce dataset size if running on limited RAM
- Modify `test_size` in train_test_split
- Lower `max_features` in TfidfVectorizer

## Future Enhancements

- [ ] Support for multiple languages
- [ ] Advanced models (BERT, DistilBERT)
- [ ] Real-time model retraining
- [ ] Multi-class sentiment analysis
- [ ] Aspect-based sentiment analysis
- [ ] Word embeddings (Word2Vec, GloVe)
- [ ] Topic modeling (LDA)
