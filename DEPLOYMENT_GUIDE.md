# Deployment Guide - GitHub & Streamlit Cloud

## Prerequisites

- Python 3.9 or higher
- Git installed on your system
- GitHub account
- Streamlit Cloud account (free)
- IMDB Dataset CSV file

## Step 1: Set Up Local Environment

### 1.1 Create a new folder (if not already done)

```bash
mkdir ca2-text-analytics
cd ca2-text-analytics
```

### 1.2 Initialize a Python virtual environment

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

### 1.3 Install required packages

```bash
pip install -r requirements.txt
```

### 1.4 Run the Jupyter notebook

```bash
jupyter notebook Text_Analytics_Pipeline.ipynb
```

- Execute all cells to train the model
- This creates the `models/` directory with `.pkl` files

### 1.5 Test the Streamlit app locally

```bash
streamlit run app.py
```

- Opens at `http://localhost:8501`
- Test sentiment analysis functionality

## Step 2: Set Up GitHub Repository

### 2.1 Initialize Git repository

```bash
git init
```

### 2.2 Create a GitHub repository

- Go to [GitHub](https://github.com)
- Click "New repository"
- Name: `ca2-text-analytics`
- Description: "IMDB Sentiment Analysis - Text Analytics Pipeline"
- Choose: Public (for easier deployment)
- Click "Create repository"

### 2.3 Add remote and push code

```bash
git add .
git commit -m "Initial commit: Text analytics ML pipeline with Streamlit deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ca2-text-analytics.git
git push -u origin main
```

### 2.4 Verify on GitHub

- Go to your repository URL
- Check if all files are uploaded:
  - Text_Analytics_Pipeline.ipynb
  - app.py
  - requirements.txt
  - README.md
  - models/ directory
  - .gitignore
  - .streamlit/

## Step 3: Deploy on Streamlit Cloud

### 3.1 Create Streamlit Cloud account

- Go to [Streamlit Cloud](https://streamlit.io/cloud)
- Sign in with GitHub account
- Authorize Streamlit

### 3.2 Deploy the app

1. Click "New app"
2. Select your GitHub account
3. Choose repository: `ca2-text-analytics`
4. Select branch: `main`
5. Set main file path to: `app.py`
6. Click "Deploy"

### 3.3 Wait for deployment

- Streamlit builds and deploys automatically
- Takes 2-3 minutes
- Shows deployment status in dashboard

### 3.4 Access your app

- Copy the public URL (e.g., `https://your-username-ca2-text-analytics.streamlit.app`)
- Share this link to use the app

## Step 4: File Structure for Deployment

Ensure your GitHub repository has this structure:

```
ca2-text-analytics/
│
├── Text_Analytics_Pipeline.ipynb    # Jupyter notebook with pipeline
├── app.py                            # Streamlit app
├── requirements.txt                  # Dependencies
├── README.md                         # Project documentation
├── DEPLOYMENT_GUIDE.md              # This file
├── .gitignore                       # Git ignore file
│
├── .streamlit/
│   └── config.toml                  # Streamlit configuration
│
├── models/
│   ├── naive_bayes_model.pkl        # Trained model
│   └── tfidf_vectorizer.pkl         # TF-IDF vectorizer
│
└── IMDB Dataset.csv                 # Dataset file
```

## Step 5: Continuous Deployment

### Every time you update the code:

```bash
# Make changes to your files
# ...

# Stage changes
git add .

# Commit changes
git commit -m "Update: Description of changes"

# Push to GitHub
git push origin main
```

Streamlit Cloud automatically redeploys when you push to the `main` branch.

## Troubleshooting

### Issue: Models not found error

**Solution**:

- Run the Jupyter notebook completely before pushing to GitHub
- Ensure `models/` directory exists with `.pkl` files

### Issue: NLTK data errors

**Solution**:

- Streamlit app auto-downloads NLTK data on first run
- If persistent, add this line in `app.py` before other imports:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Issue: Deployment fails - ModuleNotFoundError

**Solution**:

- Check `requirements.txt` includes all imports
- Update with: `pip freeze > requirements.txt`

### Issue: App loading slowly

**Solution**:

- Models are cached using `@st.cache_resource`
- First load takes 10-15 seconds
- Subsequent loads are instant

### Issue: Memory limits exceeded

**Solution**:

- Streamlit Cloud has 1GB memory limit
- If using large dataset, reduce it:

```python
df = df.head(10000)  # Use first 10k rows
```

## Monitoring & Logs

### View deployment logs:

1. Go to [Streamlit Cloud dashboard](https://share.streamlit.io)
2. Click your app
3. View "Logs" tab

### Common log messages:

```
"Collecting requirements: Downloading packages..."    # Installing dependencies
"Building wheels for collected packages"              # Building packages
"Preparing metadata"                                  # Preparing deployment
"App started on..."                                   # Successfully deployed
```

## Next Steps

After successful deployment:

1. **Share the app**: Post URL on GitHub, in reports, or with classmates
2. **Update documentation**: Keep README.md current
3. **Version control**: Use meaningful commit messages
4. **Test thoroughly**: Try various reviews to validate predictions
5. **Monitor usage**: Check logs for errors

## Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [GitHub Guides](https://guides.github.com)
- [Git Documentation](https://git-scm.com/doc)
- [Streamlit Cloud FAQ](https://docs.streamlit.io/streamlit-cloud/get-started)

## Support

For issues:

1. Check Streamlit Cloud logs
2. Review GitHub repository status
3. Verify all files are present
4. Ensure requirements.txt is complete
5. Test locally before pushing to GitHub

---

**Happy Deploying!** 🚀
