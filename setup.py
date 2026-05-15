#!/usr/bin/env python
"""
Setup script for Text Analytics ML Pipeline project
Automates common setup tasks
"""

import os
import sys
import subprocess
from pathlib import Path

def create_models_directory():
    """Create models directory if it doesn't exist"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    print("✓ Models directory ready")

def install_nltk_data():
    """Download required NLTK data"""
    print("Downloading NLTK resources...")
    import nltk
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
    
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')
    
    print("✓ NLTK resources downloaded")

def main():
    print("""
    ╔════════════════════════════════════════════════╗
    ║  Text Analytics ML Pipeline - Setup Script     ║
    ║  CSI324: Text Analytics                        ║
    ╚════════════════════════════════════════════════╝
    """)
    
    # Create models directory
    print("\n[1/2] Setting up project structure...")
    create_models_directory()
    
    # Download NLTK data
    print("\n[2/2] Downloading NLP resources...")
    try:
        install_nltk_data()
    except Exception as e:
        print(f"⚠ Warning: Could not download NLTK data: {e}")
        print("This will be done automatically when the app runs")
    
    print("""
    ╔════════════════════════════════════════════════╗
    ║          Setup Completed Successfully! ✓       ║
    ╚════════════════════════════════════════════════╝
    
    Next steps:
    
    1. Run the Jupyter notebook to train the model:
       jupyter notebook Text_Analytics_Pipeline.ipynb
    
    2. Test the Streamlit app locally:
       streamlit run app.py
    
    3. For deployment instructions, see:
       DEPLOYMENT_GUIDE.md
    
    Questions? Check README.md for more information.
    """)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Setup failed: {e}")
        sys.exit(1)
