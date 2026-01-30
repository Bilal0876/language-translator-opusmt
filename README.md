# Language Translator (OPUS-MT)

A multilingual language translation project built using Helsinki-NLP OPUS-MT models,
Naive Bayes language detection, and a Flask-based web interface. 
This project demonstrates language detection, translation between multiple languages, 
and a simple web frontend for testing translations.

## Project Structure

│
├── app.py # Main Flask application
├── language.csv # Dataset for language detection
├── templates/ # HTML frontend
│ └── index.html
├──  README.md # Project description

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Bilal0876/language-translator-opusmt.git
   cd language-translator-opusmt

Run the app:
python app.py
Open the browser at:
http://127.0.0.1:5000/

## Notes

- This project uses pretrained models; no training is required.
- First run may take longer as models are downloaded from HuggingFace.
- CSV dataset `language.csv` must be present for language detection.
