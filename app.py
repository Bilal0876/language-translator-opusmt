from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)

cv = CountVectorizer()
data = pd.read_csv(r'PATH to language.csv') #paste your file path here
x = np.array(data['Text'])
y = np.array(data['language'])
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

language_model = MultinomialNB()
language_model.fit(X_train, y_train)

def load_model(language_pair):
    model_name = f'Helsinki-NLP/opus-mt-{language_pair}'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer

translation_models = {}
language_pairs = [
    'en-ar', 'ar-en', 'en-fr', 'fr-en', 'en-ur', 'ur-en',
    'en-hi', 'hi-en', 'en-de', 'de-en', 'en-es', 'es-en',
    'en-nl', 'nl-en', 'en-ru', 'ru-en', 'en-id', 'id-en'
]
for pair in language_pairs:
    translation_models[pair] = load_model(pair)


def fallback_translation(text, from_lang, to_lang):
    if from_lang != 'en' and to_lang != 'en':
        intermediate_text = translate_text(
            text, *translation_models[f'{from_lang}-en']
        )
        return translate_text(
            intermediate_text, *translation_models[f'en-{to_lang}']
        )
    else:
        return translate_text(text, *translation_models[f'{from_lang}-{to_lang}'])

def translate_text(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def detect_language(user_input):
    data_transformed = cv.transform([user_input]).toarray()
    language = language_model.predict(data_transformed)[0]
    language_map = {
        'English': 'en', 'Arabic': 'ar', 'French': 'fr', 'Hindi': 'hi',
        'Urdu': 'ur', 'German': 'de', 'Russian': 'ru', 'Dutch': 'nl',
        'Spanish': 'es', 'Indonesian': 'id'
    }
    return language_map.get(language, language)

@app.route("/", methods=["GET", "POST"])
def home():
    translated_text = ""
    if request.method == "POST":
        user_input = request.form['text_input']
        target_lang = request.form['target_lang']

        source_lang = detect_language(user_input)

        translated_text = fallback_translation(user_input, source_lang, target_lang)

    return render_template("index.html", translated_text=translated_text)

if __name__ == "__main__":
    app.run(debug=True)
