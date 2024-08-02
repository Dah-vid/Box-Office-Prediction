import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

print("spaCy model loaded successfully")

# Set up file paths
script_folder = r"C:\Users\david\Documents\MSC Coursework\Dissertation\Code\matched_scripts"
csv_path = r"C:\Users\david\Documents\MSC Coursework\Dissertation\Code\matched_scripts.csv"


# Preprocessing functions
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()


def tokenize_and_remove_stopwords(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]


def stem_words(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]


def preprocess_script(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        cleaned_text = clean_text(content)
        tokens = tokenize_and_remove_stopwords(cleaned_text)
        stemmed_tokens = stem_words(tokens)
        return ' '.join(stemmed_tokens)
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return ""


# Feature extraction functions
def create_bow(texts, max_features=5000):
    vectorizer = CountVectorizer(max_features=max_features)
    bow_matrix = vectorizer.fit_transform(texts)
    return pd.DataFrame(bow_matrix.toarray(), columns=vectorizer.get_feature_names_out())


def create_tfidf(texts, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(texts)
    return pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())


def create_topics(tfidf_matrix, n_topics=10):
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    topics = lda.fit_transform(tfidf_matrix)
    return pd.DataFrame(topics, columns=[f'Topic_{i + 1}' for i in range(n_topics)])


def get_sentiment(text):
    return TextBlob(text).sentiment.polarity


def extract_entities(text):
    doc = nlp(text[:1000000])  # Limit text length to avoid potential issues
    entities = {ent.label_: ent.text for ent in doc.ents}
    return str(entities)  # Convert to string for easy storage in DataFrame


def extract_structural_features(script):
    lines = script.split('\n')
    total_lines = len(lines)
    dialogue_lines = sum(1 for line in lines if line.strip().startswith('"'))

    return {
        'script_length': len(script),
        'total_lines': total_lines,
        'dialogue_lines': dialogue_lines,
        'dialogue_ratio': dialogue_lines / total_lines if total_lines > 0 else 0
    }


# Main processing
print("Reading CSV file...")
df = pd.read_csv(csv_path)

print("Preprocessing scripts...")
preprocessed_scripts = []
for index, row in df.iterrows():
    script_file = row['Script File']
    script_path = os.path.join(script_folder, script_file)
    if os.path.exists(script_path):
        preprocessed_text = preprocess_script(script_path)
        preprocessed_scripts.append(preprocessed_text)
    else:
        print(f"Script file not found: {script_file}")
        preprocessed_scripts.append("")

df['preprocessed_text'] = preprocessed_scripts

print("Creating Bag of Words features...")
bow_df = create_bow(df['preprocessed_text'])

print("Creating TF-IDF features...")
tfidf_df = create_tfidf(df['preprocessed_text'])

print("Performing topic modeling...")
topic_df = create_topics(tfidf_df)

print("Calculating sentiment scores...")
df['sentiment_score'] = df['preprocessed_text'].apply(get_sentiment)

print("Extracting named entities...")
df['named_entities'] = df['preprocessed_text'].apply(extract_entities)

print("Extracting structural features...")
structural_features = df['preprocessed_text'].apply(extract_structural_features)
structural_df = pd.DataFrame(structural_features.tolist())

print("Combining all features...")
final_df = pd.concat([
    df,
    bow_df.add_prefix('BoW_'),
    tfidf_df.add_prefix('TFIDF_'),
    topic_df,
    structural_df
], axis=1)

print("Saving processed data...")
final_df.to_csv('processed_movie_data_with_features.csv', index=False)

print("Processing completed. Data saved to 'processed_movie_data_with_features.csv'")