import os
import csv
import re
import time
import nltk
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

# Set up GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

# Initialize other tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Different levels of preprocessing to ensure optimal text preparation for each feature
def minimal_preprocess(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'\s+', ' ', text).strip()  # Standardize whitespace
    return text

def intermediate_preprocess(text):
    text = minimal_preprocess(text)
    text = text.lower()
    text = re.sub(r'[^\w\s\.]', '', text)  # Remove punctuation except periods
    return text

def full_preprocess(text):
    text = intermediate_preprocess(text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

def topic_preprocess(text):
    text = minimal_preprocess(text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

def process_scripts(script_dir):
    processed_scripts = {}
    for filename in os.listdir(script_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(script_dir, filename), 'r', encoding='utf-8') as file:
                script_text = file.read()
            processed_scripts[filename] = {
                'raw': script_text,
                'minimal': minimal_preprocess(script_text),
                'intermediate': intermediate_preprocess(script_text),
                'full': full_preprocess(script_text),
                'topic': topic_preprocess(script_text)
            }
    return processed_scripts

def read_box_office_data(csv_path):
    box_office_data = {}
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            script_file = row['Script File']
            box_office_data[script_file] = {
                'budget': float(row['Budget'].replace('$', '').replace(',', '')),
                'worldwide_gross': float(row['Worldwide Gross'].replace('$', '').replace(',', ''))
            }
    return box_office_data

# Baseline features: Script structure, TF-IDF, BoW, NER, and Sentiment

def get_script_structure_features(scripts):
    structure_features = {}
    for script_name, versions in scripts.items():
        raw_text = versions['raw']
        lines = raw_text.split('\n')
        dialogue_count = sum(1 for line in lines if ':' in line and line.split(':')[0].isupper())
        description_count = len(lines) - dialogue_count
        dialogue_ratio = dialogue_count / (description_count + 1e-10)
        scene_count = sum(1 for line in lines if line.strip().upper().startswith(('INT.', 'EXT.')))
        structure_features[script_name] = {
            'dialogue_ratio': dialogue_ratio,
            'scene_count': scene_count
        }
    return structure_features

def get_tfidf_features(scripts, max_features=1000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    texts = [versions['full'] for versions in scripts.values()]
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    return {script: dict(zip([f'tfidf_{name}' for name in feature_names], tfidf_matrix[i].toarray()[0]))
            for i, script in enumerate(scripts.keys())}

def get_bow_features(scripts, max_features=1000):
    vectorizer = CountVectorizer(max_features=max_features)
    texts = [versions['full'] for versions in scripts.values()]
    bow_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    return {script: dict(zip([f'bow_{name}' for name in feature_names], bow_matrix[i].toarray()[0]))
            for i, script in enumerate(scripts.keys())}

def get_ner_features(scripts):
    ner_features = {}
    for script_name, versions in scripts.items():
        text = versions['minimal']
        sentences = sent_tokenize(text)
        entities = {'PERSON': 0, 'LOCATION': 0, 'ORGANIZATION': 0}
        for sentence in sentences:
            words = word_tokenize(sentence)
            tagged = nltk.pos_tag(words)
            chunked = nltk.ne_chunk(tagged)
            for subtree in chunked:
                if type(subtree) == nltk.Tree:
                    if subtree.label() in entities:
                        entities[subtree.label()] += 1
        ner_features[script_name] = entities
    return ner_features

def get_sentiment_features(scripts):
    sia = SentimentIntensityAnalyzer()
    sentiment_features = {}
    for script_name, versions in scripts.items():
        text = versions['minimal']  # Use minimally preprocessed text for sentiment analysis
        sentiment_scores = sia.polarity_scores(text)
        sentiment_features[script_name] = {
            'sentiment_neg': sentiment_scores['neg'],
            'sentiment_neu': sentiment_scores['neu'],
            'sentiment_pos': sentiment_scores['pos'],
            'sentiment_compound': sentiment_scores['compound']
        }
    return sentiment_features

# Word-level features: Word2Vec and N-grams

def get_word_embeddings(scripts, vector_size=100):
    tokenized_scripts = [script['minimal'].split() for script in scripts.values()]
    model = Word2Vec(sentences=tokenized_scripts, vector_size=vector_size, window=5, min_count=1, workers=4)

    script_vectors = {}
    for script, versions in scripts.items():
        words = versions['minimal'].split()
        word_vectors = [model.wv[word] for word in words if word in model.wv]
        if word_vectors:
            script_vector = np.mean(word_vectors, axis=0)
        else:
            script_vector = np.zeros(vector_size)
        script_vectors[script] = script_vector

    return {script: dict(zip([f'w2v_{i}' for i in range(vector_size)], vector))
            for script, vector in script_vectors.items()}

def get_ngram_features(scripts, n=2, max_features=500):
    vectorizer = CountVectorizer(ngram_range=(n, n), max_features=max_features)
    texts = [versions['intermediate'] for versions in scripts.values()]
    ngram_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    return {script: dict(zip([f'ngram_{name}' for name in feature_names], ngram_matrix[i].toarray()[0]))
            for i, script in enumerate(scripts.keys())}

# BERT embeddings

def get_bert_embeddings(scripts, batch_size=32):
    bert_features = {}
    script_items = list(scripts.items())

    for i in range(0, len(script_items), batch_size):
        batch = script_items[i:i+batch_size]
        texts = [item[1]['minimal'] for item in batch]

        inputs = tokenizer(texts, return_tensors='pt', truncation=True, max_length=512, padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        for j, (script, _) in enumerate(batch):
            bert_features[script] = dict(zip([f'bert_{k}' for k in range(768)], embeddings[j]))

    return bert_features

# Topic-based features: LSA and LDA

def get_lsa_features(scripts, n_components=100):
    vectorizer = TfidfVectorizer(max_features=1000)
    texts = [versions['topic'] for versions in scripts.values()]
    tfidf_matrix = vectorizer.fit_transform(texts)
    lsa = TruncatedSVD(n_components=n_components, random_state=42)
    lsa_matrix = lsa.fit_transform(tfidf_matrix)
    return {script: dict(zip([f'lsa_{i}' for i in range(n_components)], lsa_matrix[i]))
            for i, script in enumerate(scripts.keys())}

def get_lda_features(scripts, n_topics=20):
    vectorizer = CountVectorizer(max_features=1000)
    texts = [versions['topic'] for versions in scripts.values()]
    bow_matrix = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_matrix = lda.fit_transform(bow_matrix)
    return {script: dict(zip([f'lda_topic_{i}' for i in range(n_topics)], lda_matrix[i]))
            for i, script in enumerate(scripts.keys())}

def create_feature_sets(processed_scripts, box_office_data):
    feature_sets = {
        'baseline': {},
        'word_level': {},
        'bert': {},
        'topic': {},
        'all': {}
    }

    print("Extracting baseline features...")
    structure_features = get_script_structure_features(processed_scripts)
    tfidf_features = get_tfidf_features(processed_scripts)
    bow_features = get_bow_features(processed_scripts)
    ner_features = get_ner_features(processed_scripts)
    sentiment_features = get_sentiment_features(processed_scripts)

    print("Extracting word-level features...")
    word_embeddings = get_word_embeddings(processed_scripts)
    ngram_features = get_ngram_features(processed_scripts)

    print("Extracting BERT embeddings...")
    bert_features = get_bert_embeddings(processed_scripts)

    print("Extracting topic-based features...")
    lsa_features = get_lsa_features(processed_scripts)
    lda_features = get_lda_features(processed_scripts)

    for script_file in processed_scripts.keys():
        if script_file in box_office_data:
            base_dict = {
                'script_file': script_file,
                'budget': box_office_data[script_file]['budget'],
                'worldwide_gross': box_office_data[script_file]['worldwide_gross']
            }
            base_dict.update(structure_features[script_file])
            base_dict.update(tfidf_features[script_file])
            base_dict.update(bow_features[script_file])
            base_dict.update(ner_features[script_file])
            base_dict.update(sentiment_features[script_file])

            feature_sets['baseline'][script_file] = base_dict.copy()

            word_level_dict = base_dict.copy()
            word_level_dict.update(word_embeddings[script_file])
            word_level_dict.update(ngram_features[script_file])
            feature_sets['word_level'][script_file] = word_level_dict

            bert_dict = word_level_dict.copy()
            bert_dict.update(bert_features[script_file])
            feature_sets['bert'][script_file] = bert_dict

            topic_dict = bert_dict.copy()
            topic_dict.update(lsa_features[script_file])
            topic_dict.update(lda_features[script_file])
            feature_sets['topic'][script_file] = topic_dict

            feature_sets['all'][script_file] = topic_dict

    return feature_sets

def write_arff(features, output_file):
    with open(output_file, 'w') as f:
        f.write('@RELATION movie_success\n\n')
        for key in features[0].keys():
            if key != 'script_file':
                safe_key = key.replace(' ', '_').replace(',', '_')
                f.write(f'@ATTRIBUTE {safe_key} NUMERIC\n')
        f.write('\n@DATA\n')
        for feature in features:
            data_line = ','.join(str(feature[key]) for key in feature.keys() if key != 'script_file')
            f.write(data_line + '\n')

def write_arff_files(feature_sets, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for set_name, features in feature_sets.items():
        output_file = os.path.join(output_dir, f'movie_success_{set_name}.arff')
        write_arff(list(features.values()), output_file)
        print(f"ARFF file for {set_name} features created at: {output_file}")

# Main execution
script_dir = '/content/drive/MyDrive/Predicting Box Office Success/matched_scripts'
box_office_csv = '/content/drive/MyDrive/Predicting Box Office Success/matched_scripts.csv'
output_dir = '/content/drive/MyDrive/Predicting Box Office Success/incremental_features'

start_time = time.time()

print("Processing scripts...")
processed_scripts = process_scripts(script_dir)
print(f"Processed {len(processed_scripts)} scripts")

print("Reading box office data...")
box_office_data = read_box_office_data(box_office_csv)

print("Creating feature sets...")
feature_sets = create_feature_sets(processed_scripts, box_office_data)

print("Writing ARFF files...")
write_arff_files(feature_sets, output_dir)

end_time = time.time()
print(f"All ARFF files have been created in: {output_dir}")
print(f"Total execution time: {end_time - start_time:.2f} seconds")