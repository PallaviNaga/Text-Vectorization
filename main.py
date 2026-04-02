import pandas as pd

print("Starting program...")

df = pd.read_csv("IMDB Dataset.csv")

print("Dataset loaded!")

print(df.head())

import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()  # lowercase
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # remove symbols
    words = text.split()
    words = [w for w in words if w not in stop_words]  # remove stopwords
    return " ".join(words)

df['clean_review'] = df['review'].apply(preprocess)

print("\nPreprocessing done!")
print(df[['review','clean_review']].head())

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_review'], df['sentiment'], test_size=0.2
)

# Start time
start = time.time()

# Bag of Words
vectorizer = CountVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Prediction
pred = model.predict(X_test_vec)

# End time
end = time.time()

# Results
print("\n===== BAG OF WORDS =====")
print("Accuracy:", accuracy_score(y_test, pred))
print("Time taken:", end - start)

from sklearn.feature_extraction.text import TfidfVectorizer

start = time.time()

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Prediction
pred = model.predict(X_test_tfidf)

end = time.time()

print("\n===== TF-IDF =====")
print("Accuracy:", accuracy_score(y_test, pred))
print("Time taken:", end - start)

from gensim.models import Word2Vec
import numpy as np

# Convert sentences into list of words
sentences = [text.split() for text in df['clean_review']]

# Train Word2Vec model
w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=2)

# Function to convert sentence → vector
def get_vector(text):
    words = text.split()
    vecs = [w2v.wv[w] for w in words if w in w2v.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(100)

# Convert all reviews to vectors
X_w2v = np.array([get_vector(text) for text in df['clean_review']])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_w2v, df['sentiment'], test_size=0.2
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

# Accuracy
print("\n===== WORD2VEC =====")
print("Accuracy:", accuracy_score(y_test, pred))

# ===== GLOVE =====
print("\nLoading GloVe...")

embeddings = {}

with open("glove.6B.100d.txt", encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.array(values[1:], dtype='float32')
        embeddings[word] = vector

print("GloVe loaded!")

def glove_vector(text):
    words = text.split()
    vecs = [embeddings[w] for w in words if w in embeddings]
    return np.mean(vecs, axis=0) if vecs else np.zeros(100)

X_glove = np.array([glove_vector(text) for text in df['clean_review']])

X_train, X_test, y_train, y_test = train_test_split(
    X_glove, df['sentiment'], test_size=0.2
)

model = LogisticRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\n===== GLOVE =====")
print("Accuracy:", accuracy_score(y_test, pred))

from transformers import pipeline
import time

start = time.time()

classifier = pipeline("sentiment-analysis")

# Test on few samples
samples = df['review'].iloc[:100].tolist()
preds = [classifier(text[:512])[0]['label'] for text in samples]

end = time.time()

print("\n===== BERT =====")
print("Sample Predictions:", preds[:5])
print("Time taken:", end - start)

import matplotlib.pyplot as plt

methods = ['BoW', 'TF-IDF', 'Word2Vec', 'GloVe']
accuracy = [0.8721, 0.8856, 0.8643, 0.8018]

plt.bar(methods, accuracy)
plt.title("Accuracy Comparison")
plt.xlabel("Methods")
plt.ylabel("Accuracy")
plt.show()