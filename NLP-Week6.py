from email import header

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from bs4 import BeautifulSoup
import nltk
from bs4.builder import HTML

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

print("===========STEP 1===================")
#Loading the dataset and exploring the data
imdb = pd.read_csv('/Users/c3703803/OneDrive - Leeds Beckett University/ADV Machine learning Tech for AI/IMDB Dataset.csv')
print(imdb.head())
print("Shape:", imdb.shape)
print(imdb['sentiment'].value_counts())

#mini tasks
#sampling
("sample:",imdb.sample(3))
for i in imdb.sample(3).index:
    print("=======sample=======")
    print(imdb.loc[i, 'review'][:500], "\n---label:", imdb.loc[i, 'sentiment'])


print("============STEP 2- text cleaning==================")
#cleaning the text

def clean_text(text, do_lemmatize=True, remove_digits=True):
    #HTML
    text = BeautifulSoup(text, "html.parser").get_text()
    #removing annotations
    text = re.sub(r'\[[^\]]*\]', ' ', text)
    #keep letters (convert digits optionally)
    if remove_digits:
        text = re.sub(r'[^a-zA-Z\s]', '', text)
    else:
        text = re.sub(r'[^a-zA-Z0-9]\S', '', text)
    #lowercase and tokenize
    tokens = text.lower().split()
    #remove stopwords
    tokens = [t for t in tokens if t not in stop_words]
    #lemmatize
    if do_lemmatize:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ''.join(tokens)
#Apply to a small sample first (fast)
imdb_sample = imdb.sample(2000, random_state= 42).copy()
imdb_sample['clean']= imdb_sample['review'].apply(clean_text)
print(imdb_sample[['review', 'clean']].head(3))

print("==============Step 3 — Full dataset preprocessing================")
imdb['clean'] = imdb['review'].apply(clean_text)

print("==============Step 4 — Feature extraction: CountVectorizer & TF-IDF================")
#Objective: Convert text → numerical features

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
max_feat = 5000
bow_vec = CountVectorizer(max_features=max_feat, ngram_range = (1,1))
tfidf_vec = TfidfVectorizer(max_features=max_feat, ngram_range = (1,1))

X_bow = bow_vec.fit_transform(imdb['clean'])
X_tfidf = tfidf_vec.fit_transform(imdb['clean'])
print("BOW shape:",X_bow.shape)
print("TFIDF shape:",X_tfidf.shape)

#mini task
print("TOP FEATURES:",bow_vec.get_feature_names_out()[:30])

print("===============Step 5 — Train / test split===============")
from sklearn.model_selection import train_test_split

y = imdb ['sentiment'].map({'positive':1, 'negative':0}).values

X_tr_bow, X_te_bow, y_tr, y_te = train_test_split(X_bow, y, test_size=0.2, random_state=42, stratify=y)
X_tr_tfidf, X_te_tfidf, _, _ = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)

print("Train shape:", X_tr_bow.shape, "Test shape:", X_te_bow.shape)

print("=============Step 6 — Baseline models: Naive Bayes & Logistic Regression=================")
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#MultiNomialNB on BOW
mnb = MultinomialNB()
mnb.fit(X_tr_bow, y_tr)
mnb_pred = mnb.predict(X_te_bow)

print("MNB(BOW) acc:", accuracy_score(y_te,mnb_pred))
print(classification_report(y_te,mnb_pred))
# Logistic Regression on BOW
lr =LogisticRegression(max_iter=400)
lr.fit(X_tr_bow, y_tr)
lr_pred = lr.predict(X_te_bow)

print("LR(BOW) acc:", accuracy_score(y_te,lr_pred))
