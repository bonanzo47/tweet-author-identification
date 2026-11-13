import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Download NLTK Stopwords
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

print("Loading dataset...")
try:
    df = pd.read_csv("tweet_with_authors.csv", encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv("tweet_with_authors.csv", encoding='latin-1')

text_col = [c for c in df.columns if any(k in c.lower() for k in ['text', 'tweet', 'content', 'message'])][0]
author_col = [c for c in df.columns if any(k in c.lower() for k in ['author', 'user', 'label', 'class', 'target'])][0]

df = df[[text_col, author_col]].dropna()
df.columns = ['tweet', 'author']

# Preprocessing function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

print("Preprocessing text...")
df['clean_tweet'] = df['tweet'].apply(clean_text)
df = df[df['clean_tweet'].str.strip() != ""]

# Feature Extraction & Training
print("Training ML models on full dataset...")
vectorizer = TfidfVectorizer(max_features=2500)
X = vectorizer.fit_transform(df['clean_tweet'])
y = df['author']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm = SVC(kernel='linear', probability=True).fit(X_train, y_train)
nb = MultinomialNB().fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_train, y_train)

# Print accuracies for your own reference
print(f"SVM Accuracy: {accuracy_score(y_test, svm.predict(X_test)):.2%}")
print(f"Naïve Bayes Accuracy: {accuracy_score(y_test, nb.predict(X_test)):.2%}")
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf.predict(X_test)):.2%}")

# Save artifacts to disk
print("Saving models and vectorizer to disk...")
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(svm, 'model_svm.pkl')
joblib.dump(nb, 'model_nb.pkl')
joblib.dump(rf, 'model_rf.pkl')
joblib.dump(df, 'clean_df.pkl')
print("Training complete! Artifacts saved successfully.")
