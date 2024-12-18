import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

class SpamClassifier:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()

    def train(self, data: pd.DataFrame):
        X = self.vectorizer.fit_transform(data['text'])
        y = data['labels']
        self.model.fit(X, y)

    def predict(self, text: str):
        X = self.vectorizer.transform([text])
        return self.model.predict(X)[0]

    def save_model(self, model_path: str, vectorizer_path: str):
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)

    def load_model(self, model_path: str, vectorizer_path: str):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
