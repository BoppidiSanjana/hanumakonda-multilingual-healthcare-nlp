import os, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from ml.predictor import MODEL_PATH

DATA = os.path.join(os.path.dirname(__file__), "data", "synthetic_demo_dataset.csv")
df = pd.read_csv(DATA).dropna(subset=["text", "category"])
# This tiny synthetic teaching dataset is not suitable for reliable evaluation.
# Disable stratification because the starter CSV has categories with only one example.
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["category"], test_size=0.25, random_state=42
)
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), analyzer="word")),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test), zero_division=0))
with open(MODEL_PATH, "wb") as f: pickle.dump(model, f)
print("Saved DEMO model:", MODEL_PATH)
print("WARNING: metrics are based on synthetic examples, not validated healthcare data.")
