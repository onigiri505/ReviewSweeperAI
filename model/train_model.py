import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

print("Loading dataset...")
df = pd.read_csv("model/Reviews.csv")

print(f"Total reviews loaded: {len(df)}")

# --- What is helpfulness? ---
# Each review has HelpfulnessNumerator (people who voted helpful)
# and HelpfulnessDenominator (total people who voted)
# We calculate a ratio. If 70%+ of voters said helpful → label it 1 (helpful)
# If less than 30% said helpful → label it 0 (not helpful)
# Reviews with too few votes are dropped (unreliable)

df = df[df["HelpfulnessDenominator"] >= 5]  # at least 5 votes
df["helpfulness_ratio"] = df["HelpfulnessNumerator"] / df["HelpfulnessDenominator"]
df = df[df["helpfulness_ratio"] != 0.5]  # drop ambiguous middle ground
df["label"] = (df["helpfulness_ratio"] >= 0.7).astype(int)

print(f"Reviews after filtering: {len(df)}")
print(f"Helpful (1): {df['label'].sum()} | Not Helpful (0): {(df['label'] == 0).sum()}")

# --- Combine review title + body as input text ---
df["text"] = df["Summary"].fillna("") + " " + df["Text"].fillna("")

# --- Split into training and testing sets ---
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

print("Vectorizing text with TF-IDF...")
# TF-IDF converts raw text into numbers the model can understand
# max_features=50000 means we track the 50k most important words
vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Training model...")
model = LogisticRegression(max_iter=1000, C=1.0)
model.fit(X_train_vec, y_train)

print("\n--- Model Performance ---")
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred, target_names=["Not Helpful", "Helpful"]))

# --- Save the model and vectorizer ---
joblib.dump(model, "model/helpfulness_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("\nModel saved to model/helpfulness_model.pkl")
print("Vectorizer saved to model/tfidf_vectorizer.pkl")