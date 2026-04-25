import joblib

model = joblib.load("model/helpfulness_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

def score_review(title, text):
    combined = (title + " " + text).strip()
    word_count = len(combined.split())
    unique_words = len(set(combined.lower().split()))
    total_words = max(word_count, 1)
    vocab_ratio = unique_words / total_words

    # Run model FIRST
    vectorized = vectorizer.transform([combined])
    prediction = int(model.predict(vectorized)[0])
    confidence = float(model.predict_proba(vectorized)[0].max())

    # THEN apply overrides
    words = combined.lower().split()
    word_freq = {}
    for w in words:
        word_freq[w] = word_freq.get(w, 0) + 1
    max_repetition = max(word_freq.values())

    if max_repetition >= 3:
        prediction = 0
        confidence = 0.92

    
    # Ignore common English words in vocab check
    stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", 
                "for", "of", "with", "this", "it", "is", "was", "i", "my", "me"}

    meaningful_words = [w for w in combined.lower().split() if w not in stopwords]
    unique_meaningful = len(set(meaningful_words))
    total_meaningful = max(len(meaningful_words), 1)
    vocab_ratio = unique_meaningful / total_meaningful

    # Use meaningful_words for repetition check too
    word_freq = {}
    for w in meaningful_words:
        word_freq[w] = word_freq.get(w, 0) + 1
    max_repetition = max(word_freq.values()) if word_freq else 0

    if word_count < 8:
        prediction = 0
        confidence = 0.95

    return bool(prediction), round(confidence, 3)

test_reviews = [
    ("Vague positive", "Great product!! Love it 5 stars!!!"),
    ("One word", "Good."),
    ("Spam", "BEST PRODUCT EVER BUY NOW BUY NOW"),
    ("Bad Quality Product", "I bought this 3 months ago and the zipper broke after 2 weeks of light use. The material feels cheap compared to the photos. Returned it."),
    ("Detailed positive", "Been using this for 6 months daily. Sound quality is great for the price, bass is decent but not deep. Battery lasts about 5 hours. Worth it under Rs 500."),
    ("Comparison review", "Compared to the boAt version this has better mic quality but weaker bass. Build feels slightly cheaper but the fit is more comfortable for long use."),
    ("Short but specific", "Broke after 2 weeks. Avoid."),
    ("Long but vague", "I really really really love this product so much it is amazing and great and wonderful and I would recommend it to everyone I know because it is just so good"),
]

print(f"{'Review Type':<20} {'Prediction':<15} {'Confidence':<12} {'Word Count'}")
print("-" * 65)
for title, text in test_reviews:
    helpful, confidence = score_review(title, text)
    label = "Helpful ✅" if helpful else "Not Helpful ❌"
    word_count = len((title + " " + text).split())
    print(f"{title:<20} {label:<15} {confidence:.1%}        {word_count} words")

# Quick accuracy check on a sample
import pandas as pd
from sklearn.metrics import accuracy_score

print("\n--- Quick Accuracy Check ---")
df = pd.read_csv("model/Reviews.csv")
df["helpfulness_ratio"] = df["HelpfulnessNumerator"] / df["HelpfulnessDenominator"]
df = df[df["HelpfulnessDenominator"] >= 10]
df_helpful = df[df["helpfulness_ratio"] >= 0.80].copy()
df_helpful["label"] = 1
df_not = df[df["helpfulness_ratio"] <= 0.20].copy()
df_not["label"] = 0
df = pd.concat([df_helpful, df_not]).sample(2000, random_state=42)
df["text"] = df["Summary"].fillna("") + " " + df["Text"].fillna("")

X = vectorizer.transform(df["text"])
preds = model.predict(X)
print(f"Accuracy on 2000 sample: {accuracy_score(df['label'], preds):.1%}")

# Debug detailed critical
"""
title = "Detailed critical"
text = "I bought this 3 months ago and the zipper broke after 2 weeks of light use. The material feels cheap compared to the photos. Returned it."
combined = (title + " " + text).strip()
words = combined.lower().split()
stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
             "for", "of", "with", "this", "it", "is", "was", "i", "my", "me"}
meaningful_words = [w for w in words if w not in stopwords]
word_freq = {}
for w in meaningful_words:
    word_freq[w] = word_freq.get(w, 0) + 1

unique_meaningful = len(set(meaningful_words))
total_meaningful = len(meaningful_words)
vocab_ratio = unique_meaningful / total_meaningful

print(f"\nDEBUG:")
print(f"word_count: {len(words)}")
print(f"meaningful words: {meaningful_words}")
print(f"vocab_ratio: {vocab_ratio:.2f}")
print(f"max_repetition: {max(word_freq.values())}")
print(f"word_freq: {sorted(word_freq.items(), key=lambda x: -x[1])[:5]}")
"""