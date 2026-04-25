from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

app = FastAPI()

# Allow React frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default port
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model and vectorizer once at startup
print("Loading model...")
model = joblib.load("model/helpfulness_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
print("Model loaded successfully!")

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

    spam_words = {"buy", "now", "click", "order", "deal", "offer", "discount"}
    has_spam = any(w in spam_words for w in words)

    if max_repetition >= 4 or (max_repetition >= 2 and has_spam):
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

# --- Data shapes ---
class ReviewInput(BaseModel):
    text: str
    title: str = ""  # optional

class BatchReviewInput(BaseModel):
    reviews: list[dict]  # list of {text, title, rating, url}

# --- Routes ---

@app.get("/")
def root():
    return {"status": "ReviewSweeperAI backend is running"}

@app.post("/predict")
def predict_single(review: ReviewInput):
    """Predict helpfulness of a single review"""
    combined = review.title + " " + review.text
    vectorized = vectorizer.transform([combined])
    prediction = model.predict(vectorized)[0]
    confidence = model.predict_proba(vectorized)[0].max()

    return {
        "helpful": bool(prediction),
        "confidence": round(float(confidence), 3),
        "label": "Helpful" if prediction else "Not Helpful"
    }

@app.post("/analyze")
def analyze_reviews(data: BatchReviewInput):
    """
    Score all reviews, return top 3 helpful positive
    and top 3 helpful critical reviews
    """
    results = []

    for review in data.reviews:
        combined = review.get("title", "") + " " + review.get("text", "")
        vectorized = vectorizer.transform([combined])
        prediction = model.predict(vectorized)[0]
        confidence = float(model.predict_proba(vectorized)[0].max())
        rating = review.get("rating", 3)

        results.append({
            "text": review.get("text", ""),
            "title": review.get("title", ""),
            "rating": rating,
            "url": review.get("url", ""),
            "helpful": bool(prediction),
            "confidence": round(confidence, 3),
        })

    # Separate by positive (4-5 stars) vs critical (1-2 stars)
    positive = [r for r in results if r["rating"] >= 4 and r["helpful"]]
    critical = [r for r in results if r["rating"] <= 2 and r["helpful"]]

    # Sort by confidence, take top 3 each
    top_positive = sorted(positive, key=lambda x: x["confidence"], reverse=True)[:3]
    top_critical = sorted(critical, key=lambda x: x["confidence"], reverse=True)[:3]

    return {
        "top_positive": top_positive,
        "top_critical": top_critical,
        "total_analyzed": len(results),
        "helpful_count": sum(1 for r in results if r["helpful"])
    }


@app.post("/summarize")
async def summarize_reviews(data: BatchReviewInput):
    """
    Use Claude to summarize what the top reviews collectively say
    """
    # Score reviews first
    scored = []
    for review in data.reviews:
        combined = review.get("title", "") + " " + review.get("text", "")
        vectorized = vectorizer.transform([combined])
        prediction = model.predict(vectorized)[0]
        confidence = float(model.predict_proba(vectorized)[0].max())
        scored.append({**review, "helpful": bool(prediction), "confidence": confidence})

    # Get top helpful reviews
    top = sorted([r for r in scored if r["helpful"]], 
                 key=lambda x: x["confidence"], reverse=True)[:6]

    if not top:
        return {"summary": "Not enough helpful reviews to summarize."}

    # Build prompt for Claude
    review_text = "\n\n".join([
        f"Review ({r.get('rating', '?')} stars): {r.get('text', '')}"
        for r in top
    ])

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": f"""Here are the most helpful reviews for a product. 
Summarize in 3 bullet points: what buyers love, what they complain about, and who this product is best for.
Be concise and direct.

Reviews:
{review_text}"""
        }]
    )
    return {"summary": message.content[0].text}

class URLInput(BaseModel):
    url: str

@app.post("/scrape-and-analyze")
async def scrape_and_analyze(data: URLInput):
    """Take an Amazon product URL, scrape reviews, score them"""
    import re
    
    # Extract ASIN from URL (Amazon's product ID)
    asin_match = re.search(r"/dp/([A-Z0-9]{10})", data.url)
    if not asin_match:
        raise HTTPException(status_code=400, detail="Couldn't find a valid Amazon product ID in that URL. Make sure it's a product page.")
    
    asin = asin_match.group(1)

    # Call RapidAPI to get reviews
    import requests as req
    headers = {
        "x-rapidapi-host": "real-time-amazon-data.p.rapidapi.com",
        "x-rapidapi-key": os.getenv("RAPIDAPI_KEY")
    }
    
    params = {
        "asin": asin,
        "country": "IN",  # India since you're in Hyderabad
        "sort_by": "TOP_REVIEWS",
        "star_rating": "ALL",
        "verified_purchases_only": "false",
        "images_or_videos_only": "false",
        "current_format_only": "false"
    }

    response = req.get(
    "https://real-time-amazon-data.p.rapidapi.com/product-reviews",
    headers=headers,
    params=params,
    timeout=15
)

    if response.status_code != 200:
        raise HTTPException(status_code=502, detail="Failed to fetch reviews from Amazon.")

    raw = response.json()
    reviews_data = raw.get("data", {}).get("reviews", [])

    if not reviews_data:
        raise HTTPException(status_code=404, detail="No reviews found for this product.")

    # Format for our model
    reviews = []
    for r in reviews_data:
        reviews.append({
            "title": r.get("review_title", ""),
            "text": r.get("review_comment", ""),
            "rating": int(r.get("review_star_rating", 3)),
            "url": r.get("review_link", "")
        })

    # Score with model
    results = []
    for review in reviews:
        helpful, confidence = score_review(review["title"], review["text"])
        results.append({**review, "helpful": helpful, "confidence": confidence})
    positive = sorted(
        [r for r in results if r["rating"] >= 4 and r["helpful"]],
        key=lambda x: x["confidence"], reverse=True
    )[:3]

    critical = sorted(
        [r for r in results if r["rating"] <= 2 and r["helpful"]],
        key=lambda x: x["confidence"], reverse=True
    )[:3]

    return {
        "product_title": raw.get("data", {}).get("product_title", ""),
        "top_positive": positive,
        "top_critical": critical,
        "total_analyzed": len(results),
        "helpful_count": sum(1 for r in results if r["helpful"])
    }