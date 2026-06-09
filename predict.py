import joblib

# Load saved files
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# New review
review = ["I love this movie"]

# Convert to TF-IDF
review_vector = vectorizer.transform(review)

# Predict
prediction = model.predict(review_vector)

print("Prediction:", prediction[0])