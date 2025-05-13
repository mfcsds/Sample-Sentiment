import pandas as pd
import re
import joblib

from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from nltk.tokenize import wordpunct_tokenize

# Define stopwords
stop_words = ENGLISH_STOP_WORDS

# Custom lemmatization using a simple rules-based approach
def custom_lemmatizer(word):
    if word.endswith("ing") or word.endswith("ed"):
        return word[:-3]
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word

# Preprocessing using wordpunct_tokenize
def preprocess_and_lemmatize(text):
    tokens = wordpunct_tokenize(text.lower())  # safer than re.findall
    tokens = [custom_lemmatizer(word) for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

# Dataset
data = {
    "tweet": [
        # Positive
        "I’m really excited towards tomorrow for our shop opening!",
        "This product is amazing and works perfectly.",
        "I love the new design! Great work!",
        "This app is fantastic! Highly recommended.",
        "I’m thrilled with the results, thank you!",
        "The support team was very helpful and friendly.",
        "Wonderful experience! Will definitely use again.",
        "This made my day so much better, absolutely love it.",
        "Everything worked perfectly, and I’m very happy with it.",
        "The concert was amazing! Had so much fun.",
        "Fantastic job on the presentation, I’m impressed.",
        "This coffee tastes so good, absolutely delicious.",
        "I really enjoyed the movie, it was so entertaining.",
        "The product exceeded my expectations, highly recommended.",
        "Thank you for the amazing support, I’m so grateful.",
        # Negative
        "I’m so sad about what happened today.",
        "The service was terrible, and I’ll never come back.",
        "This is a boring and useless feature.",
        "Horrible experience, totally disappointed.",
        "I regret buying this product. It's a waste of money.",
        "The delivery was late, and the item was damaged.",
        "This is the worst customer service I've ever received.",
        "Absolutely not worth the price. Very disappointing.",
        "I’m frustrated with the slow and unhelpful support.",
        "The food was awful, I couldn’t even finish it.",
        "I feel cheated by this product, it doesn’t work.",
        "Terrible experience, I wouldn’t recommend it to anyone.",
        "The app keeps crashing, it’s completely unusable.",
        "The quality is poor, and it broke after one use.",
        "The staff was rude and unprofessional.",
        # Neutral
        "Technology is improving rapidly, which is exciting.",
        "It’s just another day, nothing special happening.",
        "The event was okay, neither good nor bad.",
        "I saw the announcement, but it’s not relevant to me.",
        "It’s an average product, works as expected.",
        "Just completed the task, moving on to the next.",
        "The book has some useful tips but nothing groundbreaking.",
        "I heard about the update, but I’m not sure how I feel about it.",
        "The weather today is normal, not too hot or cold.",
        "The report was fine, but it could use some improvement.",
        "The lecture was informative, but not particularly exciting.",
        "The movie was okay, but I wouldn’t watch it again.",
        "Nothing much happened during the meeting.",
        "The article provided useful information, but nothing new.",
        "It’s an ordinary day, nothing out of the ordinary."
    ],
    "sentiment": [
        "positive"] * 15 + ["negative"] * 15 + ["neutral"] * 15
}

# Create DataFrame
df = pd.DataFrame(data)

# Apply preprocessing
df["tweet"] = df["tweet"].apply(preprocess_and_lemmatize)

# Vectorization
vectorizer = CountVectorizer(max_features=1000, min_df=1, max_df=0.1)
X = vectorizer.fit_transform(df["tweet"])
y = df["sentiment"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Model Training
model = MultinomialNB()
model.fit(X_train, y_train)

# Save Model & Vectorizer
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Saved model to sentiment_model.pkl and vectorizer to vectorizer.pkl")

# Evaluation
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Predict on new samples
new_tweets = [
    "I’m so excited about the upcoming launch!",
    "This experience was horrible, I regret coming here.",
    "Technology is fascinating, but it can be frustrating sometimes."
]
new_tweets_processed = [preprocess_and_lemmatize(tweet) for tweet in new_tweets]
new_features = vectorizer.transform(new_tweets_processed)
predictions = model.predict(new_features)

# Print predictions
for tweet, sentiment in zip(new_tweets, predictions):
    print(f"Tweet: {tweet} --> Sentiment: {sentiment}")
