import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.utils import resample
import pickle

fake_df = pd.read_csv("Fake.csv")
real_df = pd.read_csv("True.csv")

fake_df["label"] = 0
real_df["label"] = 1

# Downsample fake news to match real news count
fake_df_downsampled = resample(fake_df, replace=False, n_samples=len(real_df), random_state=42)
df = pd.concat([fake_df_downsampled, real_df]).sample(frac=1, random_state=42).reset_index(drop=True)

print("Fake News:", df[df['label'] == 0].shape[0])
print("Real News:", df[df['label'] == 1].shape[0])

X = df["text"]
y = df["label"]

tfidf = TfidfVectorizer(stop_words="english", max_df=0.7)
X_tfidf = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf.pkl", "wb"))

# Test on known samples
sample_fake = "NASA confirms Earth will go dark for six days due to planetary alignment."
sample_real = "The U.S. economy added 250,000 jobs in April, beating expectations."

sample_tfidf = tfidf.transform([sample_fake, sample_real])
sample_pred = model.predict(sample_tfidf)

print("Sample Fake Prediction:", "Real" if sample_pred[0] == 1 else "Fake")
print("Sample Real Prediction:", "Real" if sample_pred[1] == 1 else "Fake")
