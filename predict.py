import pickle

model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

while True:
    text = input("\nEnter news text (or type 'exit' to quit): ")
    if text.lower() == "exit":
        break
    tfidf_input = tfidf.transform([text])
    prediction = model.predict(tfidf_input)[0]
    confidence = model.predict_proba(tfidf_input)[0][prediction]
    result = "Real" if prediction == 1 else "Fake"
    print(f"\nPrediction: {result} (Confidence: {confidence * 100:.2f}%)")
