from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Load dataset
data = pd.read_csv("spam.csv")

# Split input and output
X = data["message"]
y = data["label"]

# Convert text to numbers
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_vectorized, y)

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        message = request.form["message"]
        message_vector = vectorizer.transform([message])
        prediction = model.predict(message_vector)

        if prediction[0] == "spam":
            result = "SPAM MESSAGE ❌"
        else:
            result = "NOT SPAM ✅"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
