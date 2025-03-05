import os
import pandas as pd
import feedparser
import pyodbc
import nbformat
import json
from nbconvert.preprocessors import ExecutePreprocessor
from flask import Flask, render_template, redirect, url_for, send_file, request, flash, session,jsonify
from flask_sqlalchemy import SQLAlchemy
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pickle
from werkzeug.security import generate_password_hash, check_password_hash

# Enable HTTP for local testing
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Change this to a secure key


DATABASE_CONFIG = {
    "driver": "ODBC Driver 17 for SQL Server",
    "server": "DESKTOP-13N01JG\\SQLEXPRESS",
    "database": "Next_Word_Predictor",
}

# Construct connection string
app.config['SQLALCHEMY_DATABASE_URI'] = (
    f"mssql+pyodbc://@{DATABASE_CONFIG['server']}/{DATABASE_CONFIG['database']}?trusted_connection=yes&driver={DATABASE_CONFIG['driver']}"
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(100))
    last_name = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True)
    password_hash = db.Column(db.String(200))
    medium_id = db.Column(db.String(100), nullable=True)
    
# Define Medium Articles Table
class MediumArticle(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100))
    title = db.Column(db.String(255))
    summary = db.Column(db.String(500))
    url = db.Column(db.String(500))
    published_date = db.Column(db.String(100))
    author=db.Column(db.String(100))

# Create the database (only first time)
with app.app_context():
    db.create_all()



# Load LSTM Model and Tokenizer
model_path = "lstm_next_word_model.h5" 
tokenizer_path = "tokenizer.pkl" 

    

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        first_name = request.form["first_name"]
        last_name = request.form["last_name"]
        medium_id = request.form["medium_id"]
        email = request.form["email"]
        password = request.form["password"]
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        
        if User.query.filter_by(email=email).first():
            flash("Email already registered. Please Login", "danger")
            return redirect(url_for("signup"))
        
        new_user = User(first_name=first_name, last_name=last_name, medium_id = medium_id,email=email, password_hash=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash("Signup successful! You can now log in.", "success")
        return redirect(url_for("login"))
    
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    print("🔹 Login Page Accessed")
    if "attempts" not in session:
        session["attempts"] = 3  # Set login attempts to 3
        
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        print(f"🔹 Received Login Attempt for Email: {email}")
        user = User.query.filter_by(email=email).first()
        
        if user:
            print(f"🔹 User Found: {user.email}")
            if check_password_hash(user.password_hash, password):
                session["user_id"] = user.id
                session.pop("attempts", None)  # Reset attempts on success
                flash(f"Login successful! Welcome {user.first_name}!", "success")
                print("✅ Login Successful!")
                return redirect(url_for("post_login_actions"))

            else:
                session["attempts"] -= 1  # Reduce attempts
                print(f"❌ Wrong Password. Attempts Left: {session['attempts']}")
                flash(f"Incorrect Password. Attempts left: {session['attempts']}", "danger")
                if session["attempts"] > 0:
                    flash(f"Incorrect Credentials. Try Again. Attempts left: {session['attempts']}", "danger")
                else:
                    flash("Too many failed attempts. Please try again later.", "danger")
                    return redirect(url_for("home"))  # Redirect home after 3 failures
        else:
            print("❌ No Account Found!")
            flash("No account registered with this email. Please Sign Up.", "warning")
    
    return render_template("login.html")


@app.route("/post-login-actions")
def post_login_actions():
    
    export_lstm_data()
    
    run_lstm_notebook()

    return redirect(url_for("dashboard"))



def run_lstm_notebook():
    notebook_path = "medium_lstm.ipynb"  # Ensure the path is correct

    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    executor = ExecutePreprocessor(timeout=600, kernel_name="python3")
    executor.preprocess(notebook, {"metadata": {"path": "."}})

    # Save the executed notebook (optional)
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    print("✅ LSTM Notebook Execution Completed.")



@app.route("/logout")
def logout():
    session.pop("user_id", None)
    flash("Logged out successfully.", "success")
    return redirect(url_for("home"))

@app.route("/fetch-medium/<username>")
def fetch_medium(username):
    user = User.query.filter_by(medium_id=username).first()

    if not user:
        return jsonify({"status": "error", "message": "Medium Username not found. Please Sign Up or Login."}) 
        #flash("Medium Username not present. Please Sign Up or Login.", "warning")
        return redirect(url_for("home"))

    else:
        return jsonify({"status": "success", "message": "Medium Username found. Please enter your password.", "email": user.email, "username": username})
    
@app.route("/verify-password", methods=["POST"])
def verify_password():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    user = User.query.filter_by(medium_id=username).first()
    if user and check_password_hash(user.password_hash, password):
        session["user_id"] = user.id
    
        rss_url = f"https://medium.com/feed/@{username}"
        feed = feedparser.parse(rss_url)
        
        if not feed.entries:
            return f"Error: No articles found for Medium user '@{username}'", 404

        # Store Medium articles in DB
        for entry in feed.entries:
            article = MediumArticle(
                username=username,
                title=entry.title,
                url=entry.link,
                published_date=entry.published,
                summary = entry.get("summary","No Summary Available")[:500],
                author= entry.get("author","Unknown")
            
            )
            db.session.add(article)
        db.session.commit()
        return redirect(url_for("export_lstm_data"))
        return jsonify({"status": "success", "message": "Login successful! Redirecting to dashboard..."})
    
    return jsonify({"status": "error", "message": "Incorrect password. Try again!"})
    

@app.route("/export-medium/<username>")
def export_medium(username):
    articles = MediumArticle.query.filter_by(username=username).all()
    if not articles:
        return f"No data found for Medium user '@{username}'", 404

    # Convert to DataFrame
    data = [{"Title": a.title, "URL": a.url, "Published Date": a.published_date} for a in articles]
    df = pd.DataFrame(data)

    if df.empty:
        return "No articles found to export", 404

    csv_file = f"{username}_medium_articles.csv"
    df.to_csv(csv_file, index=False)

    return send_file(csv_file, as_attachment=True)

@app.route("/export-lstm-data")
def export_lstm_data():
    
    # Check if the user is logged in
    if "user_id" not in session:
        flash("You need to log in first.", "warning")
        return redirect(url_for("login"))

    # Get the logged-in user
    user = User.query.get(session["user_id"])
    
    if not user.medium_id:
        flash("No Medium ID associated with your account.", "danger")
        return redirect(url_for("dashboard"))
    
    
    # Fetch relevant data from the MediumArticle table
    articles = MediumArticle.query.with_entities(MediumArticle.id,MediumArticle.title, MediumArticle.summary,MediumArticle.url,MediumArticle.published_date).all()

    if not articles:
        return jsonify({"status": "error", "message": "No articles found in the database."})

    # Convert to DataFrame
    data = [{"id":article.id,"url":article.url,"title":article.title,"subtitle":article.summary,"image":"","claps":"","responses":"","reading_time":"","publication":"","date":article.published_date} for article in articles]
    df = pd.DataFrame(data)

    # Save to CSV
    csv_filename = "medium_data.csv"
    df.to_csv(csv_filename, index=False, encoding="utf-8")

    return jsonify({"status": "success", "message": "LSTM training data exported!", "file": csv_filename})


@app.route("/dashboard")
def dashboard():    
    if "user_id" not in session:
        flash("You need to log in first.", "warning")
        return redirect(url_for("home"))
    
    user = User.query.get(session["user_id"])
    return render_template("dashboard.html", user=user)


@app.route("/get_status")
def get_status():
    # Check if word_analysis.json exists
    try:
        with open("word_analysis.json", "r") as f:
            word_analysis = json.load(f)
    except FileNotFoundError:
        return jsonify({"status": "error", "message": "Word analysis data not available"}), 404

    return jsonify({
        "status": "success",
        "unique_words": word_analysis["unique_words"],
        "favorite_word": word_analysis["favorite_word"],
        "rejected_word": word_analysis["rejected_word"]
    })

@app.route("/notebook")
def open_notebook():
    return render_template("notebook.html")


try:
    model = load_model(model_path)  # Load trained LSTM model
    with open(tokenizer_path, "rb") as file:
        tokenizer = pickle.load(file)  # Load tokenizer
    print("✅ LSTM Model and Tokenizer Loaded Successfully")
except Exception as e:
    print(f"❌ Error Loading Model: {e}")

# Function to Predict Next Word
def predict_next_word(input_text, num_words=1):
    sequence = tokenizer.texts_to_sequences([input_text])[0]
    sequence = pad_sequences([sequence], maxlen=20, padding='pre')  # Adjust maxlen to match model training
    predicted_words = []

    for _ in range(num_words):
        prediction = model.predict(sequence, verbose=0)
        predicted_index = np.argmax(prediction)
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                predicted_words.append(word)
                break

    return predicted_words

# Flask API for Next-Word Prediction
@app.route("/predict-next-word", methods=["POST"])
def predict():
    data = request.json
    input_text = data.get("text", "").strip()

    if not input_text:
        return jsonify({"status": "error", "message": "No input text provided"})

    predicted_words = predict_next_word(input_text, num_words=3)  # Get top 3 suggestions
    return jsonify({"status": "success", "predictions": predicted_words})


if __name__ == "__main__":
    app.run(debug=True)
