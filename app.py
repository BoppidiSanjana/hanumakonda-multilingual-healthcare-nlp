
import os
import json
import uuid
import sqlite3
from datetime import datetime, timezone
from functools import wraps
from urllib.parse import quote

from flask import (
    Flask, render_template, request, jsonify,
    redirect, url_for, flash, session
)
from werkzeug.security import generate_password_hash, check_password_hash

from nlp.pipeline import analyze_text, build_followup_questions
from ml.predictor import predict_category, model_status


APP_DIR = os.path.dirname(os.path.abspath(__file__))

if os.environ.get("VERCEL"):
    DB_PATH = "/tmp/healthcare.db"
else:
    DB_PATH = os.path.join(APP_DIR, "instance", "healthcare.db")

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
app = Flask(__name__)
app.secret_key = os.environ.get(
    "SECRET_KEY",
    "local-development-secret-change-before-deployment"
)

app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


# ---------------- DATABASE SETUP ----------------

def init_db():

    with db() as con:

        con.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                display_name TEXT NOT NULL
            )
        """)

        con.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT,
                original_text TEXT,
                preferred_language TEXT,
                detected_language TEXT,
                analysis_json TEXT,
                followup_json TEXT,
                answers_json TEXT,
                prediction_json TEXT,
                status TEXT DEFAULT 'For professional review',
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)

        con.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id TEXT,
                reviewer TEXT,
                notes TEXT,
                created_at TEXT,
                FOREIGN KEY(report_id) REFERENCES reports(id)
            )
        """)

        # Preserve compatibility with an older local database.
        report_columns = {
            row["name"]
            for row in con.execute(
                "PRAGMA table_info(reports)"
            ).fetchall()
        }

        if "user_id" not in report_columns:
            con.execute(
                "ALTER TABLE reports ADD COLUMN user_id INTEGER NOT NULL DEFAULT 1"
            )

        if "answers_json" not in report_columns:
            con.execute(
                "ALTER TABLE reports ADD COLUMN answers_json TEXT DEFAULT '{}'"
            )

        # Restricted demonstration accounts for doctor/admin testing.
        demo_accounts = [
            ("doctor", "doctor123", "doctor", "Demo Doctor"),
            ("admin", "admin123", "admin", "Demo Administrator")
        ]

        for username, password, role, display_name in demo_accounts:

            existing = con.execute(
                "SELECT id FROM users WHERE username=?",
                (username,)
            ).fetchone()

            if not existing:
                con.execute("""
                    INSERT INTO users
                    (username, password_hash, role, display_name)
                    VALUES (?, ?, ?, ?)
                """, (
                    username,
                    generate_password_hash(password),
                    role,
                    display_name
                ))


init_db()


# ---------------- USER MANAGEMENT ----------------

def current_user():

    if "user_id" not in session:
        return None

    with db() as con:
        return con.execute("""
            SELECT id, username, role, display_name
            FROM users
            WHERE id=?
        """, (session["user_id"],)).fetchone()


def login_required(function):

    @wraps(function)
    def wrapper(*args, **kwargs):

        if not current_user():
            return redirect(url_for("login"))

        return function(*args, **kwargs)

    return wrapper


def roles_required(*roles):

    def decorator(function):

        @wraps(function)
        @login_required
        def wrapper(*args, **kwargs):

            user = current_user()

            if user["role"] not in roles:
                flash("You do not have permission to access this page.")
                return redirect(url_for("index"))

            return function(*args, **kwargs)

        return wrapper

    return decorator


@app.context_processor
def inject_user():
    return {"logged_in_user": current_user()}


# ---------------- LOGIN ----------------

@app.route("/login", methods=["GET", "POST"])
def login():

    if current_user():
        return redirect(url_for("index"))

    if request.method == "POST":

        username = request.form.get(
            "username", ""
        ).strip()

        password = request.form.get("password", "")

        with db() as con:
            user = con.execute("""
                SELECT *
                FROM users
                WHERE username=?
            """, (username,)).fetchone()

        if user and check_password_hash(
            user["password_hash"], password
        ):

            session.clear()
            session["user_id"] = user["id"]

            flash("Login successful!")

            if user["role"] == "doctor":
                return redirect(url_for("doctor"))

            if user["role"] == "admin":
                return redirect(url_for("admin"))

            return redirect(url_for("index"))

        flash("Invalid username or password.")

    return render_template("login.html")


# ---------------- CREATE ACCOUNT ----------------

@app.route("/register", methods=["GET", "POST"])
def register():

    if current_user():
        return redirect(url_for("index"))

    if request.method == "POST":

        display_name = request.form.get(
            "display_name", ""
        ).strip()

        username = request.form.get(
            "username", ""
        ).strip()

        password = request.form.get("password", "")

        confirm_password = request.form.get(
            "confirm_password", ""
        )

        if not display_name or not username or not password:
            flash("Please fill in all required fields.")
            return render_template("register.html")

        if len(username) < 4:
            flash("Username must contain at least 4 characters.")
            return render_template("register.html")

        if len(password) < 8:
            flash("Password must contain at least 8 characters.")
            return render_template("register.html")

        if password != confirm_password:
            flash("Passwords do not match.")
            return render_template("register.html")

        try:

            with db() as con:

                con.execute("""
                    INSERT INTO users
                    (username, password_hash, role, display_name)
                    VALUES (?, ?, ?, ?)
                """, (
                    username,
                    generate_password_hash(password),
                    "patient",
                    display_name
                ))

            flash("Account created successfully! Please login.")

            return redirect(url_for("login"))

        except sqlite3.IntegrityError:

            flash("Username already exists. Choose another username.")

    return render_template("register.html")


# ---------------- LOGOUT ----------------

@app.route("/logout")
def logout():

    session.clear()

    flash("You have been logged out.")

    return redirect(url_for("login"))


# ---------------- PATIENT DASHBOARD ----------------

@app.route("/")
@roles_required("patient")
def index():

    return render_template(
        "index.html",
        model=model_status()
    )




# ---------------- SPECIALIST & NUTRITION GUIDANCE ----------------

def build_care_recommendations(analysis, original_text=""):
    """
    Suggests an appropriate type of healthcare professional from reported
    symptoms/keywords. This is navigation guidance, not a diagnosis.
    """
    text = (original_text or "").lower()
    symptoms = analysis.get("symptoms", [])
    symptom_text = " ".join(str(item).lower() for item in symptoms)
    combined = f"{text} {symptom_text}"

    # Emergency warning signs must take priority over routine referrals.
    emergency_terms = [
        "chest pain", "chest tightness", "heart attack",
        "severe breathing", "difficulty breathing", "can't breathe",
        "cannot breathe", "shortness of breath", "fainting",
        "unconscious", "stroke", "severe bleeding"
    ]
    emergency = (
        bool(analysis.get("urgent_flag"))
        or bool(analysis.get("breathing_difficulty"))
        or any(term in combined for term in emergency_terms)
    )

    if emergency:
        return {
            "urgent": True,
            "message": (
                "A warning sign may be present. Seek urgent medical care now "
                "or contact local emergency services. Do not wait for an "
                "online specialist recommendation."
            ),
            "specialists": [],
            "nutrition": None
        }

    specialist_rules = [
        (
            ["cardiologist"],
            ["heart", "గుండె", "दिल", "palpitation", "palpitations", "heartbeat", "blood pressure",
             "high bp", "low bp", "cardiac", "cardiovascular"]
        ),
        (
            ["dermatologist"],
            ["skin", "చర్మం", "त्वचा", "rash", "itch", "itching", "acne", "pimple", "eczema",
             "hair fall", "skin allergy"]
        ),
        (
            ["ophthalmologist (eye specialist)"],
            ["eye pain", "red eye", "vision", "blurred vision", "eye irritation"]
        ),
        (
            ["ent specialist"],
            ["ear pain", "ear infection", "nose", "sinus", "throat pain",
             "sore throat", "hearing", "tonsil"]
        ),
        (
            ["gastroenterologist"],
            ["stomach pain", "కడుపు నొప్పి", "पेट दर्द", "abdominal pain", "acidity", "gastric",
             "vomiting", "diarrhea", "diarrhoea", "constipation",
             "digestion", "digestive"]
        ),
        (
            ["orthopedic specialist"],
            ["bone", "joint pain", "knee pain", "back pain", "fracture",
             "muscle pain", "arthritis"]
        ),
        (
            ["neurologist"],
            ["migraine", "severe headache", "numbness", "tingling",
             "seizure", "neurological"]
        ),
        (
            ["gynecologist"],
            ["menstrual", "period pain", "irregular periods",
             "pregnancy", "pelvic pain"]
        ),
        (
            ["pulmonologist"],
            ["persistent cough", "wheezing", "asthma", "lung"]
        ),
        (
            ["dentist"],
            ["tooth pain", "toothache", "gum pain", "dental"]
        )
    ]

    matched = []
    for names, terms in specialist_rules:
        if any(term in combined for term in terms):
            matched.extend(names)

    # Fever and common/general symptoms should start with a general physician.
    general_terms = [
        "fever", "జ్వరం", "बुखार", "temperature", "cold", "cough", "headache",
        "body pain", "తలనొప్పి", "సర్ది", "దగ్గు", "బुखार",
        "body pain", "tiredness", "fatigue", "weakness"
    ]
    if not matched or any(term in combined for term in general_terms):
        matched.insert(0, "general physician")

    # Preserve order and remove duplicates.
    matched = list(dict.fromkeys(matched))

    specialists = [
        {
            "name": name.title(),
            "maps_url": (
                "https://www.google.com/maps/search/?api=1&query="
                + quote(name + " near me")
            )
        }
        for name in matched
    ]

    nutrition_terms = [
        "nutrition", "nutritional", "diet", "food", "ఆహారం", "తిండి",
        "పోషకాహారం", "खाना", "आहार", "पोषण", "meal plan",
        "weight loss", "weight gain", "healthy eating", "diet plan"
    ]
    nutrition = None
    if any(term in text for term in nutrition_terms):
        nutrition = {
            "name": "Registered Dietitian / Clinical Nutritionist",
            "message": (
                "For personalized food or diet guidance, consult a qualified "
                "registered dietitian or clinical nutritionist."
            ),
            "maps_url": (
                "https://www.google.com/maps/search/?api=1&query="
                + quote("registered dietitian clinical nutritionist near me")
            )
        }

    return {
        "urgent": False,
        "message": (
            "These are suggested healthcare professional categories based "
            "on the words reported. They are not a diagnosis."
        ),
        "specialists": specialists,
        "nutrition": nutrition
    }


# ---------------- INITIAL SYMPTOM ANALYSIS ----------------

@app.route("/analyze", methods=["POST"])
@roles_required("patient")
def analyze():

    payload = request.get_json(silent=True) or request.form

    symptoms_text = (
        payload.get("symptoms") or ""
    ).strip()

    preferred_language = (
        payload.get("preferred_language") or "English"
    )

    if not symptoms_text:
        return jsonify({
            "error": "Please enter your symptoms."
        }), 400

    analysis = analyze_text(symptoms_text)

    questions = build_followup_questions(analysis)

    return jsonify({
        "analysis": analysis,
        "questions": questions,
        "preferred_language": preferred_language,
        "notice": "This is an educational research prototype, not a medical diagnosis."
    })


# ---------------- FOLLOW-UP ANSWERS AND REPORT ----------------

@app.route("/complete", methods=["POST"])
@roles_required("patient")
def complete():

    payload = request.get_json(silent=True) or request.form

    symptoms_text = (
        payload.get("symptoms") or ""
    ).strip()

    preferred_language = (
        payload.get("preferred_language") or "English"
    )

    answers = payload.get("answers", {})

    if isinstance(answers, str):
        try:
            answers = json.loads(answers)
        except Exception:
            answers = {}

    if not symptoms_text:
        return jsonify({
            "error": "Original symptom description is missing."
        }), 400

    analysis = analyze_text(symptoms_text)

    # Add follow-up answers to structured information.

    if answers.get("duration"):
        analysis["duration"] = answers["duration"]

    if answers.get("severity"):
        analysis["severity"] = answers["severity"]

    if answers.get("breathing_difficulty") == "yes":

        analysis["breathing_difficulty"] = True

        analysis["symptoms"] = sorted(
            set(
                analysis.get("symptoms", [])
                + ["breathing_difficulty"]
            )
        )

    elif answers.get("breathing_difficulty") == "no":
        analysis["breathing_difficulty"] = False

    if answers.get("fever_present") in ("yes", "no"):
        analysis["fever_present"] = answers["fever_present"]

    if answers.get("additional_symptoms"):

        extra_symptoms = [
            item.strip()
            for item in answers["additional_symptoms"].split(",")
            if item.strip()
        ]

        analysis["patient_added_symptoms"] = extra_symptoms

    if answers.get("other_details"):
        analysis["additional_patient_details"] = (
            answers["other_details"].strip()
        )

    # Generate a demonstration prediction.
    prediction = predict_category(analysis)

    # Add specialist-navigation guidance based on reported symptoms.
    # This does not diagnose a condition or replace a clinician.
    care_recommendations = build_care_recommendations(
        analysis,
        symptoms_text
    )
    prediction["care_recommendations"] = care_recommendations

    # Avoid routine prediction when a warning sign is reported.
    if analysis.get("urgent_flag") or analysis.get("breathing_difficulty"):

        prediction = {
            "category": "Professional assessment required",
            "confidence": None,
            "status": "urgent_review",
            "explanation": (
                "A possible warning sign was reported. "
                "Seek urgent professional medical assessment."
            ),
            "care_recommendations": care_recommendations
        }

    report_id = str(uuid.uuid4())[:8].upper()

    with db() as con:

        con.execute("""
            INSERT INTO reports (
                id,
                user_id,
                created_at,
                original_text,
                preferred_language,
                detected_language,
                analysis_json,
                followup_json,
                answers_json,
                prediction_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            report_id,
            current_user()["id"],
            now(),
            symptoms_text,
            preferred_language,
            analysis["language"],
            json.dumps(analysis, ensure_ascii=False),
            json.dumps(
                build_followup_questions(
                    analyze_text(symptoms_text)
                ),
                ensure_ascii=False
            ),
            json.dumps(answers, ensure_ascii=False),
            json.dumps(prediction, ensure_ascii=False)
        ))

    return jsonify({
        "report_id": report_id,
        "redirect": url_for(
            "report",
            report_id=report_id
        )
    })


# ---------------- REPORT HISTORY ----------------

@app.route("/reports")
@roles_required("patient")
def reports():

    with db() as con:

        rows = con.execute("""
            SELECT *
            FROM reports
            WHERE user_id=?
            ORDER BY created_at DESC
            LIMIT 100
        """, (current_user()["id"],)).fetchall()

    return render_template(
        "reports.html",
        reports=rows
    )


# ---------------- SINGLE REPORT ----------------

def get_report(report_id):

    with db() as con:

        return con.execute("""
            SELECT *
            FROM reports
            WHERE id=?
        """, (report_id,)).fetchone()


@app.route("/report/<report_id>")
@login_required
def report(report_id):

    row = get_report(report_id)

    user = current_user()

    if not row:
        return "Report not found", 404

    if user["role"] == "patient" and row["user_id"] != user["id"]:
        return "You cannot access another patient's report.", 403

    with db() as con:

        reviews = con.execute("""
            SELECT *
            FROM reviews
            WHERE report_id=?
            ORDER BY created_at DESC
        """, (report_id,)).fetchall()

    return render_template(
        "report.html",
        r=row,
        analysis=json.loads(row["analysis_json"]),
        prediction=json.loads(row["prediction_json"]),
        followups=json.loads(row["followup_json"]),
        answers=json.loads(row["answers_json"] or "{}"),
        reviews=reviews
    )


# ---------------- PRINT REPORT ----------------

@app.route("/report/<report_id>/download")
@login_required
def download_report(report_id):

    row = get_report(report_id)

    user = current_user()

    if not row:
        return "Report not found", 404

    if user["role"] == "patient" and row["user_id"] != user["id"]:
        return "You cannot access another patient's report.", 403

    return render_template(
        "print_report.html",
        r=row,
        analysis=json.loads(row["analysis_json"]),
        prediction=json.loads(row["prediction_json"]),
        answers=json.loads(row["answers_json"] or "{}")
    )


# ---------------- DOCTOR PORTAL ----------------

@app.route("/doctor")
@roles_required("doctor")
def doctor():

    with db() as con:

        rows = con.execute("""
            SELECT *
            FROM reports
            ORDER BY created_at DESC
            LIMIT 100
        """).fetchall()

    return render_template(
        "doctor.html",
        reports=rows
    )


@app.route("/doctor/review", methods=["POST"])
@roles_required("doctor")
def doctor_review():

    report_id = request.form.get("report_id", "")

    notes = request.form.get("notes", "").strip()

    if not notes:
        flash("Please enter review notes.")
        return redirect(url_for("doctor"))

    with db() as con:

        exists = con.execute(
            "SELECT id FROM reports WHERE id=?",
            (report_id,)
        ).fetchone()

        if not exists:
            flash("Report not found.")
            return redirect(url_for("doctor"))

        con.execute("""
            INSERT INTO reviews
            (report_id, reviewer, notes, created_at)
            VALUES (?, ?, ?, ?)
        """, (
            report_id,
            current_user()["display_name"],
            notes,
            now()
        ))

    flash("Reviewer note saved successfully.")

    return redirect(
        url_for("report", report_id=report_id)
    )


# ---------------- ADMIN DASHBOARD ----------------

@app.route("/admin")
@roles_required("admin")
def admin():

    with db() as con:

        rows = con.execute("""
            SELECT *
            FROM reports
            ORDER BY created_at DESC
        """).fetchall()

    language_counts = {}
    category_counts = {}

    for row in rows:

        language = row["detected_language"]

        language_counts[language] = (
            language_counts.get(language, 0) + 1
        )

        prediction = json.loads(row["prediction_json"])

        category = prediction.get(
            "category", "Unclassified"
        )

        category_counts[category] = (
            category_counts.get(category, 0) + 1
        )

    return render_template(
        "admin.html",
        total=len(rows),
        language_counts=language_counts,
        category_counts=category_counts,
        reports=rows,
        model=model_status()
    )


# ---------------- HEALTH CHECK ----------------

@app.route("/health")
def health():

    return jsonify({
        "status": "ok",
        "model": model_status()
    })


# ---------------- RUN APPLICATION ----------------

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5001
    )
