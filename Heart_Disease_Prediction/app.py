"""
app.py
--------------------------------------------------------------
Heart Disease Prediction — Flask Application
--------------------------------------------------------------
This file contains:
    1. The AUTHENTICATION MODULE (new)
       - Register / Login / Logout
       - Session based route protection
       - Dashboard / History / Profile pages
    2. The ORIGINAL Machine Learning + SHAP + PDF pipeline
       (UNCHANGED LOGIC — only wrapped with login protection and
       linked to the logged-in user for history tracking)

Nothing in the ML prediction logic, SHAP computation, or PDF
generation algorithm has been altered. Only presentation (the
HTML templates it renders) and the addition of `user_id` when
saving a prediction have been added.
--------------------------------------------------------------
"""

from functools import wraps
from datetime import datetime

from flask import (
    Flask, render_template, request, session,
    send_file, redirect, url_for, flash, jsonify, Response
)
from werkzeug.security import generate_password_hash, check_password_hash

import numpy as np
import pandas as pd
import joblib
import shap
import io
import re
import sqlite3

from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4

import database as db

# AI Health Chatbot (Ollama / Llama 3 integration) - see chatbot.py
from chatbot import open_ollama_stream, build_prediction_context, is_ollama_available


# =====================================================================
# PREDICTION PERSISTENCE (existing function — logic UNCHANGED)
# Only addition: `user_id` parameter so a prediction can be linked
# to the account that generated it.
# =====================================================================
def save_prediction(
    user_id,
    age,
    sex,
    chest_pain,
    resting_bp,
    cholesterol,
    fasting_bs,
    rest_ecg,
    max_hr,
    exercise_angina,
    oldpeak,
    st_slope,
    ca,
    thal,
    prediction,
    risk_level,
    confidence
):
    conn = sqlite3.connect("heart_disease.db")
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO predictions(
        user_id,
        age,
        sex,
        chest_pain,
        resting_bp,
        cholesterol,
        fasting_bs,
        rest_ecg,
        max_hr,
        exercise_angina,
        oldpeak,
        st_slope,
        ca,
        thal,
        prediction,
        risk_level,
        confidence
    )

    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """,(
        user_id,
        age,
        sex,
        chest_pain,
        resting_bp,
        cholesterol,
        fasting_bs,
        rest_ecg,
        max_hr,
        exercise_angina,
        oldpeak,
        st_slope,
        ca,
        thal,
        prediction,
        risk_level,
        confidence
    ))

    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id


app = Flask(__name__)
app.secret_key = "heart_prediction_secret"

# ---------------------------------------------------
# Initialize / migrate database (users + predictions)
# ---------------------------------------------------
db.create_database()

# ---------------------------------------------------
# Load Model & Scaler  (UNCHANGED)
# ---------------------------------------------------
model = joblib.load("Heart_Disease_Model.pkl")
scaler = joblib.load("Scaler.pkl")

# ---------------------------------------------------
# SHAP Explainer  (UNCHANGED)
# ---------------------------------------------------
explainer = shap.TreeExplainer(model)

# ---------------------------------------------------
# Feature Names  (UNCHANGED)
# ---------------------------------------------------
feature_names = [
    "Age",
    "Sex",
    "ChestPain",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "RestECG",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "ST_Slope",
    "CA",
    "Thal"
]

MODEL_ACCURACY = 92.19  # displayed on dashboard / stats section


# =====================================================================
# AUTHENTICATION HELPERS
# =====================================================================
EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MOBILE_REGEX = re.compile(r"^[6-9]\d{9}$")  # 10-digit mobile number


def login_required(view_func):
    """Decorator: redirect to login page if user is not authenticated."""
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            flash("Please login to continue.", "warning")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    return wrapped


def password_is_strong(password):
    """
    Basic password strength rule:
    at least 6 characters, containing letters and numbers.
    """
    if len(password) < 6:
        return False
    if not re.search(r"[A-Za-z]", password):
        return False
    if not re.search(r"[0-9]", password):
        return False
    return True


@app.context_processor
def inject_has_recent_prediction():
    """
    Makes `has_recent_prediction` available in every template that
    extends base.html, so the chatbot's "Explain my latest result"
    quick-action button can show up on any page (dashboard, history,
    profile, predict) once the user has made at least one prediction -
    not just on the predict page immediately after submitting.
    """
    if "user_id" not in session:
        return {"has_recent_prediction": False}

    latest = db.get_latest_prediction(session["user_id"])
    return {"has_recent_prediction": latest is not None}


# =====================================================================
# Recommendations  (UNCHANGED)
# =====================================================================
def get_recommendations(prediction):
    if prediction == 1:
        return [
            "Consult a cardiologist immediately.",
            "Reduce salt intake.",
            "Maintain healthy body weight.",
            "Exercise at least 30 minutes daily.",
            "Avoid smoking and alcohol.",
            "Monitor blood pressure regularly.",
            "Eat fruits and green vegetables.",
            "Reduce cholesterol-rich foods.",
            "Manage stress with yoga or meditation.",
            "Take medications only as prescribed."
        ]
    else:
        return [
            "Maintain a healthy lifestyle.",
            "Exercise regularly.",
            "Eat balanced meals.",
            "Stay hydrated.",
            "Avoid smoking.",
            "Maintain healthy cholesterol.",
            "Check BP periodically.",
            "Sleep 7–8 hours daily.",
            "Manage stress.",
            "Continue regular health checkups."
        ]


# =====================================================================
# AI Explanation  (UNCHANGED)
# =====================================================================
def get_ai_explanation(prediction, confidence):

    if prediction == 1:
        return [
            f"The model predicts HIGH RISK with {confidence:.2f}% confidence.",
            "Several clinical parameters indicate increased cardiovascular risk.",
            "Age, cholesterol, blood pressure, and ECG-related values strongly influenced the prediction.",
            "Early medical consultation is recommended.",
            "Lifestyle modifications can significantly reduce future complications."
        ]
    else:
        return [
            f"The model predicts LOW RISK with {confidence:.2f}% confidence.",
            "The entered clinical values appear to be within a healthy range.",
            "The possibility of heart disease is comparatively low.",
            "Continue maintaining a healthy lifestyle.",
            "Regular health checkups are still recommended."
        ]


# =====================================================================
# AUTHENTICATION ROUTES
# =====================================================================
@app.route("/", methods=["GET"])
def login_page():
    """Login page — the app's entry point."""
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return render_template("login.html")


@app.route("/login", methods=["POST"])
def login():
    """Authenticate the user and start a session."""
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    remember = request.form.get("remember")

    if not email or not password:
        flash("Please enter both email and password.", "error")
        return redirect(url_for("login_page"))

    user = db.get_user_by_email(email)

    if user is None or not check_password_hash(user["password"], password):
        flash("Invalid email or password.", "error")
        return redirect(url_for("login_page"))

    # Successful login
    session["user_id"] = user["id"]
    session["fullname"] = user["fullname"]
    session["email"] = user["email"]

    if remember:
        session.permanent = True
        app.permanent_session_lifetime = 60 * 60 * 24 * 7  # 7 days

    flash(f"Welcome back, {user['fullname']}!", "success")

    # After successful login, send user into the app (dashboard, which
    # itself provides quick access to the existing prediction page).
    return redirect(url_for("dashboard"))


@app.route("/register", methods=["GET", "POST"])
def register():
    """Registration page — create a new user account."""
    if request.method == "GET":
        return render_template("register.html")

    fullname = request.form.get("fullname", "").strip()
    email = request.form.get("email", "").strip().lower()
    mobile = request.form.get("mobile", "").strip()
    password = request.form.get("password", "")
    confirm_password = request.form.get("confirm_password", "")

    # ---------------- Server-side validation ----------------
    if not fullname or not email or not mobile or not password or not confirm_password:
        flash("All fields are required.", "error")
        return redirect(url_for("register"))

    if not EMAIL_REGEX.match(email):
        flash("Please enter a valid email address.", "error")
        return redirect(url_for("register"))

    if not MOBILE_REGEX.match(mobile):
        flash("Please enter a valid 10-digit mobile number.", "error")
        return redirect(url_for("register"))

    if not password_is_strong(password):
        flash("Password must be at least 6 characters and include letters and numbers.", "error")
        return redirect(url_for("register"))

    if password != confirm_password:
        flash("Passwords do not match.", "error")
        return redirect(url_for("register"))

    if db.email_exists(email):
        flash("An account with this email already exists. Please login.", "error")
        return redirect(url_for("register"))

    # ---------------- Create account ----------------
    hashed_password = generate_password_hash(password)
    db.add_user(fullname, email, mobile, hashed_password)

    flash("Account created successfully! Please login.", "success")
    return redirect(url_for("login_page"))


@app.route("/logout")
def logout():
    """Clear the session and return to the login page."""
    session.clear()
    flash("You have been logged out successfully.", "success")
    return redirect(url_for("login_page"))


@app.route("/forgot-password")
def forgot_password():
    """Placeholder forgot-password page (feature not yet implemented)."""
    return render_template("forgot_password.html")


# =====================================================================
# DASHBOARD
# =====================================================================
@app.route("/dashboard")
@login_required
def dashboard():
    user = db.get_user_by_id(session["user_id"])
    total_predictions = db.get_prediction_count(session["user_id"])
    recent = db.get_latest_prediction(session["user_id"])

    return render_template(
        "dashboard.html",
        user=user,
        total_predictions=total_predictions,
        recent=recent,
        model_accuracy=MODEL_ACCURACY
    )


# =====================================================================
# HEART DISEASE PREDICTION  (UNCHANGED ML / SHAP LOGIC)
# =====================================================================
@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():

    if request.method == "POST":

        try:
            # Collect input values
            inputs = [
                float(request.form["age"]),
                float(request.form["sex"]),
                float(request.form["cp"]),
                float(request.form["trestbps"]),
                float(request.form["chol"]),
                float(request.form["fbs"]),
                float(request.form["restecg"]),
                float(request.form["thalach"]),
                float(request.form["exang"]),
                float(request.form["oldpeak"]),
                float(request.form["slope"]),
                float(request.form["ca"]),
                float(request.form["thal"])
            ]
            # Save patient information for PDF report
            session["age"] = inputs[0]
            session["sex"] = inputs[1]
            session["resting_bp"] = inputs[3]
            session["cholesterol"] = inputs[4]
            session["max_hr"] = inputs[7]

            # Convert to NumPy array
            data = np.array(inputs).reshape(1, -1)

            # Convert to DataFrame (removes StandardScaler warning)
            df = pd.DataFrame(data, columns=feature_names)

            # Scale input
            scaled = scaler.transform(df)

            # Prediction
            prediction = model.predict(scaled)[0]

            # Confidence Score
            probabilities = model.predict_proba(scaled)[0]
            confidence = round(max(probabilities) * 100, 2)

            # Risk Label
            if prediction == 1:
              prediction_text = "Heart Disease Detected"
              risk_level = "High Risk"
            else:
              prediction_text = "No Heart Disease Detected"
              risk_level = "Low Risk"

            new_prediction_id = save_prediction(
                session["user_id"],
                inputs[0],   # age
                inputs[1],   # sex
                inputs[2],   # chest_pain
                inputs[3],   # resting_bp
                inputs[4],   # cholesterol
                inputs[5],   # fasting_bs
                inputs[6],   # rest_ecg
                inputs[7],   # max_hr
                inputs[8],   # exercise_angina
                inputs[9],   # oldpeak
                inputs[10],  # st_slope
                inputs[11],  # ca
                inputs[12],  # thal
                prediction_text,
                risk_level,
                confidence
            )

            # AI Explanation
            explanation = get_ai_explanation(prediction, confidence)

            # Recommendations
            recommendations = get_recommendations(prediction)

            # Store report data in session
            session["prediction"] = prediction_text
            session["risk_level"] = risk_level
            session["confidence"] = confidence
            session["recommendations"] = recommendations
            session["age"] = inputs[0]
            session["sex"] = inputs[1]
            session["resting_bp"] = inputs[3]
            session["cholesterol"] = inputs[4]
            session["max_hr"] = inputs[7]
            session["explanation"] = explanation
            session["last_prediction_id"] = new_prediction_id

            # ---------------------------------------
            # SHAP Feature Importance
            # ---------------------------------------
            try:

                shap_output = explainer(scaled)
                shap_values = shap_output.values

                # Compatible with multiple SHAP versions
                if shap_values.ndim == 3:
                    values = shap_values[0, :, 1]
                elif shap_values.ndim == 2:
                    values = shap_values[0]
                else:
                    values = shap_values

                importance = []

                for feature, value in zip(feature_names, values):
                    importance.append(
                        (
                            feature,
                            round(abs(float(value)), 4)
                        )
                    )
                importance.sort(key=lambda x: x[1], reverse=True)
                top_features = importance[:5]
                session["top_features"] = top_features

            except Exception as e:

                print("SHAP Error:", e)

                # Fallback to Random Forest feature importance
                try:
                    importance = []

                    for feature, value in zip(feature_names, model.feature_importances_):
                        importance.append(
                            (
                                feature,
                                round(float(value), 4)
                            )
                        )

                    importance.sort(key=lambda x: x[1], reverse=True)
                    top_features = importance[:5]
                    session["top_features"] = top_features

                except Exception:
                    top_features = []

            # Normalize SHAP importances into 0-100 scale for animated bars
            max_importance = max([v for _, v in top_features], default=0) or 1
            top_features_display = [
                (feature, value, round((value / max_importance) * 100, 1))
                for feature, value in top_features
            ]

            return render_template(
                "predict.html",
                prediction=prediction_text,
                risk_level=risk_level,
                confidence=confidence,
                explanation=explanation,
                recommendations=recommendations,
                top_features=top_features_display,
                patient={
                    "age": inputs[0],
                    "sex": inputs[1],
                    "cp": inputs[2],
                    "trestbps": inputs[3],
                    "chol": inputs[4],
                    "fbs": inputs[5],
                    "restecg": inputs[6],
                    "thalach": inputs[7],
                    "exang": inputs[8],
                    "oldpeak": inputs[9],
                    "slope": inputs[10],
                }
            )

        except Exception as e:
            return render_template(
                "predict.html",
                error=str(e)
            )

    return render_template("predict.html")


# =====================================================================
# PDF REPORT GENERATION — HOSPITAL STYLE (algorithm/data flow UNCHANGED,
# only visual styling improved: header, footer, page numbers, colors)
# =====================================================================
def _build_report_pdf(prediction, risk_level, confidence, recommendations,
                       age, sex, resting_bp, cholesterol, max_hr,
                       generated_for="N/A"):
    """Builds and returns a BytesIO buffer containing the styled PDF report."""

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Heading1"],
        alignment=TA_CENTER, textColor=colors.HexColor("#2563EB"),
        fontSize=22, spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        "SubtitleStyle", parent=styles["Normal"],
        alignment=TA_CENTER, textColor=colors.HexColor("#555555"),
        fontSize=11
    )
    section_style = ParagraphStyle(
        "SectionStyle", parent=styles["Heading2"],
        textColor=colors.HexColor("#0d47a1"), fontSize=14, spaceBefore=14
    )
    normal_style = ParagraphStyle(
        "NormalStyle", parent=styles["Normal"], fontSize=10.5, leading=15
    )

    story = []

    # -----------------------------
    # Hospital-style Header
    # -----------------------------
    story.append(Paragraph("&#10084; CardioAI Diagnostics", title_style))
    story.append(Paragraph("Heart Disease Prediction Report — Explainable AI System", subtitle_style))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#2563EB")))
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        f"<b>Generated for:</b> {generated_for} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<b>Generated on:</b> {datetime.now().strftime('%d %b %Y, %I:%M %p')}",
        normal_style
    ))
    story.append(Spacer(1, 12))

    # -----------------------------
    # Patient Information
    # -----------------------------
    story.append(Paragraph("Patient Information", section_style))

    patient_data = [
        ["Parameter", "Value"],
        ["Age", age],
        ["Gender", "Male" if sex == 1 else "Female"],
        ["Blood Pressure (mm Hg)", resting_bp],
        ["Cholesterol (mg/dl)", cholesterol],
        ["Maximum Heart Rate", max_hr]
    ]

    patient_table = Table(patient_data, colWidths=[3 * inch, 3 * inch])
    patient_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.75, colors.HexColor("#cccccc")),
        ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#f0f6ff")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 16))

    # -----------------------------
    # Prediction Summary
    # -----------------------------
    story.append(Paragraph("Prediction Summary", section_style))

    result_color = colors.HexColor("#d32f2f") if prediction == "Heart Disease Detected" else colors.HexColor("#2e7d32")

    prediction_data = [
        ["Parameter", "Result"],
        ["Prediction", prediction],
        ["Risk Level", risk_level],
        ["Confidence", f"{confidence}%"]
    ]

    prediction_table = Table(prediction_data, colWidths=[3 * inch, 3 * inch])
    prediction_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), result_color),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.75, colors.HexColor("#cccccc")),
        ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#f7f7f7")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(prediction_table)
    story.append(Spacer(1, 16))

    # -----------------------------
    # Recommendations
    # -----------------------------
    story.append(Paragraph("Personalized Recommendations", section_style))

    for item in recommendations:
        story.append(Paragraph(f"&#10003; {item}", normal_style))

    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#cccccc")))
    story.append(Paragraph(
        "<i>This report is generated by an AI-based decision support tool and does "
        "not replace professional medical advice. Please consult a certified "
        "cardiologist for diagnosis and treatment.</i>",
        ParagraphStyle("Disclaimer", parent=normal_style, fontSize=8.5,
                       textColor=colors.HexColor("#777777"), spaceBefore=8)
    ))

    def _footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#999999"))
        canvas.drawCentredString(
            A4[0] / 2, 0.35 * inch,
            f"CardioAI Diagnostics — Confidential Medical Report  |  Page {doc_.page}"
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)

    buffer.seek(0)
    return buffer


@app.route("/download_report")
@login_required
def download_report():
    """Download the PDF report for the prediction just made (session based).
    Data source and computation are unchanged from the original app —
    only the visual design of the PDF has been improved."""

    prediction = session.get("prediction", "N/A")
    risk_level = session.get("risk_level", "N/A")
    confidence = session.get("confidence", "N/A")
    recommendations = session.get("recommendations", [])

    age = session.get("age", "N/A")
    sex = session.get("sex", "N/A")
    resting_bp = session.get("resting_bp", "N/A")
    cholesterol = session.get("cholesterol", "N/A")
    max_hr = session.get("max_hr", "N/A")

    buffer = _build_report_pdf(
        prediction, risk_level, confidence, recommendations,
        age, sex, resting_bp, cholesterol, max_hr,
        generated_for=session.get("fullname", "N/A")
    )

    return send_file(
        buffer,
        as_attachment=True,
        download_name="Heart_Disease_Report.pdf",
        mimetype="application/pdf"
    )


@app.route("/download_report/<int:prediction_id>")
@login_required
def download_report_by_id(prediction_id):
    """Re-download the PDF for a past prediction from the History page."""

    record = db.get_prediction_by_id(prediction_id, session["user_id"])

    if record is None:
        flash("Prediction record not found.", "error")
        return redirect(url_for("history"))

    prediction_flag = 1 if record["prediction"] == "Heart Disease Detected" else 0
    recommendations = get_recommendations(prediction_flag)

    buffer = _build_report_pdf(
        record["prediction"], record["risk_level"], record["confidence"],
        recommendations,
        record["age"], record["sex"], record["resting_bp"],
        record["cholesterol"], record["max_hr"],
        generated_for=session.get("fullname", "N/A")
    )

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"Heart_Disease_Report_{prediction_id}.pdf",
        mimetype="application/pdf"
    )


# =====================================================================
# HISTORY
# =====================================================================
@app.route("/history")
@login_required
def history():
    records = db.get_predictions_by_user(session["user_id"])
    return render_template("history.html", records=records)


@app.route("/history/delete/<int:prediction_id>", methods=["POST"])
@login_required
def delete_history_record(prediction_id):
    db.delete_prediction(prediction_id, session["user_id"])
    flash("Prediction record deleted.", "success")
    return redirect(url_for("history"))


# =====================================================================
# PROFILE
# =====================================================================
@app.route("/profile")
@login_required
def profile():
    user = db.get_user_by_id(session["user_id"])
    total_predictions = db.get_prediction_count(session["user_id"])
    return render_template("profile.html", user=user, total_predictions=total_predictions)


@app.route("/profile/edit", methods=["POST"])
@login_required
def edit_profile():
    fullname = request.form.get("fullname", "").strip()
    mobile = request.form.get("mobile", "").strip()

    if not fullname or not mobile:
        flash("Name and mobile number are required.", "error")
        return redirect(url_for("profile"))

    if not MOBILE_REGEX.match(mobile):
        flash("Please enter a valid 10-digit mobile number.", "error")
        return redirect(url_for("profile"))

    db.update_user_profile(session["user_id"], fullname, mobile)
    session["fullname"] = fullname
    flash("Profile updated successfully.", "success")
    return redirect(url_for("profile"))


@app.route("/profile/change-password", methods=["POST"])
@login_required
def change_password():
    current_password = request.form.get("current_password", "")
    new_password = request.form.get("new_password", "")
    confirm_new_password = request.form.get("confirm_new_password", "")

    user = db.get_user_by_id(session["user_id"])

    if not check_password_hash(user["password"], current_password):
        flash("Current password is incorrect.", "error")
        return redirect(url_for("profile"))

    if not password_is_strong(new_password):
        flash("New password must be at least 6 characters and include letters and numbers.", "error")
        return redirect(url_for("profile"))

    if new_password != confirm_new_password:
        flash("New passwords do not match.", "error")
        return redirect(url_for("profile"))

    db.update_user_password(session["user_id"], generate_password_hash(new_password))
    flash("Password changed successfully.", "success")
    return redirect(url_for("profile"))


# =====================================================================
# AI Health Chatbot API (Ollama / Llama 3)
# =====================================================================
# All Ollama-specific logic (system prompt, HTTP calls, error handling)
# lives in chatbot.py to keep this file focused on the prediction
# system. Both routes require login since the chatbot looks up the
# current user's own latest prediction from the database.
# =====================================================================

@app.route("/api/chat/status")
@login_required
def api_chat_status():
    """
    Lets the frontend check whether a local Ollama server is running
    before the user tries to chat, so it can show a friendly
    "assistant offline" notice instead of a failed request.
    """
    return jsonify({"available": is_ollama_available()})


@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    """
    Main chat endpoint. Accepts JSON:
        {
            "message": "user's latest message",
            "history": [{"role": "user"|"assistant", "content": "..."}, ...]
        }

    Conversation history is kept client-side (in the browser) and sent
    with every request rather than stored server-side, so nothing
    chat-related is added to the session or database.

    The reply is STREAMED back as plain text, chunk by chunk, as Ollama
    generates it - so the frontend can show the answer forming in real
    time instead of a long silent wait followed by the whole message
    appearing at once. If the connection to Ollama can't even be
    opened (offline, model missing, etc.), a normal JSON error is
    returned instead of a stream - see open_ollama_stream()'s
    string-vs-generator return convention.

    The user's most recent SAVED prediction (from the `predictions`
    table) is automatically pulled in as context so the assistant can
    help explain it - this works whether the prediction was just made
    this request or was saved in an earlier session. If the request
    happens to be for the SAME prediction that is still in the current
    Flask session (i.e. the user just ran a prediction and immediately
    opened the chat), the richer session-stored explanation / SHAP
    feature importances are included too.
    """

    data = request.get_json(silent=True) or {}

    user_message = (data.get("message") or "").strip()
    history = data.get("history") or []

    if not isinstance(history, list):
        history = []

    latest_prediction = db.get_latest_prediction(session["user_id"])

    explanation_lines = None
    top_features = None

    if latest_prediction is not None and session.get("last_prediction_id") == latest_prediction["id"]:
        explanation_lines = session.get("explanation")
        top_features = session.get("top_features")

    prediction_context = build_prediction_context(
        latest_prediction,
        explanation_lines=explanation_lines,
        top_features=top_features
    )

    stream_or_error = open_ollama_stream(
        user_message,
        history=history,
        prediction_context=prediction_context
    )

    # open_ollama_stream() returns a plain string if the request could
    # not be started at all (offline Ollama, missing model, etc.) - in
    # that case respond with normal JSON so the frontend's existing
    # error-handling path (data.success === false) still works.
    if isinstance(stream_or_error, str):
        return jsonify({"success": False, "error": stream_or_error})

    return Response(stream_or_error, mimetype="text/plain")


# ---------------------------------------------------
# Run Flask App
# ---------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
