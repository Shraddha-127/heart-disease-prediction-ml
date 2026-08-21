"""
chatbot.py
-----------------------------------------------------------------
AI Health Chatbot integration for CardioAI (Heart Disease Prediction
System), powered by a locally running Ollama instance (Llama 3).

This module only handles:
  1. Talking to the local Ollama HTTP API (http://localhost:11434)
  2. The system prompt / safety rules for the assistant
  3. Turning the user's most recent prediction into plain-text context
     so the assistant can help explain it in simple terms

app.py imports from here; no Flask objects are created in this file,
keeping the chatbot logic modular and independent of the rest of the
application.

BEFORE THIS WILL WORK, Ollama must be installed and running locally:

    1. Install Ollama:       https://ollama.com/download
    2. Start the server:     ollama serve
    3. Pull the model once:  ollama pull llama3
       (or just run it, which pulls it automatically)
                              ollama run llama3

By default this module talks to http://localhost:11434 and uses the
"llama3" model. Both can be overridden with environment variables:

    OLLAMA_BASE_URL  (default: http://localhost:11434)
    OLLAMA_MODEL     (default: llama3)
-----------------------------------------------------------------
"""

import os
import json
import requests

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")

# Ollama can be slow on CPU-only machines, especially for the first
# request after startup (model has to load into memory) - keep a
# generous timeout so the user gets a real reply instead of a false
# "offline" error. Streaming (below) means the user sees the reply
# forming well before this timeout would ever matter.
OLLAMA_TIMEOUT_SECONDS = 120

# How many prior turns (user+assistant messages) to forward to Ollama
# on each request. Kept short - every extra turn adds to the prompt
# Ollama has to re-process ("prefill") before it can start generating,
# which is a big chunk of perceived latency on CPU-only setups.
MAX_HISTORY_TURNS = 6

# Caps how many tokens the model is allowed to generate per reply.
# The system prompt already asks for concise answers; this is a hard
# backstop so a rambling response can't single-handedly make the chat
# feel slow. Lower = faster replies, at the cost of shorter answers.
MAX_RESPONSE_TOKENS = 350

# Keeps the model loaded in memory between requests instead of
# unloading it after Ollama's default ~5 minutes. Avoids paying the
# (often multi-second) model-load cost again on the next message.
OLLAMA_KEEP_ALIVE = "30m"


# ---------------------------------------------------------------
# System Prompt - defines the assistant's role and safety rules
# ---------------------------------------------------------------

SYSTEM_PROMPT = """You are "CardioAI Assistant", a professional health information \
assistant embedded inside the CardioAI heart disease prediction web application.

Your role and STRICT rules (never break these, even if the user insists):

1. You are an EDUCATIONAL assistant only. You explain concepts in clear,
   simple, non-technical language.
2. You NEVER diagnose any disease or medical condition for the user, even
   if directly asked. You are not a doctor and this is not a clinical tool.
3. You NEVER prescribe, recommend, or suggest specific medications, dosages,
   supplements, or treatments.
4. You MAY explain, in general educational terms, what a risk prediction or
   confidence score means, and which clinical factors (e.g. cholesterol,
   blood pressure, ECG results, chest pain type) generally relate to
   cardiovascular risk. Always frame this as general education about how
   the factor relates to heart health in general - never as a personal
   medical judgement about the user.
5. You ALWAYS encourage the user to consult a qualified healthcare
   professional (such as a cardiologist or their doctor) for any actual
   diagnosis, treatment, or medical decision. Include a brief reminder of
   this whenever the conversation touches on their personal result.
6. You are happy to discuss general heart-health topics: diet, exercise,
   lifestyle factors, what medical terms mean, how to read the CardioAI
   dashboard, stress management, sleep, etc.
7. Keep answers concise (a few short paragraphs at most), warm, and easy
   for a non-medical audience to understand. Use plain language over
   jargon, and briefly define any medical term you do use.
8. If asked something unrelated to health or this app, politely redirect
   the conversation back to heart-health topics.

If prediction context is supplied in a separate system message below, use
it ONLY to help the user understand their own result in educational terms
(what these kinds of numbers/factors generally mean) - never to confirm,
deny, or refine a diagnosis.
"""


# ---------------------------------------------------------------
# Ollama availability check
# ---------------------------------------------------------------

def is_ollama_available():
    """
    Quick health check to see whether a local Ollama server is
    reachable. Used by the frontend to show an "assistant offline"
    notice instead of letting every chat message time out.
    """
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


# ---------------------------------------------------------------
# Build context from the user's latest prediction
# ---------------------------------------------------------------

# Human-friendly labels for the raw form/DB codes, so the chatbot's
# context reads naturally instead of showing raw numeric codes.
SEX_LABELS = {1: "Male", 0: "Female", "1": "Male", "0": "Female"}
CHEST_PAIN_LABELS = {
    0: "Typical Angina", 1: "Atypical Angina",
    2: "Non-anginal Pain", 3: "Asymptomatic"
}
RESTECG_LABELS = {0: "Normal", 1: "ST-T Abnormality", 2: "Left Ventricular Hypertrophy"}
SLOPE_LABELS = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}


def _label(mapping, raw_value):
    try:
        return mapping.get(int(float(raw_value)), raw_value)
    except (TypeError, ValueError):
        return raw_value


def build_prediction_context(latest_prediction, explanation_lines=None, top_features=None):
    """
    Turns the user's most recent prediction (a row from the `predictions`
    table, e.g. via database.get_latest_prediction) into a short
    plain-text summary the chatbot can use to explain the result.

    Parameters
    ----------
    latest_prediction : dict | sqlite3.Row | None
        Expected keys: prediction, risk_level, confidence, age, sex,
        chest_pain, resting_bp, cholesterol, max_hr, exercise_angina,
        oldpeak, st_slope, ca, thal. None if no prediction exists yet.
    explanation_lines : list[str] | None
        The plain-English explanation lines generated for the CURRENT
        request (session["explanation"]), if this is being called right
        after a fresh prediction. Optional - omitted for older/history
        predictions where this isn't stored.
    top_features : list[tuple] | None
        Optional list of (feature_name, importance_value) from SHAP /
        feature-importance, if available for the current request.

    Returns
    -------
    str | None - None if there's nothing to summarize yet.
    """

    if not latest_prediction:
        return None

    data = dict(latest_prediction)

    prediction_text = data.get("prediction")
    if not prediction_text:
        return None

    risk_level = data.get("risk_level", "N/A")
    confidence = data.get("confidence", "N/A")

    lines = [
        "The user's most recent heart disease prediction result on CardioAI:",
        f"- Prediction: {prediction_text}",
        f"- Risk Level: {risk_level}",
        f"- Confidence: {confidence}%",
    ]

    detail_parts = []
    if data.get("age") is not None:
        detail_parts.append(f"Age {data['age']}")
    if data.get("sex") is not None:
        detail_parts.append(_label(SEX_LABELS, data["sex"]))
    if data.get("chest_pain") is not None:
        detail_parts.append(f"Chest pain type: {_label(CHEST_PAIN_LABELS, data['chest_pain'])}")
    if data.get("resting_bp") is not None:
        detail_parts.append(f"Resting BP {data['resting_bp']} mmHg")
    if data.get("cholesterol") is not None:
        detail_parts.append(f"Cholesterol {data['cholesterol']} mg/dl")
    if data.get("max_hr") is not None:
        detail_parts.append(f"Max heart rate {data['max_hr']}")
    if data.get("rest_ecg") is not None:
        detail_parts.append(f"Resting ECG: {_label(RESTECG_LABELS, data['rest_ecg'])}")
    if data.get("st_slope") is not None:
        detail_parts.append(f"ST slope: {_label(SLOPE_LABELS, data['st_slope'])}")

    if detail_parts:
        lines.append("- Submitted patient details: " + ", ".join(detail_parts))

    if top_features:
        feat_str = ", ".join(f"{feature} (importance {value})" for feature, value in top_features)
        lines.append(f"- Top contributing factors identified by the model: {feat_str}")

    if explanation_lines:
        lines.append("- System-generated explanation shown to the user: " + " ".join(explanation_lines))

    return "\n".join(lines)


# ---------------------------------------------------------------
# Main chat function
# ---------------------------------------------------------------

def open_ollama_stream(user_message, history=None, prediction_context=None):
    """
    Opens a STREAMING chat completion request to the local Ollama API
    and returns either:

      - a generator yielding response text chunks as they arrive
        (so the frontend can render the reply token-by-token instead
        of waiting for the whole thing - this is what makes the chat
        feel fast even though the model itself hasn't gotten faster)

      - OR a plain string error message, if the request could not even
        be started (Ollama unreachable, model missing, etc). Checking
        `isinstance(result, str)` in the caller distinguishes the two.

    Parameters are the same as the old ask_ollama(): user_message (str),
    history (list[dict] of prior {"role", "content"} turns), and
    prediction_context (str | None) from build_prediction_context().
    """

    if not user_message or not user_message.strip():
        return "Please enter a message before sending."

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if prediction_context:
        messages.append({
            "role": "system",
            "content": "Context about the user's latest prediction on CardioAI:\n" + prediction_context
        })

    if history:
        for turn in history[-MAX_HISTORY_TURNS:]:
            role = turn.get("role")
            content = turn.get("content")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": True,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            # Hard cap on reply length - the single biggest lever against
            # a slow-feeling chat besides streaming itself.
            "num_predict": MAX_RESPONSE_TOKENS,
        }
    }

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            stream=True,
            timeout=OLLAMA_TIMEOUT_SECONDS
        )

    except requests.exceptions.ConnectionError:
        return (
            "I can't reach the AI assistant right now. Please make sure Ollama is "
            f"running locally ('ollama serve') and that the '{OLLAMA_MODEL}' model "
            f"has been pulled ('ollama pull {OLLAMA_MODEL}')."
        )

    except requests.exceptions.Timeout:
        return "The AI assistant took too long to respond. Please try again."

    except requests.exceptions.RequestException as exc:
        return f"Unexpected error while contacting the AI assistant: {exc}"

    if response.status_code == 404:
        return (
            f"Ollama responded, but the '{OLLAMA_MODEL}' model isn't available. "
            f"Run 'ollama pull {OLLAMA_MODEL}' and try again."
        )

    if response.status_code != 200:
        return f"Ollama returned an unexpected error (status {response.status_code})."

    def token_generator():
        """
        Ollama's streaming response is newline-delimited JSON - one
        small object per line, each with the next chunk of the reply
        in message.content. We forward just that text to the caller.
        """
        try:
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue

                chunk = (obj.get("message") or {}).get("content", "")
                if chunk:
                    yield chunk

                if obj.get("done"):
                    break
        finally:
            response.close()

    return token_generator()