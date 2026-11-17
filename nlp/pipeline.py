import re

TERMS = {
    "fever": ["fever", "జ్వరం", "జ్వరము", "bukhar", "बुखार", "बुख़ार", "joram", "jwaram", "jvaram"],
    "headache": ["headache", "తలనొప్పి", "తల నొప్పి", "sir dard", "सिरदर्द", "सिर दर्द", "talano ppi", "thalanoppi"],
    "cough": ["cough", "దగ్గు", "खांसी", "khansi", "daggu"],
    "body_pain": ["body pain", "శరీర నొప్పి", "ఒళ్లు నొప్పులు", "बदन दर्द", "शरीर दर्द", "ontlo nopulu"],
    "sore_throat": ["sore throat", "గొంతు నొప్పి", "గొంతునొప్పి", "गले में दर्द", "गला दर्द"],
    "stomach_pain": ["stomach pain", "కడుపు నొప్పి", "पेट दर्द", "pet dard", "kadupu noppi"],
    "vomiting": ["vomiting", "వాంతులు", "उल्टी", "ulti", "vomit"],
    "diarrhea": ["diarrhea", "loose motions", "విరేచనాలు", "दस्त", "dast"],
    "breathing_difficulty": ["difficulty breathing", "shortness of breath", "శ్వాస తీసుకోవడంలో ఇబ్బంది", "ఊపిరి ఆడటం లేదు", "सांस लेने में कठिनाई", "सांस फूलना"]
}
NEGATIONS = ["no", "not", "without", "లేదు", "లేద", "కాదు", "नहीं", "नही", "nahin", "nahi"]
SEVERITY = {
    "mild": ["mild", "తక్కువ", "हल्का", "हल्की"],
    "moderate": ["moderate", "మధ్యస్థ", "मध्यम"],
    "severe": ["severe", "తీవ్రం", "చాలా ఎక్కువ", "गंभीर", "बहुत ज्यादा"]
}
DURATION_PATTERNS = [r"(\d+)\s*(days?|రోజులు|दिन)", r"(\d+)\s*(weeks?|వారాలు|हफ्ते|सप्ताह)", r"(\d+)\s*(months?|నెలలు|महीने)"]

def detect_language(text):
    telugu = bool(re.search(r"[\u0C00-\u0C7F]", text))
    hindi = bool(re.search(r"[\u0900-\u097F]", text))
    english = bool(re.search(r"[A-Za-z]", text))
    if sum([telugu, hindi, english]) > 1: return "Mixed (Telugu/Hindi/English)"
    if telugu: return "Telugu"
    if hindi: return "Hindi"
    low = text.lower()
    if any(w in low for w in ["naku", "naaku", "undi", "vastundi", "ledu", "jwaram", "daggu"]): return "Romanized Telugu / Mixed"
    if any(w in low for w in ["mujhe", "hai", "bukhar", "dard", "khansi", "pet"]): return "Romanized Hindi / Mixed"
    return "English or Romanized input (needs confirmation)"

def find_duration(text):
    for pattern in DURATION_PATTERNS:
        m = re.search(pattern, text, flags=re.I)
        if m: return f"{m.group(1)} {m.group(2)}"
    return None

def analyze_text(text):
    low = text.lower()
    found, negated = [], []
    for symptom, variants in TERMS.items():
        for term in variants:
            idx = low.find(term.lower())
            if idx >= 0:
                context = low[max(0, idx-24):idx]
                is_negated = any(n.lower() in context[-18:] for n in NEGATIONS)
                (negated if is_negated else found).append(symptom)
                break
    severity = next((level for level, variants in SEVERITY.items() if any(v.lower() in low for v in variants)), None)
    urgent = any(x in low for x in ["chest pain", "severe breathing", "difficulty breathing", "ఊపిరి ఆడటం లేదు", "శ్వాస తీసుకోవడంలో ఇబ్బంది", "सांस लेने में कठिनाई", "सीने में दर्द"])
    return {"language": detect_language(text), "symptoms": sorted(set(found)),
            "negated_symptoms": sorted(set(negated)), "duration": find_duration(text),
            "severity": severity, "breathing_difficulty": "breathing_difficulty" in found,
            "urgent_flag": urgent,
            "note": "Keyword-based starter extraction; not clinically validated."}

def build_followup_questions(analysis):
    questions = []
    if not analysis.get("duration"):
        questions.append({"key":"duration","label":"How long have you been experiencing these symptoms?","type":"text","placeholder":"Example: 2 days"})
    if not analysis.get("severity"):
        questions.append({"key":"severity","label":"How severe are the symptoms?","type":"select","options":["Mild","Moderate","Severe","Not sure"]})
    questions.append({"key":"breathing_difficulty","label":"Are you experiencing difficulty breathing?","type":"yesno"})
    if "fever" not in analysis.get("symptoms", []) and "fever" not in analysis.get("negated_symptoms", []):
        questions.append({"key":"fever_present","label":"Are you also experiencing fever?","type":"yesno"})
    questions.append({"key":"additional_symptoms","label":"Any other symptoms you want to add? (comma-separated)","type":"text","placeholder":"Example: dizziness, tiredness"})
    questions.append({"key":"other_details","label":"Any other details you would like included?","type":"textarea","placeholder":"Optional details"})
    return questions
