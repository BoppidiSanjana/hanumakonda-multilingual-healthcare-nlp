# Hanumakonda Multilingual Healthcare NLP – v2

## What's included
- Login page shown before the application.
- Three local demonstration roles: patient, doctor and administrator.
- Patient symptom entry in English, Telugu, Hindi and mixed-language text (starter keyword/script logic only).
- Follow-up questions are rendered as answerable form fields.
- The patient submits answers; the final report is saved with original symptoms and follow-up answers.
- Patient-only report history and report access checks.
- Doctor review page for adding reviewer notes.
- Admin/research dashboard with local demo counts.
- Light/dark theme switch; preference is remembered in the browser.
- Printable report (browser Print → Save as PDF).

## Demo login credentials
| Role | Username | Password |
|---|---|---|
| Patient | `patient` | `patient123` |
| Doctor | `doctor` | `doctor123` |
| Admin | `admin` | `admin123` |

These are demonstration credentials. Change them and set a strong `SECRET_KEY` before any deployment. The app is configured for local use only.

## Run on macOS
Open this folder in VS Code, then Terminal → New Terminal:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train_model.py
python app.py
```

Open `http://127.0.0.1:5000`.

If you already created a venv in the previous project, you can remove the `venv` folder and create it again, or simply use a compatible existing environment. The v2 database is created automatically in `instance/healthcare.db`.

## Important limitations and safety
- This is an educational software prototype, not a medical device or diagnostic service.
- The included CSV and trained model use synthetic teaching examples only. A good-looking score from this tiny dataset is not evidence of clinical performance.
- The language detector and symptom extraction are small keyword-based starter rules, not comprehensive multilingual NLP. Romanized and code-mixed language support is limited.
- Do not enter real patient information.
- No medication or treatment recommendations are generated. Urgent warning signs should be assessed by qualified healthcare professionals.
- Before any real healthcare use, obtain authorized and consented/de-identified data, ethics/privacy approval, clinician-reviewed labels and questions, security review, patient-level evaluation splits, language-wise metrics, calibration and abstention testing.
- The built-in demo accounts are not production authentication. Use a production identity system, CSRF protection, HTTPS, secure secret management, rate limits, audit controls and appropriate privacy/retention policies before deployment.
