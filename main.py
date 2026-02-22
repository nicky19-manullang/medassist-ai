from fastapi import FastAPI, HTTPException, Depends, Request, Form, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from database import engine, SessionLocal
from starlette.middleware.sessions import SessionMiddleware
import joblib
import numpy as np
import models
import json
import re
import hashlib

# =========================
# INITIAL SETUP
# =========================

try:
    models.Base.metadata.create_all(bind=engine)
    print("✓ Database tables ready")
except Exception as e:
    print(f"✗ Error: {e}")

try:
    model = joblib.load("model_diagnosa.pkl")
    print(f"✓ Model loaded: {model.classes_}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

app = FastAPI(
    title="MedAssist AI",
    description="Sistem Pendukung Diagnosis Penyakit Berbasis AI",
    version="4.0.0"
)
app.add_middleware(SessionMiddleware, secret_key="supersecretkey")

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# =========================
# DATABASE DEPENDENCY
# =========================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =========================
# HELPER FUNCTIONS
# =========================

def hash_password(password):
    """Hash password menggunakan SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(plain_password, hashed_password):
    """Verifikasi password dengan hash"""
    return hash_password(plain_password) == hashed_password


def patient_to_dict(patient):
    return {
        "id": patient.id,
        "name": patient.name,
        "age": patient.age,
        "gender": patient.gender
    }


def parse_symptoms_from_text(text):
    if not text:
        return [0, 0, 0, 0, 0, 0, 0, 0]
    
    text = text.lower().strip()
    negations = ["tidak", "ga", "gak", "nggak", "bukan", "tanpa", "no", "not"]
    
    symptom_map = {
        "demam": "Demam", "panas": "Demam", "febris": "Demam", "suhu": "Demam",
        "fever": "Demam", "hot": "Demam",
        "batuk": "Batuk", "cough": "Batuk", "batik": "Batuk",
        "nyeri otot": "NyeriOtot", "otot": "NyeriOtot", "sakit otot": "NyeriOtot",
        "muscle": "NyeriOtot", "pegal": "NyeriOtot", "linu": "NyeriOtot",
        "mual": "Mual", "nausea": "Mual", "enek": "Mual", "munting": "Mual",
        "ill": "Mual",
        "nyeri perut": "NyeriPerut", "perut": "NyeriPerut", "sakit perut": "NyeriPerut",
        "lambung": "NyeriPerut", "maag": "NyeriPerut", "gerah": "NyeriPerut",
        "stomach": "NyeriPerut", "pain": "NyeriPerut",
        "pusing": "Pusing", "dizziness": "Pusing", "vertigo": "Pusing", "headache": "Pusing",
        "kepala": "Pusing", "sakit kepala": "Pusing",
        "sesak": "Sesak", "nafas": "Sesak", "breath": "Sesak", "napas": "Sesak",
        "dyspnea": "Sesak", "asthma": "Sesak", "bengek": "Sesak", "ngos": "Sesak",
        "tensi": "TensiTinggi", "tekanan darah": "TensiTinggi", "hipertensi": "TensiTinggi",
        "darah tinggi": "TensiTinggi", "pressure": "TensiTinggi", "bp": "TensiTinggi"
    }
    
    symptoms = {
        "Demam": 0, "Batuk": 0, "NyeriOtot": 0, "Mual": 0,
        "NyeriPerut": 0, "Pusing": 0, "Sesak": 0, "TensiTinggi": 0
    }
    
    words = re.findall(r'\b\w+\b', text)
    
    for i, word in enumerate(words):
        is_negated = any(neg in words[max(0, i-2):i] for neg in negations)
        if word in symptom_map:
            symptom_name = symptom_map[word]
            if not is_negated:
                symptoms[symptom_name] = 1
    
    for phrase, symptom_name in symptom_map.items():
        if phrase in text:
            phrase_idx = text.find(phrase)
            context_start = max(0, phrase_idx - 15)
            context_end = min(len(text), phrase_idx + len(phrase) + 15)
            context = text[context_start:context_end]
            is_negated = any(neg in context.split() for neg in negations)
            if not is_negated:
                symptoms[symptom_name] = 1
    
    return [
        symptoms["Demam"], symptoms["Batuk"], symptoms["NyeriOtot"], symptoms["Mual"],
        symptoms["NyeriPerut"], symptoms["Pusing"], symptoms["Sesak"], symptoms["TensiTinggi"]
    ]


# =========================
# SCHEMA
# =========================

class PatientCreate(BaseModel):
    name: str
    age: int
    gender: str


class PatientResponse(BaseModel):
    id: int
    name: str
    age: int
    gender: str

    class Config:
        from_attributes = True


class SymptomInput(BaseModel):
    patient_id: int
    Demam: int = Field(..., ge=0, le=1)
    Batuk: int = Field(..., ge=0, le=1)
    NyeriOtot: int = Field(..., ge=0, le=1)
    Mual: int = Field(..., ge=0, le=1)
    NyeriPerut: int = Field(..., ge=0, le=1)
    Pusing: int = Field(..., ge=0, le=1)
    Sesak: int = Field(..., ge=0, le=1)
    TensiTinggi: int = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    predicted_disease: str
    confidence: float


# =========================
# LOGIN & REGISTER
# =========================

@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})


@app.post("/register")
async def register(request: Request, username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    # Cek apakah username sudah ada
    existing_user = db.query(models.User).filter(models.User.username == username).first()
    if existing_user:
        return templates.TemplateResponse("register.html", {
            "request": request,
            "error": "Username sudah digunakan"
        })
    
    # Hash password dan simpan ke database
    hashed_password = hash_password(password)
    new_user = models.User(username=username, password=hashed_password)
    db.add(new_user)
    db.commit()
    
    return RedirectResponse("/", status_code=status.HTTP_302_FOUND)


@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    # Cek apakah user ada di database
    user = db.query(models.User).filter(models.User.username == username).first()
    
    if user and verify_password(password, user.password):
        request.session["user"] = username
        return RedirectResponse("/dashboard", status_code=status.HTTP_302_FOUND)
    
    return templates.TemplateResponse("login.html", {
        "request": request,
        "error": "Username atau Password salah"
    })


# =========================
# ADD PATIENT
# =========================

@app.get("/ui/add-patient", response_class=HTMLResponse)
def ui_add_patient(request: Request):
    return templates.TemplateResponse("add_patient.html", {"request": request})


@app.post("/ui/add-patient")
def ui_create_patient(
    name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    db: Session = Depends(get_db)
):
    new_patient = models.Patient(name=name, age=age, gender=gender)
    db.add(new_patient)
    db.commit()
    return RedirectResponse(url="/dashboard", status_code=303)


# =========================
# SEARCH PATIENTS
# =========================

@app.get("/api/patients/search")
def search_patients(q: str, db: Session = Depends(get_db)):
    patients = db.query(models.Patient).filter(
        models.Patient.name.ilike(f"%{q}%")
    ).limit(10).all()
    return [patient_to_dict(p) for p in patients]


# =========================
# PREDICTION
# =========================

@app.get("/ui/predict", response_class=HTMLResponse)
def ui_predict(request: Request, db: Session = Depends(get_db)):
    patients = db.query(models.Patient).all()
    patients_list = [patient_to_dict(p) for p in patients]

    return templates.TemplateResponse("predict.html", {
        "request": request,
        "patients": patients_list
    })


@app.post("/ui/predict", response_class=HTMLResponse)
async def ui_process_predict(
    request: Request,
    db: Session = Depends(get_db)
):
    if model is None:
        patients = db.query(models.Patient).all()
        patients_list = [patient_to_dict(p) for p in patients]
        return templates.TemplateResponse("predict.html", {
            "request": request,
            "patients": patients_list,
            "error": "Model ML belum dimuat. Restart server."
        })
    
    try:
        form_data = await request.form()
        
        patient_id = form_data.get("patient_id")
        if not patient_id:
            patients = db.query(models.Patient).all()
            patients_list = [patient_to_dict(p) for p in patients]
            return templates.TemplateResponse("predict.html", {
                "request": request,
                "patients": patients_list,
                "error": "Pilih pasien terlebih dahulu!"
            })
        
        patient_id = int(patient_id)
        symptom_text = form_data.get("symptoms_text", "").strip()
        symptoms_array = parse_symptoms_from_text(symptom_text)
        
        patient = db.query(models.Patient).filter(models.Patient.id == patient_id).first()
        if not patient:
            patients = db.query(models.Patient).all()
            patients_list = [patient_to_dict(p) for p in patients]
            return templates.TemplateResponse("predict.html", {
                "request": request,
                "patients": patients_list,
                "error": "Pasien tidak ditemukan!"
            })

        if sum(symptoms_array) == 0:
            patients = db.query(models.Patient).all()
            patients_list = [patient_to_dict(p) for p in patients]
            return templates.TemplateResponse("predict.html", {
                "request": request,
                "patients": patients_list,
                "error": "Tidak ada gejala terdeteksi. Contoh: 'demam dan batuk'"
            })

        input_data = np.array([symptoms_array])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data).max()

        symptom_names = []
        symptom_mapping = {
            "Demam": "Demam", "Batuk": "Batuk", "NyeriOtot": "Nyeri Otot",
            "Mual": "Mual", "NyeriPerut": "Nyeri Perut", "Pusing": "Pusing",
            "Sesak": "Sesak Napas", "TensiTinggi": "Tensi Tinggi"
        }
        for i, val in enumerate(symptoms_array):
            if val == 1:
                symptom_names.append(list(symptom_mapping.values())[i])

        # Save visit
        new_visit = models.Visit(
            patient_id=patient_id,
            predicted_disease=prediction,
            confidence=float(probability) * 100
        )
        db.add(new_visit)
        db.commit()

        return templates.TemplateResponse("result.html", {
            "request": request,
            "prediction": prediction,
            "confidence": round(float(probability) * 100, 2),
            "patient": patient_to_dict(patient),
            "symptoms": symptom_names,
            "symptom_text": symptom_text
        })
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        patients = db.query(models.Patient).all()
        patients_list = [patient_to_dict(p) for p in patients]
        return templates.TemplateResponse("predict.html", {
            "request": request,
            "patients": patients_list,
            "error": f"Error: {str(e)}"
        })


# =========================
# VISITS
# =========================

@app.get("/ui/patients/{patient_id}/visits", response_class=HTMLResponse)
def ui_patient_visits(
    request: Request,
    patient_id: int,
    db: Session = Depends(get_db)
):
    patient = db.query(models.Patient).filter(models.Patient.id == patient_id).first()
    visits = db.query(models.Visit).filter(
        models.Visit.patient_id == patient_id
    ).order_by(models.Visit.created_at.desc()).all()

    return templates.TemplateResponse("visits.html", {
        "request": request,
        "visits": visits,
        "patient": patient,
        "patient_id": patient_id
    })


# =========================
# API ENDPOINTS
# =========================

@app.post("/patients", response_model=PatientResponse)
def create_patient(data: PatientCreate, db: Session = Depends(get_db)):
    new_patient = models.Patient(name=data.name, age=data.age, gender=data.gender)
    db.add(new_patient)
    db.commit()
    db.refresh(new_patient)
    return new_patient


@app.get("/patients", response_model=list[PatientResponse])
def get_all_patients(db: Session = Depends(get_db)):
    return db.query(models.Patient).all()


@app.put("/patients/{patient_id}")
def update_patient(patient_id: int, data: PatientCreate, db: Session = Depends(get_db)):
    patient = db.query(models.Patient).filter(models.Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    patient.name = data.name
    patient.age = data.age
    patient.gender = data.gender
    db.commit()
    return {"message": "Patient updated successfully"}


@app.delete("/patients/{patient_id}")
def delete_patient(patient_id: int, db: Session = Depends(get_db)):
    patient = db.query(models.Patient).filter(models.Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    db.delete(patient)
    db.commit()
    return {"message": "Patient deleted successfully"}


@app.post("/predict", response_model=PredictionResponse)
def predict_disease(data: SymptomInput, db: Session = Depends(get_db)):
    patient = db.query(models.Patient).filter(models.Patient.id == data.patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    input_data = np.array([[ 
        data.Demam, data.Batuk, data.NyeriOtot, data.Mual,
        data.NyeriPerut, data.Pusing, data.Sesak, data.TensiTinggi
    ]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data).max()

    new_visit = models.Visit(
        patient_id=data.patient_id,
        predicted_disease=prediction,
        confidence=float(probability) * 100
    )

    db.add(new_visit)
    db.commit()

    return PredictionResponse(
        predicted_disease=prediction,
        confidence=round(float(probability) * 100, 2)
    )


@app.get("/patients/{patient_id}/visits")
def get_patient_visits(patient_id: int, db: Session = Depends(get_db)):
    visits = db.query(models.Visit).filter(models.Visit.patient_id == patient_id).all()
    return visits


# =========================
# DASHBOARD
# =========================

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(get_db)):
    if "user" not in request.session:
        return RedirectResponse("/", status_code=status.HTTP_302_FOUND)
    
    try:
        total_patients = db.query(models.Patient).count()
        total_visits = db.query(models.Visit).count()
    except Exception as e:
        print(f"Error: {e}")
        total_patients = 0
        total_visits = 0
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "user": request.session["user"],
        "total_patients": total_patients,
        "total_visits": total_visits
    })


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/", status_code=status.HTTP_302_FOUND)
