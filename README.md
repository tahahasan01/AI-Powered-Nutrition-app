# NutriFit Pakistan — AI-Powered Nutrition & Fitness Platform

Production-ready AI platform for personalized diet plans, workout generation, and progress tracking. Built for Pakistani users with local cuisine, TDEE-based nutrition, and MySQL-backed user accounts.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Production Deployment](#production-deployment)
- [API Reference](#api-reference)
- [Database Schema](#database-schema)
- [Testing](#testing)
- [Security](#security)
- [Troubleshooting](#troubleshooting)
- [License & Contributing](#license--contributing)

---

## Overview

NutriFit Pakistan is a unified web application that provides:

- **Personalized 7-day meal plans** (4 meals/day) using ML models and Pakistani/global food datasets  
- **6-day workout plans** with equipment (home/gym), level, and goal-based filtering  
- **User accounts** (signup/login) with MySQL and session-based auth  
- **Weight progress tracking** (start + 6 weekly check-ins) with plateau detection and charts  
- **Single entry point**: one Flask app serves diet, workout, and progress UIs

---

## Features

### Diet planning (Diet_Plan_Model)

| Feature | Description |
|--------|-------------|
| **TDEE-based targets** | Mifflin–St Jeor equation; goal-specific calorie and macro targets |
| **7-day meal plans** | Breakfast, lunch, dinner, snack; MLP + neural ranker for food selection |
| **Meal swapping** | Alternative meals with similar nutrition (goal-aware) |
| **Pakistani cuisine** | Local ingredients and meal types in datasets |
| **Charts** | Macro pie chart, weekly progress projection (Chart.js) |

### Workout planning (Work_Out_Model)

| Feature | Description |
|--------|-------------|
| **6-day splits** | Pre-built splits with exercise recommendations from trained model |
| **Equipment** | Home vs gym; MET-based calorie estimates |
| **Level** | Beginner / intermediate adaptations |
| **Exercise swap** | Alternatives by muscle group / equipment |
| **PDF-style plan** | Plan view with exercises and instructions |

### Progress tracking (Progress_Tracking)

| Feature | Description |
|--------|-------------|
| **Weight log** | Start weight + 6 weekly check-ins |
| **Goal mode** | Loss / gain / maintain; used for projections |
| **Plateau detection** | Simple heuristic for stalled progress |
| **Charts** | Weight-over-time and goal visualization (React/JSX component + HTML) |

### User & auth

- Signup (name, email, phone, password) and login  
- Session-based auth with Flask-Login  
- Password hashing (Werkzeug)  
- Protected routes: diet form, result, workout, progress  

---

## Architecture

- **Single Flask app**: `Diet_Plan_Model/app.py` is the main WSGI application.
- **Workout as blueprint**: Work_Out_Model is loaded as a Flask blueprint under `/workout`.
- **Progress**: Static HTML/JS served from `Progress_Tracking/` at `/progress` and `/progress-tracking/<path>`.
- **Database**: MySQL (SQLAlchemy) for users and weight logs.
- **ML/MLP**: Scikit-learn and PyTorch-style models in Diet_Plan_Model; Work_Out_Model uses a pre-trained pickle model and CSV dataset.

---

## Project Structure

```
.
├── README.md                    # This file
├── .gitignore
├── Diet_Plan_Model/             # Main Flask app + diet logic
│   ├── app.py                   # Entry point — run this
│   ├── diet_model.py            # Diet ML logic & ranker
│   ├── meal_model.py            # Meal construction helpers
│   ├── requirements.txt
│   ├── cleaned_foods_dataset.csv
│   ├── cleaned_snacks_dataset.csv
│   ├── diet_model.pkl           # Pre-trained diet model (if used)
│   ├── models/                  # Neural nets (food_net, profile_net, scalers)
│   ├── templates/               # Jinja2: index, result, NutriFit_HomePage
│   ├── static/
│   └── test_api.py
├── Work_Out_Model/              # Workout blueprint + data
│   ├── app/
│   │   ├── __init__.py          # create_app (blueprint registration)
│   │   ├── routes.py            # /workout, generate_plan, swap_exercise
│   │   ├── utils.py             # Plan generation, MET, swap logic
│   │   ├── templates/
│   │   └── static/
│   ├── workout_model.pkl
│   ├── workoutdata_with_estimated_met.csv
│   ├── train_model.py           # Retrain workout model
│   ├── wsgi.py                  # Standalone run for workout-only (optional)
│   └── requirements.txt
└── Progress_Tracking/           # Front-end assets for progress
    ├── progress.html
    └── ProgressPage.jsx
```

---

## Prerequisites

- **Python** 3.10+ (3.12 recommended)
- **MySQL** 8.x (or 5.7+) with a dedicated database and user
- **pip** and a virtual environment (venv recommended)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/tahahasan01/AI-Powered-Nutrition-app.git
cd AI-Powered-Nutrition-app
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

**Option A — one command (recommended):**

```bash
pip install -r requirements.txt
```

**Option B — per-module:**

```bash
pip install -r Diet_Plan_Model/requirements.txt
pip install -r Work_Out_Model/requirements.txt
pip install Flask-Login Flask-SQLAlchemy PyMySQL
```

Optional (if Diet_Plan_Model uses PyTorch models in `models/`):

```bash
pip install torch
```

### 4. Ensure data and models are present

- `Diet_Plan_Model/`: `cleaned_foods_dataset.csv`, `cleaned_snacks_dataset.csv`, `models/` (and any `.pkl` / `.pt` used by `diet_model.py`)
- `Work_Out_Model/`: `workoutdata_with_estimated_met.csv`, `workout_model.pkl`

---

## Configuration

### Environment variables (production)

| Variable | Description | Default (dev) |
|----------|-------------|----------------|
| `NUTRIFIT_SECRET` | Flask `secret_key` for sessions | `your-secret-key-here` |
| `SQLALCHEMY_DATABASE_URI` | MySQL connection string | See below |

### Database (MySQL)

1. Create database and user:

```sql
CREATE DATABASE nutrifit CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'nutrifit_user'@'localhost' IDENTIFIED BY 'your_secure_password';
GRANT ALL PRIVILEGES ON nutrifit.* TO 'nutrifit_user'@'localhost';
FLUSH PRIVILEGES;
```

2. Set the connection string in `Diet_Plan_Model/app.py` or via environment:

- In code (replace with your credentials):

```python
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://nutrifit_user:your_secure_password@localhost/nutrifit'
```

- Or export before run:

```bash
export SQLALCHEMY_DATABASE_URI='mysql+pymysql://user:pass@localhost/nutrifit'
```

Tables `user` and `user_weight_logs` are created automatically on first run when `db.create_all()` runs (see `app.py` `if __name__ == '__main__'`).

---

## Running the Application

From the **repository root**:

```bash
python Diet_Plan_Model/app.py
```

Then open: **http://localhost:5000**

- **Home**: Landing / login / signup  
- **Diet**: Log in → diet form → generate plan → result page with “Generate Workout Plan”  
- **Workout**: From result page or directly: **http://localhost:5000/workout/**  
- **Progress**: **http://localhost:5000/progress** (requires login)

Default run: `host='0.0.0.0'`, `port=5000`, `debug=True`. For production, use a WSGI server and turn off debug (see [Production Deployment](#production-deployment)).

---

## Production Deployment

### WSGI server (Gunicorn example)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 "Diet_Plan_Model.app:app"
```

Adjust `-w` (workers) and bind address as needed.

### Checklist

- [ ] Set `NUTRIFIT_SECRET` to a strong random value.
- [ ] Use a dedicated MySQL user and strong password; set `SQLALCHEMY_DATABASE_URI` via env.
- [ ] Run with `debug=False`.
- [ ] Put the app behind a reverse proxy (e.g. Nginx) and use HTTPS.
- [ ] Restrict DB access (firewall, no public bind).
- [ ] Serve static assets via Nginx/CDN if desired; ensure `/workout/static/` and `/progress-tracking/` still work.
- [ ] Optional: run `Work_Out_Model/train_model.py` to regenerate `workout_model.pkl` when dataset changes.

---

## API Reference

### Auth

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Home / login page |
| POST | `/login` | Login (email, password) |
| POST | `/signup` | Register (full_name, email, phone, password) |
| GET | `/logout` | Logout |

### Diet

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/diet-form` | Diet form (login required) |
| POST | `/generate-plan` | Generate meal plan (form body) |
| POST | `/predict_diet` | JSON API for diet prediction |
| POST | `/swap_meal` | Get meal alternatives |
| POST | `/get_meal_details` | Meal details by name/quantity |

### Workout

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/workout/` | Workout form |
| POST | `/workout/generate_plan` | Generate workout plan |
| POST | `/workout/swap_exercise` | Exercise alternatives |

### Progress

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/progress` | Progress page (HTML) |
| GET | `/progress-tracking/<path>` | Static assets for progress |
| GET/POST | `/api/weight-log` | Get or submit weight log entries |

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check (no auth) |

---

## Database Schema

- **user**: `id`, `full_name`, `email`, `phone`, `password_hash`, `created_at`
- **user_weight_logs**: `id`, `user_id`, `label`, `week_index`, `weight_kg`, `goal_mode` (and any other columns defined in `UserWeightLog` in `app.py`)

Schema is created by SQLAlchemy from the models in `Diet_Plan_Model/app.py`. For migrations in production, consider Flask-Migrate (Alembic).

---

## Testing

- **Diet API**: `cd Diet_Plan_Model && python test_api.py` (if script exists and targets local `app`).
- **Manual**: Use `/health`, then diet form → generate plan → workout → progress.

---

## Security

- Passwords hashed with Werkzeug (no plain text).
- Use HTTPS and secure cookies in production.
- Set a strong `NUTRIFIT_SECRET` and keep DB credentials out of version control (env vars or secrets manager).
- Validate and sanitize all inputs; the app uses server-side validation for forms and API.

---

## Troubleshooting

| Issue | What to check |
|-------|----------------|
| “No module named 'diet_model'” | Run from repo root: `python Diet_Plan_Model/app.py`; ensure `Diet_Plan_Model` is on Python path. |
| “Workout blueprint” not registered | Ensure `Work_Out_Model/app/routes.py` and `Work_Out_Model/app/utils.py` exist; check console for import errors. |
| DB connection error | Verify MySQL is running, database and user exist, and `SQLALCHEMY_DATABASE_URI` is correct. |
| Charts or progress not loading | Confirm `/progress-tracking/` and `/workout/static/` are reachable; check browser console. |
| Meal plan empty / errors | Confirm CSVs and model files exist in `Diet_Plan_Model/` and `diet_model.load_models()` succeeds. |

---

## License & Contributing

This project is open source. You may modify and use it according to your needs.

To contribute:

1. Fork the repository.
2. Create a feature branch, make changes, and test (run app + optional tests).
3. Submit a pull request with a clear description of the change.

---

**NutriFit Pakistan** — AI-driven nutrition and fitness planning for healthier living.
