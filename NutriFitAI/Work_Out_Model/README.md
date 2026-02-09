# AI Workout Planner

## Setup

1. Create venv (Windows PowerShell):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Train model (ensure `workoutdata_with_estimated_met.csv` is in project root):
   ```powershell
   python train_model.py
   ```

3. Run app:
   ```powershell
   python wsgi.py
   ```

Open `http://localhost:5000`.
