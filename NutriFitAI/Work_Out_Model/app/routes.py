from flask import Blueprint, render_template, request, jsonify, render_template_string, session
import os
import re
import joblib
import pandas as pd
try:
    from .utils import generate_workout_plan, swap_alternatives, DEFAULT_DURATION_MIN
except ImportError:
    # If relative import fails, try absolute import
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils import generate_workout_plan, swap_alternatives, DEFAULT_DURATION_MIN

bp = Blueprint("main", __name__)

_DATASET_CACHE = None
_MODEL = None
_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workoutdata_with_estimated_met.csv")
_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workout_model.pkl")


ALIAS_MAP = {
    "title": "Exercise_Name",
    "exercisename": "Exercise_Name",
    "name": "Exercise_Name",
    "bodypart": "Primary_Muscle",
    "primarymuscle": "Primary_Muscle",
    "musclegroup": "Primary_Muscle",
    "equipment": "Equipment",
    "level": "Level",
    "difficulty": "Difficulty",
    "difficultylevel": "Difficulty",
    "desc": "Instructions",
    "description": "Instructions",
    "instructions": "Instructions",
    "type": "Type",
    "category": "Type",
    "mechanics": "Mechanics",
    "movementtype": "Mechanics",
    "caloriesburned": "MET",
    "burnedcalories": "MET",
    "burned_calories": "MET",
    "calories_burned": "MET",
    "met": "MET",
    "mets": "MET",
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in list(df.columns):
        key = _norm(col)
        if key in ALIAS_MAP:
            canonical = ALIAS_MAP[key]
            if canonical not in df.columns:
                rename_map[col] = canonical
    if rename_map:
        df = df.rename(columns=rename_map)
    # Mirror Level to Difficulty if Difficulty missing
    if "Difficulty" not in df.columns and "Level" in df.columns:
        df["Difficulty"] = df["Level"].astype(str)
    # Ensure Mechanics exists for downstream sorting
    if "Mechanics" not in df.columns:
        df["Mechanics"] = "compound"
    # Map Desc to Instructions if Instructions doesn't exist
    if "Instructions" not in df.columns and "Desc" in df.columns:
        df["Instructions"] = df["Desc"].fillna("")
    elif "Instructions" not in df.columns:
        df["Instructions"] = ""
    return df


def load_dataset():
    global _DATASET_CACHE
    if _DATASET_CACHE is None:
        if not os.path.exists(_DATA_PATH):
            raise FileNotFoundError(f"Dataset not found at {_DATA_PATH}")
        df = pd.read_csv(_DATA_PATH)
        df = normalize_headers(df)
        required = [
            "Exercise_Name",
            "Primary_Muscle",
            "Equipment",
            "Difficulty",
            "Instructions",
            "MET",
            "Type",
            "Mechanics",
            "Level",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            # Relax requirement: only strictly needed for generation
            minimal = ["Exercise_Name", "Primary_Muscle", "Equipment", "Difficulty", "MET"]
            still_missing = [c for c in minimal if c not in df.columns]
            if still_missing:
                raise ValueError(f"Missing columns in dataset even after normalization: {still_missing}")
        _DATASET_CACHE = df
    return _DATASET_CACHE


def load_model():
    global _MODEL
    if _MODEL is None and os.path.exists(_MODEL_PATH):
        _MODEL = joblib.load(_MODEL_PATH)
    return _MODEL


@bp.get("/")
def index():
    # Get user data from session
    user_data = session.get('user_data', {})
    print(f"🔍 Workout route accessed. Session data: {user_data}")
    print(f"🔍 Session keys: {list(session.keys())}")
    
    # Always show the workout form, even if no session data
    # The form will have default values if no session data is available
    # Use render_template_string to ensure we get the correct template
    import os
    template_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "templates", "index.html")
    print(f"🔍 Template path: {template_path}")
    
    with open(template_path, 'r', encoding='utf-8') as f:
        template_content = f.read()
    
    # Replace template variables manually
    if user_data:
        template_content = template_content.replace('{{ user_data.age if user_data else \'\' }}', str(user_data.get('age', '')))
        template_content = template_content.replace('{% if user_data and user_data.gender == 0 %}selected{% endif %}', 'selected' if user_data.get('gender') == 0 else '')
        template_content = template_content.replace('{% if user_data and user_data.gender == 1 %}selected{% endif %}', 'selected' if user_data.get('gender') == 1 else '')
        template_content = template_content.replace('{% if not user_data %}selected{% endif %}', 'selected' if not user_data else '')
    
    return template_content


@bp.post("/generate_plan")
def generate_plan():
    payload = request.get_json() or request.form
    
    # Get user data from session as fallback
    session_data = session.get('user_data', {})
    
    # Map goal from diet model to workout model
    goal_mapping = {0: "Fat Loss", 1: "Muscle Gain", 2: "Maintain"}
    activity_mapping = {0: "Beginner", 1: "Beginner", 2: "Intermediate", 3: "Intermediate", 4: "Intermediate"}
    gender_mapping = {0: "Male", 1: "Female"}
    
    user = {
        "Age": int(payload.get("Age", session_data.get('age', 30))),
        "Gender": payload.get("Gender", gender_mapping.get(session_data.get('gender', 0), "Other")),
        "Weight": float(payload.get("Weight", session_data.get('weight', 70))),
        "Height": float(payload.get("Height", session_data.get('height', 170))),
        "Goal": payload.get("Goal", goal_mapping.get(session_data.get('goal', 2), "Maintain")),
        "Experience Level": payload.get("Experience Level", activity_mapping.get(session_data.get('activity', 2), "Beginner")),
        "Workout Preference": payload.get("Workout Preference", "Gym"),
    }

    df = load_dataset()
    load_model()

    plan, total_calories, chart_data = generate_workout_plan(df, user, duration_min=DEFAULT_DURATION_MIN)

    html = render_template("plan.html", plan=plan, total_calories=total_calories, duration=DEFAULT_DURATION_MIN, preference=user["Workout Preference"])

    return jsonify({
        "html": html,
        "chartData": chart_data,
    })


@bp.post("/swap_exercise")
def swap_exercise():
    payload = request.get_json(force=True)
    current = payload.get("current", {})
    preference = payload.get("preference", "Gym")
    df = load_dataset()
    alts = swap_alternatives(df, current=current, preference=preference)
    return jsonify({"alternatives": alts})


from flask import render_template, request, current_app, url_for

# === Video playlists by category ===
CORE_PLAYLIST = "https://www.youtube.com/embed/videoseries?list=PL2ov72VWpiOpnM89hVl1IChpWHnf1Rvnm"
UPPER_PLAYLIST = "https://www.youtube.com/embed/videoseries?list=PLvf_LH4Nzg12rnMgCf5ZX96gn-15Pz9rf"
LOWER_PLAYLIST = "https://www.youtube.com/embed/videoseries?list=PL2ov72VWpiOq5qkkM9kP8pBONIC7gV6--"


def get_video_for_muscle(primary_muscle: str) -> str:
    """
    Map your exercise's Primary_Muscle / BodyPart to a YouTube playlist.
    """
    pm = (primary_muscle or "").strip().lower()

    core_muscles = {"abdominals", "lower back"}
    lower_muscles = {
        "quadriceps", "hamstrings", "glutes", "calves",
        "adductors", "abductors"
    }
    upper_muscles = {
        "chest", "lats", "middle back", "traps",
        "shoulders", "biceps", "triceps", "forearms", "neck"
    }

    if pm in core_muscles:
        return CORE_PLAYLIST
    if pm in lower_muscles:
        return LOWER_PLAYLIST
    if pm in upper_muscles:
        return UPPER_PLAYLIST

    # default if unknown
    return CORE_PLAYLIST
