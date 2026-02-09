# app.py
import os
import sys
import json
import traceback
import random
import re
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask import (
    Flask, request, jsonify, render_template, redirect, url_for,
    flash, session, send_from_directory
)
from flask_cors import CORS

# ML/diet model (your module)
from diet_model import diet_model

# Authentication / DB
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, login_user, logout_user, login_required,
    current_user, UserMixin
)
from werkzeug.security import generate_password_hash, check_password_hash

# Add the workout model path to sys.path for imports (keeps your original intent)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Work_Out_Model'))

# ---- Flask app ----
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = os.environ.get('NUTRIFIT_SECRET', 'your-secret-key-here')
CORS(app)

# ---- DATABASE CONFIGURATION (MySQL) ----
# Replace with your DB credentials (already provided)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:123456@localhost/nutrifit'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'index'      # redirect to home page (index)
login_manager.login_message = "Please log in to continue."

# ---- USER MODEL ----
# ---- USER MODEL ----
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100))
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(20))
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

    def set_password(self, pw):
        self.password_hash = generate_password_hash(pw)

    def check_password(self, pw):
        return check_password_hash(self.password_hash, pw)


# ✅ NEW: Per-user weight progress table
class UserWeightLog(db.Model):
    __tablename__ = 'user_weight_logs'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer,
                        db.ForeignKey('user.id'),
                        nullable=False,
                        index=True)
    # "start", "week1", "week2", ...
    label = db.Column(db.String(16), nullable=False)
    # 0 = start, 1 = week1, ..., 6 = week6
    week_index = db.Column(db.Integer, nullable=False)

    # weight in kg
    weight_kg = db.Column(db.Float, nullable=False)

    # optional: what the user is trying to do at that time
    # "loss" | "gain" | "maintain"
    goal_mode = db.Column(db.String(16), nullable=True)

    created_at = db.Column(db.DateTime, server_default=db.func.now())
    updated_at = db.Column(
        db.DateTime,
        server_default=db.func.now(),
        onupdate=db.func.now()
    )

    __table_args__ = (
        db.UniqueConstraint('user_id', 'week_index', name='uq_user_week'),
    )


@login_manager.user_loader
def load_user(uid):
    try:
        return User.query.get(int(uid))
    except Exception:
        return None

# ---- Initialize diet model data ----
try:
    diet_model.load_data()
    print(f"✅ Diet model data loaded: {len(diet_model.combined_data) if hasattr(diet_model, 'combined_data') else 'unknown items'}")
except Exception as e:
    print("⚠️ Warning: diet_model.load_data() failed:", e)
    traceback.print_exc()
# ---- Load trained DL models (if available) ----
try:
    loaded = diet_model.load_models()
    print("DL models loaded:", loaded)
except Exception as e:
    print("⚠️ Warning: diet_model.load_models() failed:", e)

# ---- Import and register workout blueprint (optional) ----
try:
    # workout app folder path (adjusted to match earlier structure)
    workout_app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Work_Out_Model', 'app'))
    sys.path.insert(0, workout_app_path)

    import importlib.util
    routes_py = os.path.join(workout_app_path, "routes.py")
    if os.path.exists(routes_py):
        spec = importlib.util.spec_from_file_location("workout_routes", routes_py)
        routes_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(routes_module)

        from flask import Blueprint
        workout_template_folder = os.path.abspath(os.path.join(workout_app_path, "templates"))
        workout_bp = Blueprint('workout', __name__, template_folder=workout_template_folder)

        # Bind functions from imported module to blueprint routes if available
        if hasattr(routes_module, 'index'):
            workout_bp.add_url_rule('/', 'index', routes_module.index, methods=['GET'])
        if hasattr(routes_module, 'generate_plan'):
            workout_bp.add_url_rule('/generate_plan', 'generate_plan', routes_module.generate_plan, methods=['POST'])
        if hasattr(routes_module, 'swap_exercise'):
            workout_bp.add_url_rule('/swap_exercise', 'swap_exercise', routes_module.swap_exercise, methods=['POST'])

        app.register_blueprint(workout_bp, url_prefix='/workout')

        @app.route('/workout/static/<path:filename>')
        def workout_static(filename):
            from flask import send_from_directory
            workout_static_path = os.path.join(os.path.dirname(__file__), '..', 'Work_Out_Model', 'app', 'static')
            return send_from_directory(workout_static_path, filename)

        print("✅ Workout blueprint registered successfully")
    else:
        print("⚠️ Workout routes.py not found; skipping workout blueprint registration")
except Exception as e:
    print(f"⚠️ Warning: Could not import workout blueprint: {e}")
    traceback.print_exc()
    # continue without workout blueprint

# -------------------------
# Utility helpers
# -------------------------
RICE_PAT = re.compile(r"rice", re.I)
GRAVY_PAT = re.compile(r"(curry|karahi|nihari|qorma|handi|daal|dal|haleem|sabzi|korma)", re.I)
JUNK_PAT = re.compile(r"(pizza|burger|subway|domino|kfc|mcdonald|wrap|fries|pasta|nugget|sandwich)", re.I)
DESI_PAT = re.compile(r"(biryani|dal|daal|saag|korma|haleem|nihari|chapli|kabab|paratha|rajma|chole|aloo|gobi|paneer|pulao|roti|sabzi|masoor|bhindi|kadhi|tandoori|karahi|handi|qorma|chana)", re.I)

def strip_gram_suffix(name: str) -> str:
    return re.sub(r'\s+—\s*\d+\s*g$', '', str(name), flags=re.I).strip()

def standardize_serving(name: str):
    base = strip_gram_suffix(name)
    had_plate = re.search(r'\bplate\b', base, re.I) is not None
    had_bowl = re.search(r'\bbowl\b', base, re.I) is not None
    base = re.sub(r'\b(bowl|plate)\b', '', base, flags=re.I)
    base = re.sub(r'\s+', ' ', base).strip()
    if had_plate:
        grams = 300
    elif had_bowl:
        grams = 250
    else:
        if RICE_PAT.search(base):
            grams = 300
        elif GRAVY_PAT.search(base):
            grams = 250
        else:
            grams = 150
    display = f"{base} — {grams} g"
    return base, grams, display

SNACK_CHOICES = [
    {"name": "Apple", "cal": 80, "p":0, "c":21, "f":0},
    {"name": "Banana", "cal": 95, "p":1, "c":23, "f":0},
    {"name": "Dates (3 pcs)", "cal":70, "p":0, "c":19, "f":0},
    {"name": "Almonds (15 g)", "cal":90, "p":3, "c":3, "f":7},
    {"name": "Yogurt (100 g)", "cal":80, "p":5, "c":6, "f":3}
]
def build_daily_snack():
    s = random.choice(SNACK_CHOICES)
    base, grams, display = standardize_serving(s["name"])
    return {
        "name": display,
        "quantity": grams,
        "calories": s["cal"],
        "protein": s["p"],
        "carbs": s["c"],
        "fat": s["f"]
    }

def find_desi_alternative(target_cal, allergies=None, exclude_names=None):
    df = getattr(diet_model, "combined_data", None)
    if df is None or len(df) == 0:
        return None
    df = df[~df['Food Name'].str.contains(JUNK_PAT, na=False)]
    desi = df[df['Food Name'].str.contains(DESI_PAT, na=False)]
    if not desi.empty:
        df = desi
    if allergies:
        al = [a.lower() for a in allergies]
        mask = df['Food Name'].str.lower().apply(lambda x: not any(a in x for a in al))
        df = df[mask]
    if exclude_names:
        ex = [strip_gram_suffix(n).lower() for n in exclude_names]
        df = df[~df['Food Name'].str.lower().isin(ex)]
    if df.empty:
        return None
    # score by closeness to calories (use 100g baseline from dataset)
    df = df.copy()
    df['diff'] = (df['Calories'] - target_cal).abs()
    row = df.sort_values('diff').iloc[0]
    base, grams, display = standardize_serving(row['Food Name'])
    return {
        "name": display,
        "quantity": grams,
        "calories": int(round(row['Calories'] * grams / 100.0)),
        "protein": round(row.get('Protein (g)', row.get('Protein', 0)) * grams / 100.0, 1),
        "carbs": round(row.get('Carbohydrates (g)', row.get('Carbohydrates', 0)) * grams / 100.0, 1),
        "fat": round(row.get('Fat (g)', row.get('Fat', 0)) * grams / 100.0, 1)
    }

# -------------------------
# Routes
# -------------------------
@app.route('/')
def index():
    return render_template('NutriFit_HomePage.html')

# Authentication endpoints
from flask_login import login_required, current_user, login_user, logout_user
from werkzeug.security import generate_password_hash, check_password_hash

# ---- AUTH ROUTES ----

# --------- SIGNUP ----------
# SIGNUP
@app.route('/signup', methods=['POST'])
def signup():
    try:
        data = request.get_json() or {}
        name = data.get('name', '').strip()
        email = data.get('email', '').strip().lower()
        phone = data.get('phone', '').strip()
        password = data.get('password', '').strip()

        if not all([name, email, password]):
            return jsonify({'success': False, 'message': 'All fields are required'}), 400

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return jsonify({'success': False, 'message': 'Email already registered'}), 400

        hashed_pw = generate_password_hash(password)
        new_user = User(full_name=name, email=email, phone=phone, password_hash=hashed_pw)
        db.session.add(new_user)
        db.session.commit()

        return jsonify({'success': True, 'message': 'Signup successful! Please log in.'}), 200
    except Exception as e:
        db.session.rollback()
        print("Signup error:", e)
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'Server error during signup'}), 500


# LOGIN
@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json() or {}
        email = data.get('email', '').strip().lower()
        password = data.get('password', '').strip()

        if not email or not password:
            return jsonify({'success': False, 'message': 'Email and password are required'}), 400

        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({'success': False, 'message': 'No account found with this email'}), 401

        if not check_password_hash(user.password_hash, password):
            return jsonify({'success': False, 'message': 'Incorrect password'}), 401

        login_user(user)
        session['user_id'] = user.id
        session.permanent = True

        return jsonify({'success': True, 'message': 'Login successful', 'redirect': '/diet-form'}), 200
    except Exception as e:
        print("Login error:", e)
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'Server error during login'}), 500


@app.route('/logout', methods=['POST'])
@login_required
def logout():
    try:
        logout_user()
        session.clear()
        return jsonify({'message': 'Logout successful'}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': 'Logout failed'}), 500

# Protected pages
@app.route('/diet-form')
@login_required
def diet_form():
    return render_template('index.html')

@app.route('/results')
@login_required
def show_results():
    # Check if user_data exists in session (from form submission)
    user_data = session.get('user_data', {'age':25,'gender':0,'weight':70,'height':175,'goal':0,'activity':2})
    
    # If no meal plan data in session, show empty state
    if 'meal_plan_data' not in session:
        return render_template('result.html',
                               weekly_plan=[],
                               targets={'calories':2000,'protein':150,'carbs':200,'fat':80},
                               tdee=2000,
                               totals={'calories':0,'protein':0,'carbs':0,'fat':0},
                               user_data=user_data)
    
    # Get meal plan data from session
    plan_data = session['meal_plan_data']
    return render_template('result.html',
                           weekly_plan=plan_data['weekly_plan'],
                           targets=plan_data.get('targets', {}),
                           tdee=plan_data.get('tdee', 0),
                           totals=plan_data.get('totals', {'calories':0,'protein':0,'carbs':0,'fat':0}),
                           user_data=user_data)

@app.route('/generate-plan', methods=['POST'])
@login_required
def generate_plan():
    try:
        age = int(request.form.get('age'))
        gender = int(request.form.get('gender'))
        weight = float(request.form.get('weight'))
        height = float(request.form.get('height'))
        goal = int(request.form.get('goal'))
        activity = int(request.form.get('activity'))

        # Validate inputs
        if not (1 <= age <= 120):
            flash('Please enter a valid age', 'error'); return redirect(url_for('diet_form'))
        if not (30 <= weight <= 300):
            flash('Please enter a valid weight (30-300 kg)', 'error'); return redirect(url_for('diet_form'))
        if not (100 <= height <= 250):
            flash('Please enter a valid height (100-250 cm)', 'error'); return redirect(url_for('diet_form'))

        session['user_data'] = {'age':age,'gender':gender,'weight':weight,'height':height,'goal':goal,'activity':activity}

        plan_data = diet_model.generate_meal_plan(age, gender, weight, height, goal, activity)
        if not plan_data or not plan_data.get('weekly_plan'):
            flash('Unable to generate meal plan. Please try again.', 'error'); return redirect(url_for('diet_form'))

        # Post-process: standardize servings and ensure one snack per day, desi fallback
        new_week = []
        allergies = request.form.getlist('allergies') or []

        for day_data in plan_data['weekly_plan']:
            meals = []
            for meal in day_data['meals']:
                base, grams, display = standardize_serving(meal.get('name',''))
                m = dict(meal)
                m['name'] = display
                m['quantity'] = grams
                # if meal seems like junk, try to find alternative
                if JUNK_PAT.search(base):
                    alt = find_desi_alternative(m.get('calories', 400), allergies, exclude_names=[m['name']])
                    if alt:
                        m = alt
                meals.append(m)

            # guarantee at least 3 mains
            while len(meals) < 3:
                alt = find_desi_alternative(400, allergies, exclude_names=[mm['name'] for mm in meals])
                if not alt:
                    alt = {'name': 'Plain Roti — 100 g', 'quantity': 100, 'calories': 250, 'protein': 8, 'carbs': 40, 'fat': 5}
                meals.append(alt)

            # always append one healthy snack
            meals.append(build_daily_snack())
            new_week.append({'day': day_data.get('day'), 'day_number': day_data.get('day_number'), 'meals': meals})

        plan_data['weekly_plan'] = new_week

        # totals
        total_calories = total_protein = total_carbs = total_fat = 0
        for d in plan_data['weekly_plan']:
            for m in d['meals']:
                total_calories += m.get('calories', 0)
                total_protein += m.get('protein', 0)
                total_carbs += m.get('carbs', 0)
                total_fat += m.get('fat', 0)
        
        # 🛡️ Ensure every meal has all keys required by result.html
        for day_data in plan_data['weekly_plan']:
            for meal in day_data['meals']:
                meal.setdefault('sugar', 0)
                meal.setdefault('fiber', 0)
                meal.setdefault('protein', 0)
                meal.setdefault('carbs', 0)
                meal.setdefault('fat', 0)
                meal.setdefault('calories', 0)

        # Store meal plan data in session
        session['meal_plan_data'] = {
            'weekly_plan': plan_data['weekly_plan'],
            'targets': plan_data.get('targets', {}),
            'tdee': plan_data.get('tdee', 0),
            'totals': {'calories': total_calories, 'protein': total_protein, 'carbs': total_carbs, 'fat': total_fat}
        }
        
        # Redirect to results page
        return redirect(url_for('show_results'))
    except Exception as e:
        traceback.print_exc()
        print("DEBUG FORM DATA:", request.form)

        flash('An error occurred while generating your meal plan. Please try again.', 'error')
        return redirect(url_for('diet_form'))

@app.route('/predict_diet', methods=['POST'])
@login_required
def predict_diet():
    try:
        data = request.get_json()
        required_fields = ['age', 'gender', 'weight', 'height', 'goal', 'activity_level']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        age = int(data['age']); gender = int(data['gender']); weight = float(data['weight'])
        height = float(data['height']); goal = int(data['goal']); activity_level = int(data['activity_level'])

        if not (1 <= age <= 120): return jsonify({'error':'Age must be between 1 and 120'}),400
        if not (30 <= weight <= 300): return jsonify({'error':'Weight must be between 30 and 300 kg'}),400
        if not (100 <= height <= 250): return jsonify({'error':'Height must be between 100 and 250 cm'}),400
        if goal not in [0,1,2]: return jsonify({'error':'Goal must be 0,1 or 2'}),400
        if activity_level not in [0,1,2,3,4]: return jsonify({'error':'Activity level must be between 0 and 4'}),400

        plan_data = diet_model.generate_meal_plan(age, gender, weight, height, goal, activity_level)
        if not plan_data or not plan_data.get('weekly_plan'):
            return jsonify({'error':'Unable to generate meal plan'}),500

        weekly_plans = []
        weekly_calories = weekly_protein = weekly_carbs = weekly_fat = 0

        for day_data in plan_data['weekly_plan']:
            daily_meals = []; daily_cal = daily_p = daily_c = daily_f = 0
            for i, meal in enumerate(day_data['meals']):
                meal_type = ['Breakfast','Lunch','Dinner','Snack'][i] if i < 4 else 'Snack'
                daily_meals.append({
                    'meal_type': meal_type,
                    'food_name': meal.get('name'),
                    'quantity_grams': meal.get('quantity'),
                    'calories': meal.get('calories'),
                    'protein_g': meal.get('protein'),
                    'carbohydrates_g': meal.get('carbs'),
                    'fats_g': meal.get('fat'),
                    'sugar_g': meal.get('sugar', 0)
                })
                daily_cal += meal.get('calories', 0)
                daily_p += meal.get('protein', 0)
                daily_c += meal.get('carbs', 0)
                daily_f += meal.get('fat', 0)
            weekly_plans.append({
                'day': day_data.get('day'),
                'day_number': day_data.get('day_number'),
                'meals': daily_meals,
                'daily_totals': {'calories': round(daily_cal,1), 'protein_g': round(daily_p,1), 'carbohydrates_g': round(daily_c,1), 'fats_g': round(daily_f,1)}
            })
            weekly_calories += daily_cal; weekly_protein += daily_p; weekly_carbs += daily_c; weekly_fat += daily_f

        response = {
            'weekly_plan': weekly_plans,
            'summary': {
                'total_weekly_calories': round(weekly_calories,1),
                'total_weekly_protein_g': round(weekly_protein,1),
                'total_weekly_carbohydrates_g': round(weekly_carbs,1),
                'total_weekly_fats_g': round(weekly_fat,1),
                'average_daily_calories': round(weekly_calories/7,1),
                'target_daily_calories': round(plan_data.get('targets', {}).get('calories', 0),1),
                'tdee': round(plan_data.get('tdee', 0),1),
                'nutritional_balance': 'Aligned with goal' if abs((weekly_calories/7) - plan_data.get('targets', {}).get('calories', 0)) < 50 else 'Slightly off target'
            },
            'user_profile': {
                'age': age,
                'gender': 'Male' if gender == 0 else 'Female',
                'weight_kg': weight,
                'height_cm': height,
                'fitness_goal': ['Weight Loss','Muscle Gain','Maintain'][goal],
                'activity_level': ['Sedentary','Light','Moderate','Active','Very Active'][activity_level]
            }
        }
        return jsonify(response)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error':'Internal server error'}),500

@app.route('/swap_meal', methods=['POST'])
@login_required
def swap_meal():
    try:
        data = request.get_json()
        required_fields = ['current_meal_name','goal','meal_type']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}),400
        current_meal_name = data['current_meal_name']; goal = int(data['goal']); meal_type = data['meal_type']
        if goal not in [0,1,2]: return jsonify({'error':'Invalid goal'}),400
        valid_meal_types = ['Breakfast','Lunch','Dinner','Snack']
        if meal_type not in valid_meal_types: return jsonify({'error': f'Meal type must be one of: {valid_meal_types}'}),400

        alternatives = diet_model.get_meal_alternatives(current_meal_name, goal, meal_type)
        if not alternatives:
            # fallback: find by calories if provided
            target_cal = float(data.get('target_calories', 0))
            if target_cal > 0:
                alt = find_desi_alternative(target_cal, data.get('allergies', []), exclude_names=[current_meal_name])
                if alt:
                    alternatives = [alt]
        if not alternatives:
            return jsonify({'error':'No suitable alternatives found'}),404

        return jsonify({
            'current_meal': current_meal_name,
            'goal': ['Weight Loss','Muscle Gain','Maintain'][goal],
            'meal_type': meal_type,
            'alternatives': alternatives,
            'message': f'Found {len(alternatives)} alternatives for {current_meal_name}'
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error':'Internal server error'}),500

@app.route('/get_meal_details', methods=['POST'])
@login_required
def get_meal_details():
    try:
        data = request.get_json()
        if 'meal_name' not in data:
            return jsonify({'error':'Missing required field: meal_name'}),400
        meal_name = strip_gram_suffix(data['meal_name'])
        quantity = float(data.get('quantity', 100))

        meal_data = diet_model.combined_data[diet_model.combined_data['Food Name'].str.contains(meal_name, case=False, na=False)]
        if len(meal_data) == 0:
            return jsonify({'error': f'Meal \"{meal_name}\" not found'}),404
        meal = meal_data.iloc[0]
        resp = {
            'meal_name': meal['Food Name'],
            'quantity_grams': quantity,
            'nutritional_breakdown': {
                'calories': round(meal['Calories'] * quantity / 100, 1),
                'protein_g': round(meal.get('Protein (g)', meal.get('Protein', 0)) * quantity / 100, 1),
                'carbohydrates_g': round(meal.get('Carbohydrates (g)', meal.get('Carbohydrates', 0)) * quantity / 100, 1),
                'sugars_g': round(meal.get('Sugars (g)', 0) * quantity / 100, 1),
                'fat_g': round(meal.get('Fat (g)', meal.get('Fat', 0)) * quantity / 100, 1),
                'fiber_g': round(meal.get('Fiber (g)', 0) * quantity / 100, 1)
            },
            'additional_info': {
                'category': meal.get('Category'),
                'meal_type': meal.get('Meal_Type'),
                'is_snack': bool(meal.get('is_snack', 0))
            }
        }
        return jsonify(resp)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error':'Internal server error'}),500

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': diet_model.combined_data is not None if hasattr(diet_model, 'combined_data') else False,
        'total_foods': len(diet_model.combined_data) if hasattr(diet_model, 'combined_data') else 0
    })
def _parse_float(val):
    try:
        if val is None:
            return None
        s = str(val).strip()
        if not s:
            return None
        return float(s)
    except (TypeError, ValueError):
        return None


def detect_plateau(weights_dict):
    ordered_keys = ["start", "week1", "week2", "week3", "week4", "week5", "week6"]
    numeric_values = [_parse_float(weights_dict.get(k)) for k in ordered_keys]

    start_weight = numeric_values[0]
    if start_weight is None:
        return {
            "detected": False,
            "reason": "Starting weight not set yet.",
            "net_change_kg": None,
            "weeks_considered": 0,
        }

    filled_weeks = []
    for i in range(1, len(numeric_values)):
        v = numeric_values[i]
        if v is not None:
            filled_weeks.append((i, v))

    if len(filled_weeks) < 3:
        return {
            "detected": False,
            "reason": "Not enough data yet (need at least 3 weekly check-ins).",
            "net_change_kg": None,
            "weeks_considered": len(filled_weeks),
        }

    latest_idx, latest_val = filled_weeks[-1]
    net_change = float(latest_val - start_weight)
    net_change_abs = abs(net_change)
    negligible_net_change = net_change_abs < 0.5

    has_meaningful_drop = False
    DROP_THRESHOLD = 0.2
    for i in range(1, len(filled_weeks)):
        prev_val = filled_weeks[i - 1][1]
        curr_val = filled_weeks[i][1]
        if prev_val is not None and curr_val is not None and (prev_val - curr_val) >= DROP_THRESHOLD:
            has_meaningful_drop = True
            break

    plateau = negligible_net_change or not has_meaningful_drop

    if plateau:
        if negligible_net_change:
            reason = "Body weight has changed less than 0.5 kg since you started."
        else:
            reason = "No meaningful week-to-week change detected across recent check-ins."
    else:
        reason = "Weight trend shows meaningful change, no plateau detected."

    return {
        "detected": plateau,
        "reason": reason,
        "net_change_kg": round(net_change, 2),
        "weeks_considered": len(filled_weeks),
    }


@app.route('/api/progress/weights', methods=['GET', 'POST'])
@login_required
def progress_weights():
    ordered_keys = ["start", "week1", "week2", "week3", "week4", "week5", "week6"]
    key_to_index = {k: i for i, k in enumerate(ordered_keys)}

    if request.method == 'GET':
        logs = UserWeightLog.query.filter_by(user_id=current_user.id).all()

        weights = {k: None for k in ordered_keys}
        goal_mode = None

        for log in logs:
            if 0 <= log.week_index < len(ordered_keys):
                key = ordered_keys[log.week_index]
                weights[key] = float(log.weight_kg)
            if log.goal_mode and not goal_mode:
                goal_mode = log.goal_mode

        plateau = detect_plateau(weights)

        return jsonify({
            "success": True,
            "weights": weights,
            "goal_mode": goal_mode or "loss",
            "plateau": plateau,
        })

    data = request.get_json() or {}
    incoming = data.get('weights') or {}
    goal_mode = data.get('goal_mode') or None

    cleaned = {}
    for key in ordered_keys:
        cleaned[key] = _parse_float(incoming.get(key))

    for key, value in cleaned.items():
        week_index = key_to_index[key]
        if value is None:
            continue

        log = UserWeightLog.query.filter_by(
            user_id=current_user.id,
            week_index=week_index
        ).first()

        if log is None:
            log = UserWeightLog(
                user_id=current_user.id,
                label=key,
                week_index=week_index,
                weight_kg=value,
                goal_mode=goal_mode,
            )
            db.session.add(log)
        else:
            log.weight_kg = value
            if goal_mode:
                log.goal_mode = goal_mode

    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        traceback.print_exc()
        return jsonify({"success": False, "message": "Failed to save progress."}), 500

    plateau = detect_plateau(cleaned)

    return jsonify({
        "success": True,
        "weights": cleaned,
        "goal_mode": goal_mode,
        "plateau": plateau,
    })

@app.route('/progress')
@login_required
def progress_page():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    progress_dir = os.path.join(base_dir, 'Progress_Tracking')
    return send_from_directory(progress_dir, 'progress.html')

@app.route('/progress-tracking/<path:filename>')
@login_required
def progress_static(filename):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    progress_dir = os.path.join(base_dir, 'Progress_Tracking')
    return send_from_directory(progress_dir, filename)

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ---- Run app ----
if __name__ == '__main__':
    with app.app_context():
        try:
            db.create_all()
            print("✅ Database tables ensured (MySQL: nutrifit).")
        except Exception as e:
            print("⚠️ Could not create DB tables automatically:", e)
            traceback.print_exc()

    print("Starting NutriFit Diet Plan Generator...")
    try:
        total_foods = len(diet_model.combined_data) if hasattr(diet_model, 'combined_data') else 0
        print(f"Total food items loaded: {total_foods}")
    except Exception:
        pass

    app.run(debug=True, host='0.0.0.0', port=5000)