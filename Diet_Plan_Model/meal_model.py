import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

class DietPlanMLP:
    def __init__(self):
        self.foods_data = None
        self.snacks_data = None
        self.combined_data = None
        self.scaler = StandardScaler()
        self.goal_encoder = LabelEncoder()
        self.activity_encoder = LabelEncoder()
        self.meal_type_encoder = LabelEncoder()

    def load_data(self):
        """Load and preprocess the datasets safely"""
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            foods_path = os.path.join(base_dir, "cleaned_foods_dataset.csv")
            snacks_path = os.path.join(base_dir, "cleaned_snacks_dataset.csv")

            self.foods_data = pd.read_csv(foods_path)
            self.snacks_data = pd.read_csv(snacks_path)

            # Mark and align columns
            self.foods_data["is_snack"] = 0
            self.snacks_data["is_snack"] = 1

            # Ensure all needed columns exist, fill missing with 0
            required_cols = [
                "Food Name", "Calories", "Carbohydrates (g)", "Sugars (g)",
                "Fat (g)", "Protein (g)", "Fiber (g)", "Category", "Meal_Type", "is_snack"
            ]
            for col in required_cols:
                if col not in self.foods_data.columns:
                    self.foods_data[col] = 0
                if col not in self.snacks_data.columns:
                    self.snacks_data[col] = 0

            self.foods_data = self.foods_data[required_cols]
            self.snacks_data = self.snacks_data[required_cols]
            self.combined_data = pd.concat([self.foods_data, self.snacks_data], ignore_index=True)

            print(f"✅ Loaded {len(self.foods_data)} foods and {len(self.snacks_data)} snacks.")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

    # ------------------ CALCULATION METHODS ------------------ #
    def calculate_tdee(self, age, gender, weight, height, activity_level):
        """Calculate Total Daily Energy Expenditure"""
        if gender == 0:
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        else:
            bmr = 10 * weight + 6.25 * height - 5 * age - 161
        activity_multipliers = {0: 1.2, 1: 1.375, 2: 1.55, 3: 1.725, 4: 1.9}
        return bmr * activity_multipliers.get(activity_level, 1.2)

    def calculate_target_calories(self, tdee, goal):
        """Adjust calorie target based on fitness goal"""
        if goal == 0:
            return tdee * 0.85
        elif goal == 1:
            return tdee * 1.1
        return tdee

    def calculate_macro_targets(self, target_calories, goal):
        """Set macro distribution"""
        if goal == 0:
            p, c, f = 0.3, 0.35, 0.35
        elif goal == 1:
            p, c, f = 0.25, 0.45, 0.30
        else:
            p, c, f = 0.25, 0.40, 0.35
        return {
            "calories": round(target_calories, 1),
            "protein": round(target_calories * p / 4, 1),
            "carbs": round(target_calories * c / 4, 1),
            "fat": round(target_calories * f / 9, 1)
        }

    # ------------------ MEAL PLAN GENERATION ------------------ #
    def generate_meal_plan(self, age, gender, weight, height, goal, activity_level):
        """Generate a 7-day meal plan"""
        tdee = self.calculate_tdee(age, gender, weight, height, activity_level)
        target_calories = self.calculate_target_calories(tdee, goal)
        macros = self.calculate_macro_targets(target_calories, goal)

        meal_calories = {
            "Breakfast": target_calories * 0.25,
            "Lunch": target_calories * 0.35,
            "Dinner": target_calories * 0.30,
            "Snack": target_calories * 0.10
        }

        def get_foods(meal_type, snack=False):
            df = self.combined_data.copy()
            if snack:
                return df[df["is_snack"] == 1]
            return df[(df["Meal_Type"] == meal_type) & (df["is_snack"] == 0)]

        breakfast_foods = get_foods("Breakfast")
        lunch_foods = get_foods("Lunch")
        dinner_foods = get_foods("Dinner")
        snack_foods = get_foods("", snack=True)

        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekly_plan = []
        used_meals = set()

        for day_idx, day_name in enumerate(days):
            day_meals = []
            for mtype, foods, cal_key in [
                ("Breakfast", breakfast_foods, "Breakfast"),
                ("Lunch", lunch_foods, "Lunch"),
                ("Dinner", dinner_foods, "Dinner"),
                ("Snack", snack_foods, "Snack")
            ]:
                if len(foods) > 0:
                    meal = self.select_meal(foods, meal_calories[cal_key], goal, day_idx, used_meals)
                    if meal:
                        day_meals.append(meal)
                        used_meals.add(meal["name"])
            weekly_plan.append({"day": day_name, "day_number": day_idx + 1, "meals": day_meals})

        return {"weekly_plan": weekly_plan, "targets": macros, "tdee": round(tdee, 1)}

    def select_meal(self, food_options, target_calories, goal, day=0, used_meals=None):
        """Select meal safely"""
        if len(food_options) == 0:
            return None
        used_meals = used_meals or set()
        food_options = food_options.copy()

        min_cal, max_cal = target_calories * 0.85, target_calories * 1.15
        suitable = food_options[(food_options["Calories"] >= min_cal) & (food_options["Calories"] <= max_cal)]
        if len(suitable) == 0:
            suitable = food_options.copy()
        suitable = suitable[~suitable["Food Name"].isin(used_meals)]
        if len(suitable) == 0:
            suitable = food_options.copy()

        if goal == 0:
            suitable["score"] = suitable["Protein (g)"] * 2 - suitable["Carbohydrates (g)"] * 0.5
        elif goal == 1:
            suitable["score"] = suitable["Protein (g)"] * 2 + suitable["Carbohydrates (g)"] * 1.5
        else:
            suitable["score"] = (
                suitable["Protein (g)"] + suitable["Carbohydrates (g)"] + suitable["Fat (g)"]
            )

        np.random.seed(42 + day)
        top_foods = suitable.nlargest(max(1, int(len(suitable) * 0.2)), "score")
        selected = top_foods.sample(1, random_state=42 + day).iloc[0] if len(top_foods) > 1 else top_foods.iloc[0]
        quantity = target_calories / selected["Calories"] * 100 if selected["Calories"] > 0 else 100

        def safe_val(col):
            return round(float(selected.get(col, 0) or 0) * quantity / 100, 1)

        return {
            "name": selected.get("Food Name", "Unknown Meal"),
            "quantity": round(quantity, 1),
            "calories": safe_val("Calories"),
            "protein": safe_val("Protein (g)"),
            "carbs": safe_val("Carbohydrates (g)"),
            "fat": safe_val("Fat (g)"),
            "sugar": safe_val("Sugars (g)"),
            "fiber": safe_val("Fiber (g)")
        }

    # ------------------ SAVE / LOAD METHODS ------------------ #
    def save_model(self, filename="diet_model.pkl"):
        """Save model configuration and datasets"""
        model_data = {
            "scaler": self.scaler,
            "goal_encoder": self.goal_encoder,
            "activity_encoder": self.activity_encoder,
            "meal_type_encoder": self.meal_type_encoder,
            "foods_data": self.foods_data,
            "snacks_data": self.snacks_data,
            "combined_data": self.combined_data
        }
        joblib.dump(model_data, filename)
        print(f"💾 Model saved to {filename}")

    def load_model(self, filename="diet_model.pkl"):
        """Load previously saved model"""
        try:
            model_data = joblib.load(filename)
            self.scaler = model_data["scaler"]
            self.goal_encoder = model_data["goal_encoder"]
            self.activity_encoder = model_data["activity_encoder"]
            self.meal_type_encoder = model_data["meal_type_encoder"]
            self.foods_data = model_data["foods_data"]
            self.snacks_data = model_data["snacks_data"]
            self.combined_data = model_data["combined_data"]
            print(f"✅ Model loaded from {filename}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False


# ------------------ RUN IF EXECUTED DIRECTLY ------------------ #
diet_model = DietPlanMLP()

if __name__ == "__main__":
    if diet_model.load_data():
        print("✅ Diet Plan Model ready.")
        diet_model.save_model()  # <-- This line creates the .pkl file
        plan = diet_model.generate_meal_plan(25, 0, 70, 175, 0, 2)
        print(f"\nSample: {plan['targets']}")
        for day in plan["weekly_plan"][:1]:
            print(f"\n{day['day']}:")
            for meal in day["meals"]:
                print(f" - {meal['name']}: {meal['calories']} kcal")
    else:
        print("❌ Failed to initialize model.")
