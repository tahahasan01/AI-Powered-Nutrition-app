


# diet_model.py (DL version with no-zero-calorie fallback and proper else branches)
import os
import json
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler

# ---- Deep learning (PyTorch) ----
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as e:
    raise RuntimeError(
        "PyTorch is required for the DL diet model. Install with: pip install torch --index-url https://download.pytorch.org/whl/cpu"
    ) from e


# ------------------------------
# Energy constants (per your request)
# ------------------------------
KCAL_PER_G_PROTEIN = 4.0
KCAL_PER_G_CARB = 4.0
KCAL_PER_G_FAT = 8.0  # <-- as requested (NOT 9)

def kcal_from_macros(p_g: float, c_g: float, f_g: float) -> float:
    """Compute kcal from macros using your requested multipliers."""
    return (p_g * KCAL_PER_G_PROTEIN) + (c_g * KCAL_PER_G_CARB) + (f_g * KCAL_PER_G_FAT)

def ensure_positive_calories(cal: float, p_g: float, c_g: float, f_g: float, default: float = 200.0) -> float:
    """
    If 'cal' is <= 0 (or missing), recompute from macros; if still <= 0, use a safe default.
    Ensures we never return zero calories.
    """
    if cal is None:
        cal = 0.0
    if cal <= 0:
        cal = kcal_from_macros(p_g, c_g, f_g)
        if cal <= 0:
            cal = float(default)
    return float(cal)

def clamp_qty(qty: float, lo: float = 50.0, hi: float = 600.0) -> float:
    """Clamp serving quantity (grams) to a reasonable range."""
    return max(lo, min(hi, float(qty)))


# ------------------------------
# Small helper modules (PyTorch)
# ------------------------------
class ProfileToTargetsNet(nn.Module):
    """
    Input: encoded profile vector
    Outputs:
      - tdee             (1)
      - target_calories  (1)
      - macro grams      (3) -> protein_g, carbs_g, fat_g
      - per-meal cals    (4) -> breakfast, lunch, dinner, snack
    """
    def __init__(self, in_dim: int, hidden: List[int] = [128, 128]):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.backbone = nn.Sequential(*layers)

        self.head_tgt = nn.Linear(last, 2)
        self.head_macro = nn.Linear(last, 3)
        self.head_meals = nn.Linear(last, 4)

    def forward(self, x: torch.Tensor):
        z = self.backbone(x)

        tgt_raw = self.head_tgt(z)      # [B, 2]
        macro_raw = self.head_macro(z)  # [B, 3]
        meal_logits = self.head_meals(z)  # [B, 4]

        # Positivity constraints
        # (ReLU to keep outputs positive; softmax for meal fractions)
        tdee = F.relu(tgt_raw[:, 0]) + 1000.0         # add a baseline to avoid near-zero
        target_cal = F.relu(tgt_raw[:, 1]) + 1000.0

        macro = F.relu(macro_raw)  # grams, >= 0

        # fractions sum to 1, multiply by total calories, then renormalize defensively
        meal_fracs = torch.softmax(meal_logits, dim=1)      # [B, 4], sums to 1
        meal_cals = meal_fracs * target_cal.unsqueeze(1)    # [B, 4]
        # Defensive re-normalization to avoid numeric drift:
        s = meal_cals.sum(dim=1, keepdim=True).clamp_min(1e-6)
        meal_cals = meal_cals / s * target_cal.unsqueeze(1)

        return tdee, target_cal, macro, meal_cals


class FoodSuitabilityNet(nn.Module):
    """
    Feeds a concatenated feature vector describing:
      - user profile (scaled numerics + one-hots)
      - predicted targets (tdee, target_cal, target_meal_cal)
      - meal type one-hot
      - food nutrition features
    Output: scalar suitability score (higher = better)
    """
    def __init__(self, in_dim: int, hidden: List[int] = [128, 64], dropout: float = 0.1):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.net(x).squeeze(1)  # [B]


# ------------------------------
# Main Diet model (trainable)
# ------------------------------
class DietPlanDL:
    """
    End-to-end trainable replacement for the old rule-based DietPlanMLP.
    - Trains two NNs:
        * ProfileToTargetsNet   (predict tdee/targets/macros/meal splits)
        * FoodSuitabilityNet    (rank foods given profile + meal type)
    - After training, generate_meal_plan() uses ONLY the neural nets.
    """
    MEAL_TYPES = ["Breakfast", "Lunch", "Dinner", "Snack"]

    def __init__(self, use_math_fallback: bool = True, seed: int = 42):
        self.foods_data: Optional[pd.DataFrame] = None
        self.snacks_data: Optional[pd.DataFrame] = None
        self.combined_data: Optional[pd.DataFrame] = None

        # Saved/loaded artifacts
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        os.makedirs(self.model_dir, exist_ok=True)

        # Scalers for model inputs
        self.profile_num_scaler = StandardScaler()   # age, weight, height, bmi
        self.ranker_num_scaler = StandardScaler()    # numeric block for ranker (see below)

        # Nets
        self.profile_net: Optional[ProfileToTargetsNet] = None
        self.food_net: Optional[FoodSuitabilityNet] = None

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Config
        self.use_math_fallback = use_math_fallback
        self.rng = np.random.default_rng(seed)

        # For swap_meal convenience (use last user profile if available)
        self._last_profile_tuple: Optional[Tuple[int, int, float, float, int, int]] = None

    # ------------------ DATA ------------------ #
    def load_data(self) -> bool:
        """Load and align foods + snacks datasets."""
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            foods_path = os.path.join(base_dir, "cleaned_foods_dataset.csv")
            snacks_path = os.path.join(base_dir, "cleaned_snacks_dataset.csv")

            df_f = pd.read_csv(foods_path)
            df_s = pd.read_csv(snacks_path)

            df_f["is_snack"] = 0
            df_s["is_snack"] = 1

            required_cols = [
                "Food Name", "Calories", "Carbohydrates (g)", "Sugars (g)",
                "Fat (g)", "Protein (g)", "Fiber (g)", "Category", "Meal_Type", "is_snack"
            ]
            for col in required_cols:
                if col not in df_f.columns: df_f[col] = 0
                if col not in df_s.columns: df_s[col] = 0

            df_f = df_f[required_cols]
            df_s = df_s[required_cols]

            self.foods_data = df_f
            self.snacks_data = df_s
            self.combined_data = pd.concat([df_f, df_s], ignore_index=True)

            print(f"✅ Loaded {len(df_f)} foods and {len(df_s)} snacks.")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

    # ------------------ "TEACHER" MATH (for weak labels) ------------------ #
    # These are NOT used at inference. They only generate pseudo-labels for training.
    def _math_tdee(self, age, gender, weight, height, activity_level):
        # Mifflin-St Jeor + activity multipliers
        if gender == 0:  # Male
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        else:  # Female
            bmr = 10 * weight + 6.25 * height - 5 * age - 161
        activity_multipliers = {0: 1.2, 1: 1.375, 2: 1.55, 3: 1.725, 4: 1.9}
        return bmr * activity_multipliers.get(activity_level, 1.2)

    def _math_target_cal(self, tdee, goal):
        # 0: loss 0.85x, 1: gain 1.1x, 2: maintain
        if goal == 0: return tdee * 0.85
        if goal == 1: return tdee * 1.10
        return tdee

    def _math_macros(self, target_cal, goal):
        # grams
        if goal == 0:
            p, c, f = 0.30, 0.35, 0.35
        elif goal == 1:
            p, c, f = 0.25, 0.45, 0.30
        else:
            p, c, f = 0.25, 0.40, 0.35
        return (
            target_cal * p / 4.0,
            target_cal * c / 4.0,
            target_cal * f / KCAL_PER_G_FAT,  # use your fat-kcal here for consistency
        )

    def _math_meal_splits(self, target_cal):
        # classic 25/35/30/10 -> returns absolute per-meal calories
        return (
            target_cal * 0.25,
            target_cal * 0.35,
            target_cal * 0.30,
            target_cal * 0.10,
        )

    def _teacher_score_food(self, goal, meal_target_cal, row):
        """
        Weak-label scoring used to train the FoodSuitabilityNet.
        Higher is better.
        Applies no-zero-kcal fallback from macros when Calories is missing/0.
        """
        p = float(row.get("Protein (g)", row.get("Protein", 0)) or 0)
        c = float(row.get("Carbohydrates (g)", row.get("Carbohydrates", 0)) or 0)
        f = float(row.get("Fat (g)", 0) or 0)
        s = float(row.get("Sugars (g)", 0) or 0)
        fi = float(row.get("Fiber (g)", 0) or 0)
        is_snack = int(row.get("is_snack", 0) or 0)

        cal_raw = float(row.get("Calories", 0) or 0)
        cal = ensure_positive_calories(cal_raw, p, c, f)

        # calorie closeness (per 100g vs per-meal target)
        cal_diff = abs(cal - meal_target_cal)
        cal_score = - (cal_diff / max(1.0, meal_target_cal)) * 2.0  # weight

        # macro preference by goal
        if goal == 0:      # weight loss
            macro_score = (2.0 * p) - (0.5 * c) - (0.7 * s) + (0.3 * fi) - (0.2 * f)
        elif goal == 1:    # muscle gain
            macro_score = (2.0 * p) + (1.2 * c) - (0.2 * s) + (0.1 * fi)
        else:              # maintain
            macro_score = (1.0 * p) + (1.0 * c) + (0.8 * f) - (0.5 * s) + (0.2 * fi)

        # snack alignment bonus/penalty
        snack_bonus = 0.2 if (is_snack == 1 and meal_target_cal < 400) else 0.0
        snack_pen = -0.2 if (is_snack == 1 and meal_target_cal >= 400) else 0.0

        return macro_score + cal_score + snack_bonus + snack_pen

    # ------------------ ENCODING ------------------ #
    def _encode_profile(self, age, gender, weight, height, goal, activity):
        bmi = weight / ((height / 100.0) ** 2 + 1e-6)
        goal_oh = [1.0 if goal == i else 0.0 for i in range(3)]
        act_oh = [1.0 if activity == i else 0.0 for i in range(5)]
        # numeric (4): age, weight, height, bmi   | gender (1) | goal oh (3) | activity oh (5)
        x = np.array([age, weight, height, bmi, float(gender)], dtype=np.float32)
        x_cat = np.array(goal_oh + act_oh, dtype=np.float32)
        return x, x_cat  # (4+1) and (3+5)

    def _scale_profile_for_profile_net(self, x_num, x_cat):
        # scale only first 4 numerics (age, weight, height, bmi). keep gender + onehots unchanged
        x_scaled4 = self.profile_num_scaler.transform(x_num[:, :4])
        x_out = np.concatenate([x_scaled4, x_num[:, 4:5], x_cat], axis=1)
        return x_out

    # ------------------ TRAINING ------------------ #
    def _build_profile_training(self, n_profiles: int = 50000):
        """Create synthetic profiles + 'teacher' labels."""
        ages = self.rng.integers(18, 70, size=n_profiles)
        genders = self.rng.integers(0, 2, size=n_profiles)           # 0 male, 1 female
        weights = self.rng.uniform(45, 140, size=n_profiles)         # kg
        heights = self.rng.uniform(150, 200, size=n_profiles)        # cm
        goals = self.rng.integers(0, 3, size=n_profiles)             # 0/1/2
        acts = self.rng.integers(0, 5, size=n_profiles)              # 0..4

        X_num, X_cat = [], []
        Y_tdee, Y_target, Y_macros, Y_meals = [], [], [], []

        for i in range(n_profiles):
            age = int(ages[i]); gender = int(genders[i]); w = float(weights[i]); h = float(heights[i])
            goal = int(goals[i]); act = int(acts[i])

            tdee = self._math_tdee(age, gender, w, h, act)
            tgt = self._math_target_cal(tdee, goal)
            p, c, f = self._math_macros(tgt, goal)
            b, l, d, s = self._math_meal_splits(tgt)

            xnum, xcat = self._encode_profile(age, gender, w, h, goal, act)
            X_num.append(xnum); X_cat.append(xcat)
            Y_tdee.append([tdee]); Y_target.append([tgt])
            Y_macros.append([p, c, f])
            Y_meals.append([b, l, d, s])

        X_num = np.stack(X_num, axis=0)    # [N, 5]
        X_cat = np.stack(X_cat, axis=0)    # [N, 8]
        Y_tdee = np.array(Y_tdee, dtype=np.float32)
        Y_target = np.array(Y_target, dtype=np.float32)
        Y_macros = np.array(Y_macros, dtype=np.float32)
        Y_meals = np.array(Y_meals, dtype=np.float32)

        # fit scaler on first 4 numerics
        self.profile_num_scaler.fit(X_num[:, :4])

        X_profile = self._scale_profile_for_profile_net(X_num, X_cat)  # [N, 13]
        return X_profile, Y_tdee, Y_target, Y_macros, Y_meals

    def train_profile_net(self, n_profiles: int = 50000, batch_size: int = 512, epochs: int = 35, lr: float = 1e-3):
        X, y_tdee, y_tgt, y_mac, y_meals = self._build_profile_training(n_profiles)

        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y1 = torch.tensor(y_tdee.squeeze(1), dtype=torch.float32).to(self.device)
        y2 = torch.tensor(y_tgt.squeeze(1), dtype=torch.float32).to(self.device)
        y3 = torch.tensor(y_mac, dtype=torch.float32).to(self.device)
        y4 = torch.tensor(y_meals, dtype=torch.float32).to(self.device)

        ds = TensorDataset(X_t, y1, y2, y3, y4)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

        self.profile_net = ProfileToTargetsNet(in_dim=X.shape[1]).to(self.device)
        opt = torch.optim.Adam(self.profile_net.parameters(), lr=lr)
        mse = nn.MSELoss()

        self.profile_net.train()
        for ep in range(epochs):
            total = 0.0
            for xb, tdee_b, tgt_b, mac_b, meal_b in dl:
                opt.zero_grad()
                p_tdee, p_tgt, p_mac, p_meal = self.profile_net(xb)
                loss = (
                    mse(p_tdee, tdee_b) +
                    mse(p_tgt, tgt_b) +
                    mse(p_mac, mac_b) +
                    mse(p_meal, meal_b)
                )
                loss.backward()
                opt.step()
                total += float(loss.item()) * xb.size(0)
            print(f"[ProfileNet] epoch {ep+1}/{epochs}  loss={total/len(ds):.4f}")

        # Save model + scaler
        torch.save(self.profile_net.state_dict(), os.path.join(self.model_dir, "profile_net.pt"))
        joblib.dump(self.profile_num_scaler, os.path.join(self.model_dir, "profile_num_scaler.joblib"))
        print("✅ Saved profile_net + scaler")

    def _build_ranker_training(self, profiles: int = 8000, foods_per_meal: int = 40):
        assert self.combined_data is not None and len(self.combined_data) > 0, "Load data first."

        # pick candidate pools by meal type
        df_all = self.combined_data.copy()
        df_b = df_all[(df_all["Meal_Type"] == "Breakfast") & (df_all["is_snack"] == 0)]
        df_l = df_all[(df_all["Meal_Type"] == "Lunch") & (df_all["is_snack"] == 0)]
        df_d = df_all[(df_all["Meal_Type"] == "Dinner") & (df_all["is_snack"] == 0)]
        df_s = df_all[df_all["is_snack"] == 1]

        pools = {
            "Breakfast": df_b if len(df_b) else df_all[df_all["is_snack"] == 0],
            "Lunch": df_l if len(df_l) else df_all[df_all["is_snack"] == 0],
            "Dinner": df_d if len(df_d) else df_all[df_all["is_snack"] == 0],
            "Snack": df_s if len(df_s) else df_all[df_all["is_snack"] == 1],
        }

        # containers
        X_num_list, X_cat_list, y_list = [], [], []

        for _ in range(profiles):
            # profile
            age = int(self.rng.integers(18, 70))
            gender = int(self.rng.integers(0, 2))
            weight = float(self.rng.uniform(45, 140))
            height = float(self.rng.uniform(150, 200))
            goal = int(self.rng.integers(0, 3))
            act = int(self.rng.integers(0, 5))
            xnum, xcat = self._encode_profile(age, gender, weight, height, goal, act)

            # teacher targets for this profile
            tdee = self._math_tdee(age, gender, weight, height, act)
            tgt = self._math_target_cal(tdee, goal)
            b, l, d, s = self._math_meal_splits(tgt)
            meal_targets = {"Breakfast": b, "Lunch": l, "Dinner": d, "Snack": s}

            for mtype in self.MEAL_TYPES:
                pool = pools[mtype]
                if len(pool) == 0:  # no foods? skip
                    continue

                # sample a subset of foods for speed
                k = min(foods_per_meal, len(pool))
                idx = self.rng.choice(len(pool), size=k, replace=False)
                sub = pool.iloc[idx]

                for _, row in sub.iterrows():
                    meal_cal_t = meal_targets[mtype]

                    # numeric features for ranker
                    # num block: age, weight, height, bmi, gender, tdee, tgt, target_meal_cal,
                    #            food: cal, p, c, f, sugar, fiber, is_snack
                    bmi = weight / ((height / 100.0) ** 2 + 1e-6)
                    fp = float(row.get("Protein (g)", row.get("Protein", 0)) or 0)
                    fc = float(row.get("Carbohydrates (g)", row.get("Carbohydrates", 0)) or 0)
                    ff = float(row.get("Fat (g)", 0) or 0)
                    fsu = float(row.get("Sugars (g)", 0) or 0)
                    ffi = float(row.get("Fiber (g)", 0) or 0)
                    isn = float(row.get("is_snack", 0) or 0)

                    fcal_raw = float(row.get("Calories", 0) or 0)
                    fcal = ensure_positive_calories(fcal_raw, fp, fc, ff)

                    x_num = np.array([
                        age, weight, height, bmi, float(gender),
                        tdee, tgt, meal_cal_t,
                        fcal, fp, fc, ff, fsu, ffi, isn
                    ], dtype=np.float32)

                    # categorical/one-hot: goal(3) + activity(5) + meal_type(4)
                    goal_oh = [1.0 if goal == i else 0.0 for i in range(3)]
                    act_oh = [1.0 if act == i else 0.0 for i in range(5)]
                    meal_oh = [1.0 if mtype == mt else 0.0 for mt in self.MEAL_TYPES]
                    x_cat = np.array(goal_oh + act_oh + meal_oh, dtype=np.float32)

                    # teacher score (uses no-zero fallback internally)
                    y = self._teacher_score_food(goal, meal_cal_t, row)

                    X_num_list.append(x_num)
                    X_cat_list.append(x_cat)
                    y_list.append(y)

        X_num = np.stack(X_num_list, axis=0)  # [N, 15]
        X_cat = np.stack(X_cat_list, axis=0)  # [N, 12]
        y = np.array(y_list, dtype=np.float32)

        # fit scaler on numeric block
        self.ranker_num_scaler.fit(X_num)

        X_num_scaled = self.ranker_num_scaler.transform(X_num)
        X_all = np.concatenate([X_num_scaled, X_cat], axis=1)  # [N, 27]
        return X_all, y

    def train_food_net(self, profiles: int = 8000, foods_per_meal: int = 40,
                       batch_size: int = 1024, epochs: int = 25, lr: float = 1e-3):
        X, y = self._build_ranker_training(profiles=profiles, foods_per_meal=foods_per_meal)

        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.float32).to(self.device)

        ds = TensorDataset(X_t, y_t)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

        self.food_net = FoodSuitabilityNet(in_dim=X.shape[1]).to(self.device)
        opt = torch.optim.Adam(self.food_net.parameters(), lr=lr)
        mse = nn.MSELoss()

        self.food_net.train()
        for ep in range(epochs):
            total = 0.0
            for xb, yb in dl:
                opt.zero_grad()
                pred = self.food_net(xb)
                loss = mse(pred, yb)
                loss.backward()
                opt.step()
                total += float(loss.item()) * xb.size(0)
            print(f"[FoodNet] epoch {ep+1}/{epochs}  loss={total/len(ds):.4f}")

        # Save model + scaler
        torch.save(self.food_net.state_dict(), os.path.join(self.model_dir, "food_net.pt"))
        joblib.dump(self.ranker_num_scaler, os.path.join(self.model_dir, "ranker_num_scaler.joblib"))
        print("✅ Saved food_net + scaler")

    def train_all(self,
                  profile_samples: int = 50000, profile_epochs: int = 35,
                  ranker_profiles: int = 8000, foods_per_meal: int = 40, ranker_epochs: int = 25):
        assert self.combined_data is not None, "Call load_data() first."

        print("🚀 Training Profile→Targets net...")
        self.train_profile_net(n_profiles=profile_samples, epochs=profile_epochs)

        print("🚀 Training Food Suitability net...")
        self.train_food_net(profiles=ranker_profiles, foods_per_meal=foods_per_meal, epochs=ranker_epochs)

    # ------------------ LOADING ------------------ #
    def load_models(self) -> bool:
        """Load trained nets + scalers from disk, if present."""
        try:
            prof_sd = os.path.join(self.model_dir, "profile_net.pt")
            food_sd = os.path.join(self.model_dir, "food_net.pt")
            prof_scl = os.path.join(self.model_dir, "profile_num_scaler.joblib")
            rank_scl = os.path.join(self.model_dir, "ranker_num_scaler.joblib")

            if not (os.path.exists(prof_sd) and os.path.exists(food_sd) and
                    os.path.exists(prof_scl) and os.path.exists(rank_scl)):
                print("⚠️ Models or scalers not found; train first (python diet_model.py --train).")
                return False

            # Restore scalers
            self.profile_num_scaler = joblib.load(prof_scl)
            self.ranker_num_scaler = joblib.load(rank_scl)

            # Dummy in-dims to construct modules
            in_dim_profile = 13  # 4 scaled numerics + gender + 3 goal + 5 activity
            in_dim_ranker = 27   # 15 numeric scaled + 12 one-hot

            self.profile_net = ProfileToTargetsNet(in_dim=in_dim_profile).to(self.device)
            self.profile_net.load_state_dict(torch.load(prof_sd, map_location=self.device))
            self.profile_net.eval()

            self.food_net = FoodSuitabilityNet(in_dim=in_dim_ranker).to(self.device)
            self.food_net.load_state_dict(torch.load(food_sd, map_location=self.device))
            self.food_net.eval()

            print("✅ Loaded trained models.")
            return True
        except Exception as e:
            print(f"❌ Failed to load models: {e}")
            return False

    # ------------------ INFERENCE HELPERS ------------------ #
    def _predict_profile_targets(self, age, gender, weight, height, goal, activity):
        """Use the profile_net to predict TDEE, target calories, macro grams, per-meal calories."""
        assert self.profile_net is not None, "profile_net not loaded/trained."

        xnum, xcat = self._encode_profile(age, gender, weight, height, goal, activity)  # (5) and (8)
        xnum = xnum.reshape(1, -1)
        xcat = xcat.reshape(1, -1)

        x_prof = self._scale_profile_for_profile_net(xnum, xcat)   # [1, 13]
        x_t = torch.tensor(x_prof, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            tdee, tgt, macros, meal_cals = self.profile_net(x_t)
        tdee = float(tdee.item())
        tgt = float(tgt.item())
        p, c, f = [float(v) for v in macros.squeeze(0).cpu().numpy()]
        bcals, lcals, dcals, scals = [float(v) for v in meal_cals.squeeze(0).cpu().numpy()]
        return tdee, tgt, (p, c, f), (bcals, lcals, dcals, scals)

    def _score_candidates(self, age, gender, weight, height, goal, activity,
                          meal_type: str, target_meal_cal: float, candidates: pd.DataFrame) -> pd.DataFrame:
        """Use food_net to score a set of candidate foods for this profile + meal_type."""
        assert self.food_net is not None, "food_net not loaded/trained."

        # prepare profile parts (unscaled BMI etc)
        bmi = weight / ((height / 100.0) ** 2 + 1e-6)
        # also get profile-level predictions to include as features
        tdee, tgt, _, _ = self._predict_profile_targets(age, gender, weight, height, goal, activity)

        # Build numeric + cat blocks
        goal_oh = np.array([1.0 if goal == i else 0.0 for i in range(3)], dtype=np.float32)
        act_oh = np.array([1.0 if activity == i else 0.0 for i in range(5)], dtype=np.float32)
        meal_oh = np.array([1.0 if meal_type == mt else 0.0 for mt in self.MEAL_TYPES], dtype=np.float32)

        X_rows = []
        for _, row in candidates.iterrows():
            fp = float(row.get("Protein (g)", row.get("Protein", 0)) or 0)
            fc = float(row.get("Carbohydrates (g)", row.get("Carbohydrates", 0)) or 0)
            ff = float(row.get("Fat (g)", 0) or 0)
            fsu = float(row.get("Sugars (g)", 0) or 0)
            ffi = float(row.get("Fiber (g)", 0) or 0)
            isn = float(row.get("is_snack", 0) or 0)

            fcal_raw = float(row.get("Calories", 0) or 0)
            fcal = ensure_positive_calories(fcal_raw, fp, fc, ff)

            x_num = np.array([
                age, weight, height, bmi, float(gender),
                tdee, tgt, target_meal_cal,
                fcal, fp, fc, ff, fsu, ffi, isn
            ], dtype=np.float32)

            x_num_scaled = self.ranker_num_scaler.transform(x_num.reshape(1, -1))
            x_all = np.concatenate([x_num_scaled, goal_oh.reshape(1, -1), act_oh.reshape(1, -1), meal_oh.reshape(1, -1)], axis=1)
            X_rows.append(x_all)

        X = np.vstack(X_rows) if len(X_rows) else np.zeros((0, 27), dtype=np.float32)
        if len(X) == 0:
            candidates = candidates.copy()
            candidates["__score__"] = -1e9
            return candidates

        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            scores = self.food_net(X_t).cpu().numpy()
        scored = candidates.copy()
        scored["__score__"] = scores
        return scored

    def _get_candidates_for_meal(self, meal_type: str) -> pd.DataFrame:
        assert self.combined_data is not None
        df = self.combined_data.copy()
        if meal_type == "Snack":
            return df[df["is_snack"] == 1]
        else:
            # prefer items labeled for that meal (non-snack)
            base = df[(df["Meal_Type"] == meal_type) & (df["is_snack"] == 0)]
            if len(base) > 0:
                return base
            # fallback: any non-snack
            return df[df["is_snack"] == 0]

    # ------------------ PUBLIC API ------------------ #
    def generate_meal_plan(self, age, gender, weight, height, goal, activity_level):
        """
        Generate a 7-day plan using only neural networks (if available).
        If models are missing and use_math_fallback=True, falls back to math for targets and naive pick.
        """
        # remember last profile for swap endpoint
        self._last_profile_tuple = (age, gender, weight, height, goal, activity_level)

        if self.profile_net is None or self.food_net is None:
            print("⚠️ DL models are not loaded. ", end="")
            if self.use_math_fallback:
                print("Using math fallback to generate a plan.")
                return self._math_generate_meal_plan(age, gender, weight, height, goal, activity_level)
            print("Returning empty plan.")
            return {"weekly_plan": [], "targets": {}, "tdee": 0.0}

        # Predict profile targets
        tdee, tgt, (p, c, f), (bcals, lcals, dcals, scals) = self._predict_profile_targets(
            age, gender, weight, height, goal, activity_level
        )
        meal_targets = {"Breakfast": bcals, "Lunch": lcals, "Dinner": dcals, "Snack": scals}

        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekly_plan = []
        used_meals = set()

        for di, day in enumerate(days):
            day_meals = []
            for mtype in self.MEAL_TYPES:
                pool = self._get_candidates_for_meal(mtype)
                if len(pool) == 0:
                    continue

                # model-based ranking
                scored = self._score_candidates(
                    age, gender, weight, height, goal, activity_level,
                    mtype, meal_targets[mtype], pool
                ).sort_values("__score__", ascending=False)

                # avoid repeats across the week
                scored = scored[~scored["Food Name"].isin(used_meals)]
                if len(scored) == 0:
                    scored = pool.copy().sample(n=min(1, len(pool)), random_state=42 + di)
                selected = scored.iloc[0]

                # ---- compute quantity to hit predicted per-meal calories (with fallback) ----
                target_cal = float(meal_targets[mtype] or 0)

                fp = float(selected.get("Protein (g)", selected.get("Protein", 0)) or 0)
                fc = float(selected.get("Carbohydrates (g)", selected.get("Carbohydrates", 0)) or 0)
                ff = float(selected.get("Fat (g)", 0) or 0)

                cal_per_100_raw = float(selected.get("Calories", 0) or 0)
                cal_per_100 = ensure_positive_calories(cal_per_100_raw, fp, fc, ff)

                qty = clamp_qty((target_cal / cal_per_100) * 100.0)
                kcal = cal_per_100 * qty / 100.0

                # compute macros at this quantity
                protein_g = round(fp * qty / 100.0, 1)
                carbs_g   = round(fc * qty / 100.0, 1)
                fat_g     = round(ff * qty / 100.0, 1)

                # final no-zero guarantee for calories
                kcal_final = round(kcal, 1)
                if kcal_final <= 0:
                    kcal_final = round(kcal_from_macros(protein_g, carbs_g, fat_g), 1)

                meal_out = {
                    "name": selected.get("Food Name", "Unknown Meal"),
                    "quantity": round(qty, 1),
                    "calories": kcal_final,
                    "protein": protein_g,
                    "carbs": carbs_g,
                    "fat": fat_g,
                    "sugar": round(float(selected.get("Sugars (g)", 0) or 0) * qty / 100.0, 1),
                    "fiber": round(float(selected.get("Fiber (g)", 0) or 0) * qty / 100.0, 1)
                }
                day_meals.append(meal_out)
                used_meals.add(meal_out["name"])

            weekly_plan.append({"day": day, "day_number": len(weekly_plan) + 1, "meals": day_meals})

        targets_dict = {
            "calories": round(tgt, 1),
            "protein": round(p, 1),
            "carbs": round(c, 1),
            "fat": round(f, 1)
        }
        return {"weekly_plan": weekly_plan, "targets": targets_dict, "tdee": round(tdee, 1)}

    def get_meal_alternatives(self, current_meal_name: str, goal: int, meal_type: str, top_k: int = 5):
        """
        Rank alternatives for the current meal using the FoodSuitabilityNet.
        Uses the last profile (from last generate_meal_plan call) if available,
        otherwise a sensible default profile.
        """
        if self.food_net is None:
            return []

        # Use last known profile if available
        if self._last_profile_tuple is not None:
            age, gender, weight, height, g, act = self._last_profile_tuple
            goal = g if goal not in (0, 1, 2) else goal
        else:
            # Default profile
            age, gender, weight, height, act = 30, 0, 75.0, 175.0, 2  # moderate

        # Build candidates excluding current meal
        pool = self._get_candidates_for_meal(meal_type)
        pool = pool[pool["Food Name"].str.lower() != str(current_meal_name or "").lower()]
        if len(pool) == 0:
            return []

        # Get target meal calories from profile_net
        tdee, tgt, _, (bc, lc, dc, sc) = self._predict_profile_targets(age, gender, weight, height, goal, act)
        mcal = {"Breakfast": bc, "Lunch": lc, "Dinner": dc, "Snack": sc}[meal_type]

        scored = self._score_candidates(age, gender, weight, height, goal, act, meal_type, mcal, pool)
        scored = scored.sort_values("__score__", ascending=False).head(top_k)

        alts = []
        for _, row in scored.iterrows():
            fp = float(row.get("Protein (g)", row.get("Protein", 0)) or 0)
            fc = float(row.get("Carbohydrates (g)", row.get("Carbohydrates", 0)) or 0)
            ff = float(row.get("Fat (g)", 0) or 0)

            cal_per_100_raw = float(row.get("Calories", 0) or 0)
            cal_per_100 = ensure_positive_calories(cal_per_100_raw, fp, fc, ff)

            qty = clamp_qty((mcal / cal_per_100) * 100.0)
            kcal = cal_per_100 * qty / 100.0

            protein_g = round(fp * qty / 100.0, 1)
            carbs_g   = round(fc * qty / 100.0, 1)
            fat_g     = round(ff * qty / 100.0, 1)

            kcal_final = round(kcal, 1)
            if kcal_final <= 0:
                kcal_final = round(kcal_from_macros(protein_g, carbs_g, fat_g), 1)

            alts.append({
                "name": row.get("Food Name", "Unknown Meal"),
                "quantity": round(qty, 1),
                "calories": kcal_final,
                "protein": protein_g,
                "carbs": carbs_g,
                "fat": fat_g,
                "sugar": round(float(row.get("Sugars (g)", 0) or 0) * qty / 100.0, 1),
                "fiber": round(float(row.get("Fiber (g)", 0) or 0) * qty / 100.0, 1)
            })
        return alts

    # ------------------ OPTIONAL: Math fallback generator ------------------ #
    def _math_generate_meal_plan(self, age, gender, weight, height, goal, activity_level):
        """
        This replicates the old rule-based flow and is used only if models are missing.
        Includes no-zero-calorie fallback.
        """
        tdee = self._math_tdee(age, gender, weight, height, activity_level)
        tgt = self._math_target_cal(tdee, goal)
        p, c, f = self._math_macros(tgt, goal)
        bcal, lcal, dcal, scal = self._math_meal_splits(tgt)

        def get_foods(meal_type, snack=False):
            df = self.combined_data.copy()
            if snack: return df[df["is_snack"] == 1]
            return df[(df["Meal_Type"] == meal_type) & (df["is_snack"] == 0)]

        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekly_plan = []
        used = set()
        meal_targets = {"Breakfast": bcal, "Lunch": lcal, "Dinner": dcal, "Snack": scal}

        for di, day in enumerate(days):
            day_meals = []
            for mtype in self.MEAL_TYPES:
                pool = get_foods(mtype, snack=(mtype == "Snack"))
                if len(pool) == 0:
                    continue
                # simple heuristic: closeness in calories + goal macro weights
                pool = pool.copy()
                pool["__score__"] = pool.apply(lambda r: self._teacher_score_food(goal, meal_targets[mtype], r), axis=1)
                pool = pool.sort_values("__score__", ascending=False)
                pool = pool[~pool["Food Name"].isin(used)]
                if len(pool) == 0:
                    continue
                sel = pool.iloc[0]

                # calories per 100g with fallback
                fp = float(sel.get("Protein (g)", sel.get("Protein", 0)) or 0)
                fc = float(sel.get("Carbohydrates (g)", sel.get("Carbohydrates", 0)) or 0)
                ff = float(sel.get("Fat (g)", 0) or 0)
                cal100_raw = float(sel.get("Calories", 0) or 0)
                cal100 = ensure_positive_calories(cal100_raw, fp, fc, ff)

                tcal = meal_targets[mtype]
                qty = clamp_qty((tcal / cal100) * 100.0)

                protein_g = round(fp * qty / 100.0, 1)
                carbs_g   = round(fc * qty / 100.0, 1)
                fat_g     = round(ff * qty / 100.0, 1)

                kcal = cal100 * qty / 100.0
                kcal_final = round(kcal, 1)
                if kcal_final <= 0:
                    kcal_final = round(kcal_from_macros(protein_g, carbs_g, fat_g), 1)

                day_meals.append({
                    "name": sel.get("Food Name", "Unknown Meal"),
                    "quantity": round(qty, 1),
                    "calories": kcal_final,
                    "protein": protein_g,
                    "carbs": carbs_g,
                    "fat": fat_g,
                    "sugar": round(float(sel.get("Sugars (g)", 0) or 0) * qty / 100.0, 1),
                    "fiber": round(float(sel.get("Fiber (g)", 0) or 0) * qty / 100.0, 1)
                })
                used.add(day_meals[-1]["name"])
            weekly_plan.append({"day": day, "day_number": di + 1, "meals": day_meals})

        targets = {"calories": round(tgt, 1), "protein": round(p, 1), "carbs": round(c, 1), "fat": round(f, 1)}
        return {"weekly_plan": weekly_plan, "targets": targets, "tdee": round(tdee, 1)}


# ------------------ module-level instance ------------------ #
diet_model = DietPlanDL(use_math_fallback=True)


# ------------------ CLI training helper ------------------ #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or test the DietPlanDL model.")
    parser.add_argument("--train", action="store_true", help="Train both networks (profile + ranker).")
    parser.add_argument("--profile_samples", type=int, default=50000)
    parser.add_argument("--profile_epochs", type=int, default=35)
    parser.add_argument("--ranker_profiles", type=int, default=8000)
    parser.add_argument("--foods_per_meal", type=int, default=40)
    parser.add_argument("--ranker_epochs", type=int, default=25)
    args = parser.parse_args()

    # Initialize data
    if not diet_model.load_data():
        raise SystemExit("Failed to load datasets.")

    if args.train:
        diet_model.train_all(
            profile_samples=args.profile_samples,
            profile_epochs=args.profile_epochs,
            ranker_profiles=args.ranker_profiles,
            foods_per_meal=args.foods_per_meal,
            ranker_epochs=args.ranker_epochs
        )
    else:
        # Try loading models and print a sample prediction
        if diet_model.load_models():
            tdee, tgt, (p, c, f), (bc, lc, dc, sc) = diet_model._predict_profile_targets(25, 0, 70, 175, 0, 2)
            print("Sample profile prediction:")
            print(f"TDEE: {tdee:.1f}  Target calories: {tgt:.1f}")
            print(f"Macros (g): protein={p:.1f}, carbs={c:.1f}, fat={f:.1f}")
            print(f"Meal cals: Breakfast={bc:.0f}, Lunch={lc:.0f}, Dinner={dc:.0f}, Snack={sc:.0f}")
        else:
            print("No trained models. Run with --train.")