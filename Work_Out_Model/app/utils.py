from __future__ import annotations
import math
import pandas as pd
from typing import Dict, List, Any, Tuple

CALORIES_COEFF = 3.5 / 200.0
DEFAULT_DURATION_MIN = 10
KCALS_PER_KG = 7700.0


def _s(x):
    v = "" if pd.isna(x) else str(x)
    return v


def calculate_calories(met: float, weight_kg: float, duration_min: int = DEFAULT_DURATION_MIN) -> float:
    return float(met * 3.5 * weight_kg / 200.0 * duration_min)


def filter_exercises(df: pd.DataFrame, preference: str, experience: str) -> pd.DataFrame:
    filtered = df.copy()
    equip = filtered["Equipment"].fillna("").astype(str)
    if preference and str(preference).lower() == "home":
        filtered = filtered[equip.str.lower().isin(["body only", "none", "no equipment"])].copy()
    if experience:
        diff = filtered["Difficulty"].fillna("").astype(str)
        filtered = filtered[diff.str.lower() == str(experience).lower()].copy()
    return filtered


def prioritize_by_goal(df: pd.DataFrame, goal: str) -> pd.DataFrame:
    met = pd.to_numeric(df["MET"], errors="coerce").fillna(0.0)
    if goal == "Fat Loss":
        return df.assign(_met=met).sort_values(by=["_met"], ascending=False).drop(columns=["_met"]) if "_met" in df.columns else df.assign(_met=met).sort_values(by=["_met"], ascending=False)
    if goal == "Muscle Gain":
        mech = df.get("Mechanics", pd.Series([""] * len(df))).fillna("").astype(str).str.lower().eq("compound").astype(int)
        diff_rank = df.get("Difficulty", pd.Series([""] * len(df))).fillna("").astype(str).str.lower().map({"beginner": 0, "intermediate": 1, "advanced": 2}).fillna(0).astype(int)
        return df.assign(is_compound=mech, diff_rank=diff_rank, _met=met).sort_values(by=["is_compound", "diff_rank", "_met"], ascending=[False, False, True]).drop(columns=["_met"])
    strength_mask = df.get("Type", pd.Series([""] * len(df))).fillna("").astype(str).str.contains("strength", case=False, na=False)
    cardio_mask = df.get("Type", pd.Series([""] * len(df))).fillna("").astype(str).str.contains("cardio", case=False, na=False)
    strength_bias = df[strength_mask]
    cardio_bias = df[cardio_mask]
    mixed = pd.concat([strength_bias.head(3), cardio_bias.head(3)])
    remainder = df.drop(mixed.index, errors="ignore")
    return pd.concat([mixed, remainder])


def build_split() -> Dict[str, List[str]]:
    return {
        "Day 1": ["Chest", "Triceps"],
        "Day 2": ["Back", "Biceps"],
        "Day 3": ["Legs"],
        "Day 4": ["Core", "Cardio"],
        "Day 5": ["Upper"],
        "Day 6": ["Full Body"],
    }


def pick_exercises(df: pd.DataFrame, muscles: List[str], n: int = 6) -> List[Dict[str, Any]]:
    if not len(df):
        return []
    pattern = "|".join([m for m in muscles if m]) or ".*"
    primary = df.get("Primary_Muscle", pd.Series([""] * len(df))).fillna("").astype(str)
    subset = df[primary.str.contains(pattern, case=False, na=False)]
    if subset.empty:
        subset = df
    picks = subset.head(n)
    results = []
    for _, row in picks.iterrows():
        instr = _s(row.get("Instructions"))
        instr = "" if instr.strip().lower() == "nan" else instr
        results.append({
            "Exercise_Name": row.get("Exercise_Name"),
            "Primary_Muscle": row.get("Primary_Muscle"),
            "Equipment": row.get("Equipment"),
            "Difficulty": row.get("Difficulty"),
            "Instructions": instr,
            "MET": float(row.get("MET", 0) or 0),
            "Type": row.get("Type"),
            "Mechanics": row.get("Mechanics"),
            "Level": row.get("Level"),
        })
    return results


def swap_alternatives(df: pd.DataFrame, current: Dict[str, Any], preference: str) -> List[Dict[str, Any]]:
    prim_series = df.get("Primary_Muscle", pd.Series([""] * len(df))).fillna("").astype(str).str.lower()
    diff_series = df.get("Difficulty", pd.Series([""] * len(df))).fillna("").astype(str).str.lower()
    cur_prim = _s(current.get("Primary_Muscle")).lower()
    cur_diff = _s(current.get("Difficulty")).lower()
    same = df[(prim_series == cur_prim) & (diff_series == cur_diff)]

    if preference and str(preference).lower() == "home":
        equip = same.get("Equipment", pd.Series([""] * len(same))).fillna("").astype(str).str.lower()
        same = same[equip.isin(["body only", "none", "no equipment"])].copy()
    if same.empty:
        mech_series = df.get("Mechanics", pd.Series([""] * len(df))).fillna("").astype(str).str.lower()
        mech = _s(current.get("Mechanics")).lower()
        alt = df[mech_series == mech].head(3)
        if alt.empty:
            # fallback by primary muscle only
            alt = df[prim_series == cur_prim].head(5)
    else:
        alt = same.head(5)
    out = []
    for _, r in alt.iterrows():
        instr = _s(r.get("Instructions"))
        instr = "" if instr.strip().lower() == "nan" else instr
        out.append({
            "Exercise_Name": r.get("Exercise_Name"),
            "Primary_Muscle": r.get("Primary_Muscle"),
            "Equipment": r.get("Equipment"),
            "Difficulty": r.get("Difficulty"),
            "Instructions": instr,
            "MET": float(r.get("MET", 0) or 0),
        })
    return out


def compute_pie_for_goal(goal: str) -> Tuple[List[str], List[int], List[str]]:
    labels = ["Cardio", "Strength", "Flexibility", "Core"]
    if goal == "Fat Loss":
        values = [45, 35, 5, 15]
    elif goal == "Muscle Gain":
        values = [15, 65, 5, 15]
    else:
        values = [25, 45, 10, 20]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f1c40f"]  # distinct
    return labels, values, colors


def project_monthly_progress(daily_calories: List[float], goal: str) -> Tuple[List[str], List[float], float, str]:
    # Build 30-day projection by repeating weekly pattern
    if not daily_calories:
        return [str(i) for i in range(1, 31)], [0.0] * 30, 0.0, "No data"
    week = daily_calories
    month_cals: List[float] = []
    for i in range(30):
        month_cals.append(float(week[i % len(week)]))
    total_month = float(sum(month_cals))

    if goal == "Fat Loss":
        kg_change = -(total_month / KCALS_PER_KG)
        summary = f"Estimated weight change: {kg_change:.2f} kg loss over 30 days"
    elif goal == "Muscle Gain":
        # Heuristic small positive gain bounded
        kg_change = min(0.8, 0.00006 * total_month)
        summary = f"Estimated weight change: +{kg_change:.2f} kg gain over 30 days"
    else:
        kg_change = 0.0
        summary = "Estimated weight change: ~0.00 kg (maintenance)"

    labels = [f"Day {i}" for i in range(1, 31)]
    return labels, month_cals, kg_change, summary


def generate_workout_plan(df: pd.DataFrame, user: Dict[str, Any], duration_min: int = DEFAULT_DURATION_MIN) -> Tuple[Dict[str, List[Dict[str, Any]]], float, Dict[str, Any]]:
    preference = user.get("Workout Preference", "Gym")
    experience = user.get("Experience Level", "Beginner")
    goal = user.get("Goal", "Maintain")
    weight = float(user.get("Weight", 70))

    df2 = filter_exercises(df, preference=preference, experience=experience)
    df2 = prioritize_by_goal(df2, goal=goal)

    split = build_split()
    plan: Dict[str, List[Dict[str, Any]]] = {}
    daily_calories = []

    for day, muscles in split.items():
        picks = pick_exercises(df2, muscles=muscles, n=6)
        for ex in picks:
            ex["calories"] = round(calculate_calories(ex.get("MET", 0.0), weight, duration_min))
            instructions = _s(ex.get("Instructions"))
            if not instructions or instructions.strip() == "":
                instructions = f"Perform {ex.get('Exercise_Name', 'this exercise')} for the specified duration. Focus on proper form and controlled movements."
            ex["Instructions"] = instructions
        plan[day] = picks
        daily_calories.append(sum(ex["calories"] for ex in picks))

    total_calories = sum(daily_calories)

    pie_labels, pie_values, pie_colors = compute_pie_for_goal(goal)
    month_labels, month_values, kg_change, progress_summary = project_monthly_progress(daily_calories, goal)

    chart_data = {
        "pie": {"labels": pie_labels, "values": pie_values, "colors": pie_colors},
        "bar": {"labels": list(split.keys()), "values": daily_calories},
        "monthly": {"labels": month_labels, "calories": month_values, "kgChange": kg_change},
        "summary": progress_summary,
    }

    from routes import get_video_for_muscle 
    for day, exercises in plan.items():
        for ex in exercises:
            primary = (
                ex.get("Primary_Muscle")
                or ex.get("BodyPart")
                or ""
            )
            ex["Video_URL"] = get_video_for_muscle(primary)
    return plan, total_calories, chart_data
