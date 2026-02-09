# NutriFit Pakistan - Intelligent Diet Plan Generator

An AI-powered diet plan generator that creates personalized meal plans using MLP (Multi-Layer Perceptron) models and Flask backend. The system recommends daily meal plans based on user input including age, gender, weight, height, fitness goals, and activity level.

## Features

- **Personalized Meal Plans**: Generate daily meal plans with 3 meals + 1 snack
- **Smart Meal Swapping**: Get alternative meal suggestions with similar nutritional profiles
- **Nutrition Tracking**: Detailed macro and micronutrient breakdown
- **Interactive Charts**: Visualize weekly macros and projected progress
- **Pakistani Cuisine Focus**: Uses Pakistani and global food datasets
- **Goal-Oriented Plans**: Tailored for Weight Loss, Muscle Gain, or Maintenance

## Tech Stack

- **Backend**: Flask (Python)
- **ML Model**: Scikit-learn MLPRegressor
- **Frontend**: HTML5, CSS3, JavaScript, Chart.js
- **Data Processing**: Pandas, NumPy
- **API**: RESTful endpoints with JSON responses

## Installation

1. **Clone or download the project files**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure you have the required data files**:
   - `cleaned_foods_dataset.csv`
   - `cleaned_snacks_dataset.csv`

## Usage

### Running the Application

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

3. **Fill in the form** with your details:
   - Age (1-120)
   - Gender (Male/Female)
   - Weight in kg (30-300)
   - Height in cm (100-250)
   - Fitness Goal (Weight Loss/Muscle Gain/Maintain)
   - Activity Level (Sedentary to Very Active)

4. **Generate your meal plan** and explore the interactive features!

### API Endpoints

The application provides several REST API endpoints:

#### 1. Generate Diet Plan
```http
POST /predict_diet
Content-Type: application/json

{
    "age": 25,
    "gender": 0,
    "weight": 70,
    "height": 175,
    "goal": 0,
    "activity_level": 2
}
```

**Response**:
```json
{
    "daily_plan": {
        "meals": [
            {
                "meal_type": "Breakfast",
                "food_name": "Oatmeal with Milk",
                "quantity_grams": 150,
                "calories": 320,
                "protein_g": 12.5,
                "carbohydrates_g": 55.2,
                "fats_g": 8.1,
                "sugar_g": 12.3
            }
        ]
    },
    "summary": {
        "total_daily_calories": 1850,
        "total_protein_g": 120,
        "total_carbohydrates_g": 220,
        "total_fats_g": 65,
        "target_calories": 1850,
        "nutritional_balance": "Aligned with goal"
    }
}
```

#### 2. Swap Meal
```http
POST /swap_meal
Content-Type: application/json

{
    "current_meal_name": "Grilled Chicken",
    "goal": 1,
    "meal_type": "Lunch"
}
```

#### 3. Get Meal Details
```http
POST /get_meal_details
Content-Type: application/json

{
    "meal_name": "Brown Rice",
    "quantity": 100
}
```

#### 4. Health Check
```http
GET /health
```

## Data Structure

### Input Parameters
- **Age**: Integer (1-120)
- **Gender**: 0 (Male), 1 (Female)
- **Weight**: Float (30-300 kg)
- **Height**: Float (100-250 cm)
- **Goal**: 0 (Weight Loss), 1 (Muscle Gain), 2 (Maintain)
- **Activity Level**: 0-4 (Sedentary to Very Active)

### Output Structure
Each meal plan includes:
- **4 meals per day**: Breakfast, Lunch, Dinner, Snack
- **Nutritional data**: Calories, Protein, Carbs, Fats, Sugar, Fiber
- **Quantity information**: Grams/ml per serving
- **Goal alignment**: Calorie targets based on TDEE calculations

## Nutrition Logic

### Weight Loss (Goal = 0)
- **Calorie Deficit**: 15-20% below TDEE
- **Macros**: High protein (30%), moderate fats (35%), low carbs (35%)
- **Focus**: Lean proteins, vegetables, minimal processed foods

### Muscle Gain (Goal = 1)
- **Calorie Surplus**: 10-15% above TDEE
- **Macros**: High protein (25%), high carbs (45%), moderate fats (30%)
- **Focus**: Protein-rich foods, complex carbs, healthy fats

### Maintenance (Goal = 2)
- **Calorie Balance**: At TDEE level
- **Macros**: Balanced distribution (25% protein, 40% carbs, 35% fats)
- **Focus**: Balanced nutrition, variety

## Features in Detail

### Interactive Charts
- **Macro Distribution**: Pie chart showing protein/carbs/fat breakdown
- **Progress Projection**: Line chart showing estimated weekly weight changes

### Meal Swapping
- **Smart Alternatives**: Find meals with similar calorie and macro profiles
- **Goal Alignment**: Alternatives are filtered based on fitness goals
- **Real-time Updates**: Instantly update meal plans with new selections

### Data Validation
- **Input Validation**: Comprehensive checks for all user inputs
- **Health Safety**: Prevents unrealistic calorie outputs
- **Error Handling**: Graceful error messages and fallbacks

## File Structure

```
Diet Plan Model/
├── app.py                          # Flask application
├── diet_model.py                   # MLP model and meal logic
├── requirements.txt                # Python dependencies
├── cleaned_foods_dataset.csv      # Main food database
├── cleaned_snacks_dataset.csv     # Snack database
├── templates/
│   ├── index.html                 # Main form page
│   └── result.html                # Results with charts
└── README.md                      # This file
```

## Customization

### Adding New Foods
1. Update the CSV files with new food items
2. Ensure columns match the expected format:
   - Food Name, Calories, Carbohydrates (g), Sugars (g), Fat (g), Protein (g), Fiber (g)
   - Category, Meal_Type, is_snack

### Modifying Nutrition Logic
Edit the `calculate_macro_targets()` method in `diet_model.py` to adjust macro ratios for different goals.

### Styling Changes
Modify the CSS variables in the HTML templates to change colors, fonts, and layout.

## Troubleshooting

### Common Issues

1. **"No meal plan generated"**
   - Check that CSV files are present and readable
   - Verify input values are within valid ranges

2. **"Module not found" errors**
   - Run `pip install -r requirements.txt`
   - Ensure Python 3.7+ is installed

3. **Charts not displaying**
   - Check internet connection (Chart.js CDN)
   - Verify JavaScript is enabled in browser

### Performance Tips

- The model loads all data into memory for fast lookups
- For large datasets, consider implementing pagination
- Cache frequently accessed meal combinations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source. Feel free to modify and distribute according to your needs.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the API documentation
3. Test with the `/health` endpoint
4. Check server logs for detailed error messages

---

**NutriFit Pakistan** - Empowering healthy eating with AI-driven meal planning! 🍽️🤖
