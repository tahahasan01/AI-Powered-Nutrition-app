# NutriFitAI - Unified Fitness Assistant 🏋️‍♂️🍎

A comprehensive AI-powered fitness platform that combines personalized diet planning and workout generation in a single unified application.

## 🚀 Features

### 🍽️ Diet Planning Module
- **Personalized Meal Plans**: 7-day meal plans with 4 meals per day
- **Smart Nutrition**: TDEE calculation using Mifflin-St Jeor equation
- **Goal-Oriented**: Weight Loss, Muscle Gain, or Maintenance
- **Meal Swapping**: Interactive meal alternatives with similar nutritional profiles
- **Pakistani Cuisine**: Local ingredients and dietary preferences
- **Visual Analytics**: Macronutrient charts and progress projections

### 💪 Workout Planning Module
- **6-Day Workout Splits**: Comprehensive exercise routines
- **Equipment-Based**: Home vs Gym workout options
- **Experience Levels**: Beginner to Intermediate adaptations
- **MET-Based Calculations**: Accurate calorie burning estimates
- **Exercise Alternatives**: Smart swapping with similar muscle groups
- **Progress Tracking**: Monthly projections and goal alignment

## 🏗️ Architecture

### Unified Flask Application
- **Single Entry Point**: `Diet Plan Model/app.py`
- **Blueprint Integration**: Workout module integrated as Flask blueprint
- **Session Management**: Seamless data sharing between modules
- **Modular Design**: Both modules remain independent and maintainable

### Machine Learning Models
- **Diet Model**: MLPRegressor for nutrition prediction
- **Workout Model**: Pre-trained pickle model for exercise recommendations
- **Data Sources**: CSV datasets with comprehensive food and exercise data

## 📁 Project Structure

```
NutriFitAI/
├── Diet Plan Model/                 # Main application directory
│   ├── app.py                      # 🚀 Main Flask application (START HERE)
│   ├── diet_model.py              # AI diet planning logic
│   ├── cleaned_foods_dataset.csv  # Food database
│   ├── cleaned_snacks_dataset.csv # Snack database
│   ├── requirements.txt           # Python dependencies
│   └── templates/                 # Frontend templates
│       ├── NutriFit_HomePage.html # Landing page
│       ├── index.html             # Diet form
│       └── result.html            # Diet results + workout button
│
├── Work Out Model/                 # Workout module (integrated)
│   ├── app/                       # Workout Flask blueprint
│   │   ├── routes.py              # Workout API routes
│   │   ├── utils.py               # Workout logic
│   │   ├── templates/             # Workout templates
│   │   └── static/                # CSS/JS assets
│   ├── workout_model.pkl          # Trained workout model
│   ├── workoutdata_with_estimated_met.csv # Exercise database
│   └── requirements.txt           # Workout dependencies
│
├── test_integration.py            # Integration test script
└── README.md                      # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- pip package manager

### Installation & Setup

1. **Navigate to the project directory:**
   ```bash
   cd NutriFitAI
   ```

2. **Install dependencies:**
   ```bash
   cd "Diet Plan Model"
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Open your browser:**
   ```
   http://localhost:5000
   ```

### 🧪 Testing Integration
```bash
python test_integration.py
```

## 🔄 User Workflow

### Complete User Journey
1. **Homepage** (`/`) → NutriFit landing page with login/signup
2. **Diet Form** (`/diet-form`) → User inputs personal details
3. **Diet Results** (`/result`) → Personalized meal plan with charts
4. **Workout Button** → "Generate Workout Plan 💪" button
5. **Workout Form** (`/workout/`) → Pre-populated with user data
6. **Workout Results** → 6-day exercise plan with progress tracking

### Session Data Sharing
The application automatically shares user data between modules:
- **Age, Gender, Weight, Height**: Physical characteristics
- **Goal**: Fat Loss, Muscle Gain, or Maintain
- **Activity Level**: Sedentary to Very Active

## 🎯 Key Features

### Diet Planning
- **TDEE Calculation**: Accurate calorie needs based on user profile
- **Macro Distribution**: Goal-specific protein/carb/fat ratios
- **Meal Variety**: 7-day rotation with different meals
- **Interactive Charts**: Macronutrient pie charts and progress projections
- **Meal Swapping**: Real-time alternatives with nutritional matching

### Workout Planning
- **Equipment Filtering**: Home vs Gym workout options
- **Experience Adaptation**: Beginner to Intermediate difficulty
- **Goal Alignment**: Fat Loss, Muscle Gain, or Maintenance focus
- **Exercise Database**: Comprehensive exercise library with MET values
- **Progress Tracking**: Monthly weight change projections

## 🔧 Technical Details

### Flask Integration
- **Blueprint Registration**: Workout module registered with `/workout` prefix
- **Session Management**: Flask sessions for data persistence
- **Error Handling**: Graceful fallbacks for missing modules
- **Path Management**: Dynamic path resolution for data files

### Machine Learning
- **Diet Model**: Scikit-learn MLPRegressor for nutrition prediction
- **Workout Model**: Pre-trained pickle model for exercise recommendations
- **Data Processing**: Pandas for data manipulation and analysis
- **Feature Engineering**: Goal-based scoring and filtering

### Frontend
- **Responsive Design**: Mobile-friendly interface
- **Chart.js Integration**: Interactive charts and visualizations
- **Modern UI**: Clean, professional design with Poppins font
- **Real-time Updates**: Dynamic content loading and form pre-population

## 📊 API Endpoints

### Diet Planning
- `GET /` → Homepage
- `GET /diet-form` → Diet input form
- `POST /generate-plan` → Generate meal plan
- `POST /predict_diet` → API diet prediction
- `POST /swap_meal` → Get meal alternatives
- `POST /get_meal_details` → Detailed meal information

### Workout Planning
- `GET /workout/` → Workout input form
- `POST /workout/generate_plan` → Generate workout plan
- `POST /workout/swap_exercise` → Get exercise alternatives

## 🎨 User Interface

### Design Philosophy
- **Unified Branding**: Consistent NutriFit Pakistan theme
- **Intuitive Navigation**: Clear user flow from diet to workout
- **Visual Feedback**: Charts, progress indicators, and status messages
- **Responsive Layout**: Works on desktop, tablet, and mobile devices

### Color Scheme
- **Primary**: #2ecc71 (Green) - Health and growth
- **Secondary**: #3498db (Blue) - Trust and reliability
- **Accent**: #f39c12 (Orange) - Energy and motivation
- **Background**: #f8fafc (Light Gray) - Clean and modern

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Data File Missing**: Check that CSV files are in correct directories
3. **Blueprint Not Loading**: Verify workout model files are present
4. **Session Issues**: Clear browser cache and cookies

### Debug Mode
```bash
python app.py
# Check console output for integration status
```

## 🚀 Future Enhancements

- **User Authentication**: Login/signup system
- **Progress Tracking**: Long-term goal monitoring
- **Social Features**: Community challenges and sharing
- **Mobile App**: React Native or Flutter implementation
- **Advanced Analytics**: Machine learning insights and recommendations

## 📝 License

This project is open source. Feel free to modify and distribute according to your needs.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

**NutriFitAI** - Empowering healthy living with AI-driven nutrition and fitness planning! 🏋️‍♂️🍎✨
