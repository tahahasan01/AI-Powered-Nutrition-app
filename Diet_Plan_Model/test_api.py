#!/usr/bin/env python3
"""
Test script for NutriFit Diet Plan Generator API
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:5000"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_diet_prediction():
    """Test the diet prediction endpoint"""
    print("\nTesting diet prediction...")
    
    test_data = {
        "age": 25,
        "gender": 0,  # Male
        "weight": 70,
        "height": 175,
        "goal": 0,  # Weight Loss
        "activity_level": 2  # Moderate
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict_diet",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Diet prediction successful!")
            print(f"Average daily calories: {data['summary']['average_daily_calories']}")
            print(f"Target daily calories: {data['summary']['target_daily_calories']}")
            print(f"Number of days: {len(data['weekly_plan'])}")
            
            print("\nFirst day meals:")
            first_day = data['weekly_plan'][0]
            for i, meal in enumerate(first_day['meals']):
                print(f"  {i+1}. {meal['food_name']} - {meal['calories']} kcal")
            
            return first_day['meals'][0]['food_name']  # Return first meal for swap test
        else:
            print(f"❌ Diet prediction failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Diet prediction error: {e}")
        return None

def test_meal_swap(meal_name):
    """Test the meal swap endpoint"""
    print(f"\nTesting meal swap for: {meal_name}")
    
    test_data = {
        "current_meal_name": meal_name,
        "goal": 0,  # Weight Loss
        "meal_type": "Breakfast"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/swap_meal",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Meal swap successful!")
            print(f"Found {len(data['alternatives'])} alternatives")
            
            for i, alt in enumerate(data['alternatives'][:3]):  # Show first 3
                print(f"  {i+1}. {alt['name']} - {alt['calories']} kcal")
            
            return True
        else:
            print(f"❌ Meal swap failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Meal swap error: {e}")
        return False

def test_meal_details(meal_name):
    """Test the meal details endpoint"""
    print(f"\nTesting meal details for: {meal_name}")
    
    test_data = {
        "meal_name": meal_name,
        "quantity": 100
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/get_meal_details",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Meal details successful!")
            print(f"Meal: {data['meal_name']}")
            print(f"Calories: {data['nutritional_breakdown']['calories']}")
            print(f"Protein: {data['nutritional_breakdown']['protein_g']}g")
            print(f"Carbs: {data['nutritional_breakdown']['carbohydrates_g']}g")
            print(f"Fat: {data['nutritional_breakdown']['fat_g']}g")
            return True
        else:
            print(f"❌ Meal details failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Meal details error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing NutriFit Diet Plan Generator API")
    print("=" * 50)
    
    # Test health check
    if not test_health_check():
        print("\n❌ Health check failed. Make sure the Flask app is running.")
        return
    
    # Test diet prediction
    first_meal = test_diet_prediction()
    if not first_meal:
        print("\n❌ Diet prediction failed. Stopping tests.")
        return
    
    # Test meal swap
    test_meal_swap(first_meal)
    
    # Test meal details
    test_meal_details(first_meal)
    
    print("\n" + "=" * 50)
    print("🎉 All tests completed!")
    print("\nYou can now:")
    print("1. Open http://localhost:5000 in your browser")
    print("2. Fill out the form to generate meal plans")
    print("3. Use the interactive features like meal swapping")

if __name__ == "__main__":
    main()
