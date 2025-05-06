# Load and use the model
model = load_model('stu_model.keras')

def get_user_input():
    user_input = []
    for feature in features:
        value = input(f"Enter value for {feature}: ")
        try:
            if feature in categorical_features:
                value = value.lower()
                if value in label_encoders[feature].classes_:
                    value = label_encoders[feature].transform([value])[0]  # Encode categorical inputs
                else:
                    print(f"Warning: '{value}' not recognized. Assigning 'unknown' category.")
                    value = label_encoders[feature].transform(['unknown'])[0]  # Assign known 'unknown' category
            else:
                value = float(value)
        except ValueError:
            print(f"Invalid input for {feature}. Please enter a valid number or category.")
            return None
        user_input.append(value)
    return user_input

user_input = get_user_input()
if user_input:
    scaled_input = scaler.transform([user_input])
    predicted_grade = model.predict(scaled_input)[0][0]
    print(f"Predicted final grade: {predicted_grade:.2f}")