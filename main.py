from flask import Flask, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping

app = Flask(__name__)

# Load dataset once at the start
filename = 'StudentPerformanceFactors.csv'
data = pd.read_csv(filename)

features = ['Hours_Studied', 'Attendance', 'Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
            'Sleep_Hours', 'Previous_Scores', 'Motivation_Level', 'Internet_Access', 'Tutoring_Sessions',
            'Family_Income', 'Teacher_Quality', 'School_Type', 'Peer_Influence', 'Physical_Activity',
            'Learning_Disabilities', 'Parental_Education_Level', 'Distance_from_Home', 'Gender']

# Function to preprocess data
def preprocess_data(data):
    X = data[features].copy()
    y = data['Exam_Score']

    categorical_features = ['Parental_Involvement', 'Access_to_Resources', 'Motivation_Level', 'Internet_Access', 
                            'Family_Income', 'Teacher_Quality', 'School_Type', 'Peer_Influence', 'Learning_Disabilities', 
                            'Parental_Education_Level', 'Gender', 'Distance_from_Home']

    label_encoders = {}
    for col in categorical_features:
        X[col] = X[col].astype(str).str.lower().fillna('unknown')
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    numerical_features = list(set(features) - set(categorical_features))
    X.loc[:, numerical_features] = X[numerical_features].apply(pd.to_numeric, errors='coerce').astype(float).fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# Function to build the model
def build_model(input_shape):
    inputs = Input(shape=(input_shape,))
    x = Dense(128, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    residual = Dense(256, activation='relu')(x)
    residual = BatchNormalization()(residual)
    residual = Dense(256, activation='relu')(residual)
    residual = BatchNormalization()(residual)
    residual = Dropout(0.3)(residual)

    x = Add()([x, residual])

    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)

    outputs = Dense(1, activation='relu')(x)

    model = tf.keras.models.Model(inputs, outputs)
    model.compile(optimizer=Adam(), loss=Huber(), metrics=['mae'])
    return model

@app.route('/train', methods=['GET'])
def train_model():
    try:
        # Preprocess the loaded data
        X_scaled, y = preprocess_data(data)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

        # Build model
        model = build_model(len(features))

        # Train model
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        history = model.fit(X_train, y_train, epochs=500, batch_size=10, validation_split=0.2, verbose=1, callbacks=[early_stopping])

        # Save model
        model.save('stu_model.keras')

        return jsonify({'message': 'Model trained and saved successfully.'})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
