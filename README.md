 Flask-Student-Progress-Prediction-AppDescription
This Flask application uses machine learning to predict student performance based on various factors such as study habits, attendance, parental involvement, and more. The application is built with a deep learning model that is trained on historical student performance data. Once trained, the model predicts a student's final exam score based on provided input features.

The model is built using TensorFlow and Keras, and the application allows users to input data through a web interface, which is then processed and used to predict the student's performance. The application also includes the functionality to train and save the model, so the user can update predictions after model training.

Features
Data preprocessing, including handling categorical and numerical features

Training and saving the model with TensorFlow and Keras

Predicting exam scores based on input data

Web interface to train the model and input new data

Preprocessing of inputs with label encoding and scaling for accurate predictions

Early stopping during training to prevent overfitting

Model saved for future predictions without retraining

Technologies Used
Python 3

Flask

TensorFlow & Keras (Deep Learning)

Pandas (Data Handling)

Scikit-learn (Preprocessing and Splitting Data)

HTML and Flask templates (Web Interface)

How the Application Works
Training the Model:

The application uses a dataset containing various student performance factors (e.g., study hours, attendance, parental involvement, etc.) to train a deep learning model. The model is a neural network with multiple layers, including residual connections to improve accuracy.

Once the model is trained, it is saved to a file (stu_model.keras) for future use.

The model is trained using the Huber loss function and Adam optimizer with early stopping for better convergence.

Prediction:

Once the model is trained, the user can input new student data, which will be preprocessed and passed through the model to predict the student’s final exam score.

The user is prompted to enter values for various features such as "Hours Studied," "Parental Involvement," "Attendance," etc.

After the data is entered and preprocessed (categorical values are encoded, numerical values are scaled), the model makes a prediction for the student’s final score.

Project Structure
app.py (Flask application to handle training and prediction)

StudentPerformanceFactors.csv (Dataset containing features like hours studied, attendance, etc.)

stu_model.keras (Saved model after training)

How to Run the Application
Install Dependencies:

Make sure Python 3 is installed

Install the necessary Python packages using pip:

nginx
Copy
Edit
pip install flask tensorflow pandas scikit-learn
Run the Application:

Ensure your dataset (StudentPerformanceFactors.csv) is in the same directory as your Flask app

Start the Flask application by running the following command:

nginx
Copy
Edit
python app.py
The app will start a local server. You can access the app at http://localhost:5000.

Training the Model:

To train the model, navigate to http://localhost:5000/train in your browser.

The application will preprocess the data, train the model, and save it for future use.

Predicting Student Performance:

After training, you can enter new data in the terminal when prompted by the get_user_input() function.

The app will preprocess the input, and the model will return a predicted final grade based on the provided features.

API Usage
The application provides a simple API endpoint for training the model:

POST /train

Starts training the model on the provided dataset.

Response:

json
Copy
Edit
{ "message": "Model trained and saved successfully." }
Input for Prediction:
The user will be prompted to input values for various features (e.g., Hours Studied, Attendance, etc.). After the input is processed, the model will predict the final grade.

Example Input
When prompted for input, the application asks for features like:

Hours Studied: 5

Attendance: 90%

Parental Involvement: High

Motivation Level: Medium

Previous Scores: 75

Family Income: Middle Class

Teacher Quality: Good

The user inputs these values in the terminal, and the model predicts the student’s final grade.

Future Enhancements
Improve Model Performance: Experiment with different neural network architectures, hyperparameters, and optimization techniques to enhance prediction accuracy.

Feature Expansion: Add more features such as student’s emotional well-being, peer influence, and other social factors that could impact performance.

Web Interface: Build a more user-friendly web interface for data input and result display.

Model Fine-Tuning: Fine-tune the model periodically with new data for better predictions.
