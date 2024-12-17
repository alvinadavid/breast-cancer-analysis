# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv('Breast_train.csv')

# Define the features and target
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')

# Create a function to make predictions
def predict_cancer(mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness):
    input_data = np.array([[mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness]])
    prediction = model.predict(input_data)
    return prediction[0]

# Create a Flask app
from flask import Flask, request, render_template

app = Flask(__name__)

# Create a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Create a route for the prediction page
@app.route('/predict', methods=['POST'])
def predict():
    mean_radius = float(request.form['mean_radius'])
    mean_texture = float(request.form['mean_texture'])
    mean_perimeter = float(request.form['mean_perimeter'])
    mean_area = float(request.form['mean_area'])
    mean_smoothness = float(request.form['mean_smoothness'])
    
    prediction = predict_cancer(mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness)
    
    if prediction == 0:
        result = 'Benign'
    else:
        result = 'Malignant'
    
    return render_template('result.html', result=result)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
    
    
    
# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv('breast_cancer_data.csv')

# Define the features and target
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Create a function to make predictions
def predict_cancer(mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness):
    input_data = np.array([[mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness]])
    prediction = model.predict(input_data)
    return prediction[0]

# Test the function
print(predict_cancer(17.99, 10.38, 122.8, 1001, 0.1184))  # Should print 0 ( cancer)
print(predict_cancer(13.54, 14.36, 87.46, 566.3, 0.09779))  # Should print 1 (no cancer)