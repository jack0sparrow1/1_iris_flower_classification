# main.py

# Add src/ to the Python path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import preprocessing and training modules
from data_preprocessing import preprocess
from train import train_model, evaluate_model, save_model

# Step 1: Preprocess the data
X_train, X_test, y_train, y_test = preprocess('data/iris.data')

# Step 2: Train the model
model = train_model(X_train, y_train)

# Step 3: Evaluate the model
accuracy, report = evaluate_model(model, X_test, y_test)
print("Model Accuracy:", accuracy)
print("Classification Report:\n", report)

# Step 4: Save the model
save_path = 'models/SVM.pickle'
save_model(model, save_path)
print(f"Model saved to {save_path}")

from predict import load_model, make_predictions
import numpy as np

model = load_model('models/SVM.pickle')
X_new = np.array([
    [3, 2, 1, 0.2],
    [4.9, 2.2, 3.8, 1.1],
    [5.3, 2.5, 4.6, 1.9]
])
predictions = make_predictions(model, X_new)
print("Predictions for new data:", predictions)
