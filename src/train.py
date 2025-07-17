import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def train_model(X_train, y_train):
    """
    Trains an SVM classifier on the given training data.
    """
    model = SVC()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on test data and prints accuracy and classification report.
    """
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return acc, report

def save_model(model, filename):
    """
    Saves the trained model to the given filename using pickle.
    """
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
