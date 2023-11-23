import joblib

# Save the trained model
joblib.dump(knn_model, 'knn_model.joblib')

# Load the model in the production environment
loaded_model = joblib.load('knn_model.joblib')


def make_predictions(new_data):
    # Load the trained model
    model = joblib.load('knn_model.joblib')

    # Preprocess the new data if needed
    # ...

    # Make predictions
    predictions = model.predict(new_data)
    return predictions