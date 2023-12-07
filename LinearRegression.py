# Predict the size of clothing using multiple regression technique

# import required packages

import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


def multiple_regression(df):
    X = df.drop('size_encoded', axis=1)
    y = df['size_encoded']

    # Split the data into training and testing sets
    print('\nSplit data to test and train datasets')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit model
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)

    # Make predictions on the test set
    print('\nMake Predictions')
    y_pred = regr.predict(X_test)

    # Evaluate the model
    print('\nEvaluate the model')
    error = mean_squared_error(y_test, y_pred)
    accuracy = r2_score(y_test, y_pred)

    print("Model Evaluation:")
    print(f"Accuracy: {accuracy}")
    print("\nMean squared error:")
    print(error)

    """
    Evaluate the model
    Model Evaluation:
    Accuracy: 0.6520328792959561
    
    Mean squared error:
    1.1940263053194162
    
    Overall the regression model peforms better than the classification model
    The dependent variable increases with an increase in age, height and weight
    """
