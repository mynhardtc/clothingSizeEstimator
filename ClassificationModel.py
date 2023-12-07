'''
This function is used to estimate the sie of clothing
we will base the estimation on:
- height
- weight
- age

this estimator makes use of Naive Bayes
'''

# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB



# from sklearn.preprocessing import LabelEncoder

def naive_bayes(df):
    # Split the data into features (X) and target variable (y)
    X = df.drop('size_encoded', axis=1)
    y = df['size_encoded']

    # Split the data into training and testing sets
    print('\nSplit data to test and train datasets')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply Naive Bayes classification
    naive_bayes_model = GaussianNB()
    naive_bayes_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = naive_bayes_model.predict(X_test)

    # Evaluate the model
    print('\nEvaluate the model')
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    print("Model Evaluation:")
    print(f"Accuracy: {accuracy}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(classification_rep)

    '''
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.59      0.29      0.39      1984
               1       0.48      0.51      0.49      4376
               2       0.44      0.68      0.54      5960
               3       0.32      0.08      0.12      3521
               4       0.41      0.40      0.41      3820
               5       0.00      0.00      0.00        20
               6       0.74      0.78      0.76      4150
    
        accuracy                           0.50     23831
       macro avg       0.43      0.39      0.39     23831
    weighted avg       0.49      0.50      0.47     23831
    
    Additional Notes:
    Long running
    '''
