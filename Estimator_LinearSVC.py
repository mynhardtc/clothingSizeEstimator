'''
This function is used to estimate the sie of clothing
we will base the estimation on:
- height
- weight
- age

this estimator makes use of Naive Bayes
'''

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# from sklearn.preprocessing import LabelEncoder

# Load your dataset (replace 'your_dataset.csv' with your actual dataset)
df = pd.read_csv('final_test.csv')

# Display basic information about the dataset
print("Dataset Information:")
print(df.info())

# Display summary statistics of the dataset
print("\nSummary Statistics:")
print(df.describe())

# Display information about missing values
print("Missing Values Before Preprocessing:")
print(df.isnull().sum())

# Drop rows with any NaN values
df = df.dropna()

# Display information after handling missing values
print("\nMissing Values After Preprocessing:")
print(df.isnull().sum())

# Define a mapping dictionary
category_mapping = {'XXS': 0, 'S': 1, 'M': 2, 'L': 3, 'XL': 4, 'XXL': 5, 'XXXL': 6}

# Apply the mapping to the 'Category' column
df['size_encoded'] = df['size'].map(category_mapping)

# drop the original categorical column
df_mod = df.drop('size', axis=1)
df_mod = df_mod.drop('age', axis=1)

# Visualize the data using a pair plot
# print('\nStarting pair plot')
# sns.pairplot(df_mod, hue='size_encoded')
# plt.title('Pair Plot')
# plt.show()

# Visualize the correlation matrix using a heatmap
# print('\nStarting correlation matrix')
# correlation_matrix = df_mod.corr()
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
# plt.title('Correlation Matrix Heatmap')
# plt.show()

# Split the data into features (X) and target variable (y)
X = df_mod.drop('size_encoded', axis=1)
y = df_mod['size_encoded']

# Split the data into training and testing sets
print('\nSplit data to test and train datasets')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for SVC)
print('\nStandardise dataset')
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a Linear Support Vector Classifier
linear_svc_model = SVC(kernel='linear')

# Train the model
print('\nTrain model')
linear_svc_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
print('\nMake Predictions')
y_pred = linear_svc_model.predict(X_test_scaled)

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

