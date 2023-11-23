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
# from sklearn.preprocessing import LabelEncoder

# Load your dataset (replace 'your_dataset.csv' with your actual dataset)
df = pd.read_csv('final_test.csv')

# Display basic information about the dataset
print("Dataset Information:")
print(df.info())

# Display summary statistics of the dataset
print("\nSummary Statistics:")
print(df.describe())

# Drop rows with any NaN values
df = df.dropna()

'''
# Convert categorical values to numerical values for the model
category_column = 'size'

# Use LabelEncoder to convert text categories to numerical labels
label_encoder = LabelEncoder()
df[category_column + '_encoded'] = label_encoder.fit_transform(df[category_column])

# You can use the following for automated assignment of categories, 
# but I decided to manually assign values in order
# Display the mapping between original categories and encoded labels
print("Mapping of Original Categories to Encoded Labels:")
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(label_mapping)
'''

# Define a mapping dictionary
category_mapping = {'XXS': 0, 'S': 1, 'M': 2, 'L': 3, 'XL': 4, 'XXL': 5, 'XXXL': 6}

# Apply the mapping to the 'Category' column
df['size_encoded'] = df['size'].map(category_mapping)

# drop the original categorical column
df_mod = df.drop('size', axis=1)
df_mod = df_mod.drop('age', axis=1)

# # Visualize the data using a pair plot
# sns.pairplot(df_mod, hue='size_encoded')
# plt.title('Pair Plot')
# plt.show()

# Visualize the correlation matrix using a heatmap
# correlation_matrix = df_mod.corr()
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
# plt.title('Correlation Matrix Heatmap')
# plt.show()

# Split the data into features (X) and target variable (y)
X = df_mod.drop('size_encoded', axis=1)
y = df_mod['size_encoded']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Naive Bayes classification
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = naive_bayes_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Accuracy: {accuracy}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_rep)


'''
Classification Report:
              precision    recall  f1-score   support

           0       0.53      0.45      0.48      1984
           1       0.46      0.44      0.45      4376
           2       0.42      0.63      0.51      5960
           3       0.33      0.14      0.19      3521
           4       0.40      0.43      0.41      3820
           5       0.00      0.00      0.00        20
           6       0.80      0.67      0.73      4150

    accuracy                           0.48     23831
   macro avg       0.42      0.39      0.40     23831
weighted avg       0.49      0.48      0.47     23831

Additional Notes:
Fast Running

'''