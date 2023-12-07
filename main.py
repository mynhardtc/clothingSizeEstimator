# Create an application to estimate the clothing size based on height, weight and age.
# create a separate function for doing the estimation

# Import required for application
from EDA import *
from LinearRegression import *
from ClassificationModel import *

file = 'final_test.csv'

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = produce_eda(file)
    multiple_regression(df)
    naive_bayes(df)

