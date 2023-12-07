# This file contains the procedure for exploratory data analysis

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Function to:
# Produce necessary statistics of the data
# Produce visual outputs to assess data

def produce_eda(file):
    df = pd.read_csv(file)

    print('\nMain Data Information\n')
    print(df.info(verbose=1))

    print('\nData Description and Summary\n')
    print(df.describe(include='all'))
    cols = df.columns.values.tolist()

    # Display information about missing values
    print("\nMissing Values Before Preprocessing:\n")
    print(df.isnull().sum())

    # drop nan values if there are any
    df = df.dropna()

    # show top 100 values in the dataset
    print('\nShow the top 100 rows of data\n')
    print(df.head(n=100))

    # Define a mapping dictionary
    # Since sizing is ordinal, specify ordered list
    category_mapping = {'XXS': 0, 'S': 1, 'M': 2, 'L': 3, 'XL': 4, 'XXL': 5, 'XXXL': 6}

    # Apply the mapping to the 'Category' column
    df['size_encoded'] = df['size'].map(category_mapping)
    df = df.drop('size', axis=1)

    # Visualize data in bar graph
    # Visualize the data using a pair plot for numerical data
    # print('\nStarting pair plot\n')
    # sns.pairplot(df, hue='size_encoded')
    # plt.title('Pair Plot')
    # plt.savefig('PairPlot.png')

    # check correlation between variables
    print(df.corr())

    # plot correlation heat map
    # print('\nStarting heatmap plot\n')
    # plt.figure(figsize=(16, 6))
    # heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
    # heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 18}, pad=12)
    # # save heatmap as .png file
    # # dpi - sets the resolution of the saved image in dots/inches
    # # bbox_inches - when set to 'tight' - does not allow the labels to be cropped
    # plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')

    """
                    weight       age    height  size_encoded
    weight        1.000000  0.068157  0.388551      0.792904
    age           0.068157  1.000000 -0.003044      0.177417
    height        0.388551 -0.003044  1.000000      0.264991
    size_encoded  0.792904  0.177417  0.264991      1.000000
    
    Observations:
    Weak correlation    age     : weight, height
    Medium correlation  weight  : height
                        size    : age, height
    strong correlation  size    : weight
    
    Use regression , multivariate
    
    """

    return df
