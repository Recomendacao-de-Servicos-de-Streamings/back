import pandas as pd
from os import listdir
import sklearn


print("Dependencies test: OK")


def lambda_handler(event, context):
    df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    print(df)
    listdir()
    print(listdir('..'))
    print(listdir('.'))
    print(listdir('datasets'))
    print(listdir('datasets/'))
    
    # read tags.csv from datasets folder
    df = pd.read_csv('datasets/tags.csv')
    print(df)

    return {
        'statusCode': 200,
        'body': 'Dependencies test: OK'
    }
