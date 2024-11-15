"""
Following are utility functions created for listing files in a directory and generate a usable key when random values are present in column names

"""
import os
import re
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

"""
Column Names
,id,
Name,
Gender,
Age,
City,
Working Professional or Student,
Profession,
Academic Pressure,
Work Pressure,
CGPA,
Study Satisfaction,
Job Satisfaction,
Sleep Duration,
Dietary Habits,
Degree,
Have you ever had suicidal thoughts ?,
Work/Study Hours,
Financial Stress,
Family History of Mental Illness,
Depression

"""

STR_COLS = [
    'Gender', # Cleaned
    'City', #Cleaned
    'Working Professional or Student', # Clean data
    'Profession', # Minor erros need not fix as statistically numbers are not great
    'Dietary Habits', # minor issues not statistically relevant
    'Degree',# minor issues not statistically relevant
    'Have you ever had suicidal thoughts ?', # Data  is clean
    'Family History of Mental Illness', #Clean
]

ALLOWED_COLS = [
    'id',
    # 'Name',
    'Gender', # Cleaned
    'Age', # Cleand
    'City', #Cleaned
    'Working Professional or Student', # Clean data
    'Profession', # Minor erros need not fix as statistically numbers are not great
    'Academic Pressure', # Data is clean
    'Work Pressure', # Data is clean
    'CGPA', # Cleaned
    'Study Satisfaction', #data is clean
    'Job Satisfaction', #data is clean
    'Sleep Duration', # Cleaned
    'Dietary Habits', # minor issues not statistically relevant
    'Degree',# minor issues not statistically relevant
    'Have you ever had suicidal thoughts ?', # Data  is clean
    'Work/Study Hours', # Clean
    'Financial Stress', # clean
    'Family History of Mental Illness', #Clean
    'Depression',
]

DTYPE_DICT = {   
    'id':int,
    'Depression': str ,
    'CGPA': str,
    'Sleep Duration':str
}

CLEANED_DTYPES = {
    'Age' :float, 
    'City':str, 
    'Working Professional or Student':str, 
    'Profession' : str, 
    'Academic Pressure':float, 
    'Work Pressure' : float, 
    'CGPA' :float, 
    'Study Satisfaction' : float, 
    'Job Satisfaction' : float, 
    'Sleep Duration' : float, 
    'Dietary Habits' :str , 
    'Degree':str,
    'Have you ever had suicidal thoughts ?' :str, 
    'Work/Study Hours' :float , 
    'Financial Stress' : float 
}
    
def slugify(s):
    s = s.lower().strip()
    s = re.sub(r'[^A-Za-z0-9]+', '-', string=s)
    return s

def list_files(directory):
    absolute_paths = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            absolute_paths.append(os.path.abspath(filepath))
    return absolute_paths
# print(f)


def load_trimaing_data() :
    train_list = list_files('../data/processed/')
    df_list = []
 
    for f in train_list:
        if not('train' in f):
            continue

        d = pd.read_csv(f,
            usecols=ALLOWED_COLS,  
            dtype= CLEANED_DTYPES)

        df_list.append(d)
            # Combine the list of dataframes
    df =  pd.concat(df_list)
    # df['id'] = df['id'].to_numpy()
    return df



column_encoder = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(),  STR_COLS) 
    ],
)



