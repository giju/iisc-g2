"""
Following are utility functions created for listing files in a directory and generate a usable key when random values are present in column names

"""

import os
import re
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from joblib import load

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


EXEC_MODE = "v1"


base = os.path.dirname(__file__)

model = load(os.path.join(base,f"../models/logistic-{EXEC_MODE}.joblib"))


# base = os.path.dirname(__file__)
# model_path =  os.path('../models/catboost_(students)_model.pkl')  # Update with your model path
# with open(model_path, 'rb') as model_file:
#     model = pickle.load(model_file)

load_dotenv()
client = OpenAI()


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


ALLOWED_COLS = [
    "id",
    # 'Name',
    "Gender",  # Cleaned
    "Age",  # Cleand
    "City",  # Cleaned
    "Working Professional or Student",  # Clean data
    "Profession",  # Minor erros need not fix as statistically numbers are not great
    "Academic Pressure",  # Data is clean
    "Work Pressure",  # Data is clean
    "CGPA",  # Cleaned
    "Study Satisfaction",  # data is clean
    "Job Satisfaction",  # data is clean
    "Sleep Duration",  # Cleaned
    "Dietary Habits",  # minor issues not statistically relevant
    "Degree",  # minor issues not statistically relevant
    "Have you ever had suicidal thoughts ?",  # Data  is clean
    "Work/Study Hours",  # Clean
    "Financial Stress",  # clean
    "Family History of Mental Illness",  # Clean
    "Depression",
]

DTYPE_DICT = {"id": int, "Depression": str, "CGPA": str, "Sleep Duration": str}
NAME_MAP = {
    "Working Professional or Student": "Profession_Status",
    "Academic Pressure": "Academic_Pressure",
    "Work Pressure": "Work_Pressure",
    "Study Satisfaction": "Study_Satisfaction",
    "Job Satisfaction": "Job_Satisfaction",
    "Sleep Duration": "Sleep_Duration",
    "Dietary Habits": "Dietary_Habits",
    "Have you ever had suicidal thoughts ?": "Suicidal_Thoughts",
    "Work/Study Hours": "Work_Study_Hours",
    "Financial Stress": "Financial_Stress",
    "Family History of Mental Illness": "Family_History",
}


CLEANED_DTYPES = {
    "id": int,
    "Gender": str,
    "Age": float,
    "City": str,
    "Profession_Status": str,
    "Profession": str,
    "Academic_Pressure": float,
    "Work_Pressure": float,
    "CGPA": float,
    "Study_Satisfaction": float,
    "Job_Satisfaction": float,
    "Sleep_Duration": float,
    "Dietary_Habits": str,
    "Degree": str,
    "Suicidal_Thoughts": str,
    "Work_Study_Hours": float,
    "Financial_Stress": float,
    "Family_History": str,
    "Depression": float,
}


if EXEC_MODE == "v2":
    CLEANED_DTYPES = {
        "id": int,
        "Gender": str,
        "Age": float,
        "City": str,
        "Profession_Status": int,
        "Profession": str,
        "Academic_Pressure": float,
        "Work_Pressure": float,
        "CGPA": float,
        "Study_Satisfaction": float,
        "Job_Satisfaction": float,
        "Sleep_Duration": float,
        "Dietary_Habits": float,
        "Degree": str,
        "Suicidal_Thoughts": int,
        "Work_Study_Hours": float,
        "Financial_Stress": float,
        "Family_History": int,
        "Depression": float,
    }


CLEANED_COLS = list(CLEANED_DTYPES.keys())
STR_COLS = list(filter(lambda x: CLEANED_DTYPES[x] == str, CLEANED_COLS))


column_encoder = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(), STR_COLS)],
)


def slugify(s):
    s = s.lower().strip()
    s = re.sub(r"[^A-Za-z0-9]+", "-", string=s)
    return s


def list_files(directory):
    absolute_paths = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            absolute_paths.append(os.path.abspath(filepath))
    return absolute_paths


# print(f)


def load_training_data(version=EXEC_MODE):
    train_list = list_files(f"../data/processed/{version}/")
    df_list = []

    for f in train_list:
        if not ("train" in f):
            continue

        d = pd.read_csv(f, dtype=CLEANED_DTYPES)

        df_list.append(d)
        # Combine the list of dataframes
    df = pd.concat(df_list)
    # df['id'] = df['id'].to_numpy()
    return df


def get_options():

    base = os.path.dirname(__file__)

    print(f"Base {base}")

    df_city = pd.read_csv(os.path.join(base, "./labels/city.csv"))
    df_diet = pd.read_csv(os.path.join(base, "./labels/dietary-habits.csv"))
    df_prof = pd.read_csv(os.path.join(base, "./labels/profession.csv"))
    df_degree = pd.read_csv(os.path.join(base, "./labels/degree.csv"))

    return {
        "profession": df_prof["name"].to_list(),
        "degree": df_degree["name"].to_list(),
        "city": df_city["name"].to_list(),
        "diet": df_diet["name"].to_list(),
    }


def get_gpt_response(prompt):
    key = os.getenv("OPENAI_API_KEY")
    print(f"Promot : {prompt}")
    response = client.chat.completions.create(
        model="o1-preview", messages=[{"role": "user", "content": prompt}]
    )
    # data = json.loads(content)
    # print(data['status'])
    return response.choices[0].message.content

def generate_prompt(data):
  
    """
    id,Gender,Age,City,Profession_Status,
    Profession,Academic_Pressure,Work_Pressure,CGPA,Study_Satisfaction,
    Job_Satisfaction,Sleep_Duration,Dietary_Habits,Degree, Suicidal_Thoughts,
    Work_Study_Hours,Financial_Stress,Family_History,Depression

    """
    cols = list(CLEANED_DTYPES.keys())
    cols.remove('Depression')
    

    if(data['professionType'] == 'Student'):
        data['workPressure'] = '-1'
        data['jobSatisfaction'] = '-1'
        data['academicPressure'] = data['pressure']
        data['studySatisfaction'] = data['satisfaction']
        
    else:

        data['academicPressure'] = '-1'
        data['studySatisfaction'] = '-1'

        data['workPressure'] = data['pressure']
        data['jobSatisfaction'] = data['satisfaction']

        data['cgpa'] = -1

        

    df_test = pd.DataFrame(
        [[
            99, 
            data["gender"],
            data["age"],
            data["city"],
            data["professionType"],

            data["profession"],
            float(f"{data["academicPressure"]}"),
            float(f"{data["workPressure"]}"),
            float(f"{data["cgpa"]}"),
            float(f"{data["studySatisfaction"]}"),

            float(f"{data["jobSatisfaction"]}"),
            float(f"{data["sleepDuration"]}"),
            data["dietaryHabits"],
            data["degree"],
            data["suicidalThoughts"],

            float(f"{data["timeSpent"]}"),
            float(f"{data["financialStress"]}"),
            data["familyHistory"],
        ]],columns= cols
    )
    
    # ecoded =  

 
    prediction = model.predict(df_test)
    prompt = []
    print(prediction)
    if data["professionType"] != "Student":

        prompt.append(
            f"{data['name']} is a {data['age']} year old  {data['gender']} a working professional from the city: {data['city']}"
        )
        prompt.append(
            f"the person is working as a {data['profession']} who works on average {data['timeSpent']} hours a day."
        )

        prompt.append(
            f"the person rates themselves on work pressure has a {data['workPressure']}/5.0(5.0 being High pressure)"
        )
        prompt.append(
            f"the person rates themselves on job satisfaction pressure has a {data['jobSatisfaction']}/5.0 (5.0 being High Satisfaction)"
        )
    else:
        prompt.append(
            f"{data['name']} is a {data['age']} year old  {data['gender']} from {data['city']}"
        )
        prompt.append(
            f"who is doing a {data['degree']} course with a CGPA of {data['cgpa']} who studies on average {data['timeSpent']} hours a day."
        )

        prompt.append(
            f"the person rates themselves on academic pressure has a {data['academicPressure']}/5.0 (5.0 being High Pressure)"
        )
        prompt.append(
            f"the person rates themselves on study satisfaction pressure has a {data['studySatisfaction']}/5.0 (5.0 being High Satisfaction)"
        )
        prompt.append(
            f"the person rates themselves on financial pressure on {data['studySatisfaction']}/5.0 (Highest)"
        )

    prompt.append(f"with {data['dietaryHabits']} eating habits  ")
    prompt.append(f"and average {data['sleepDuration']} hours of sleep")

    if data["suicidalThoughts"] == "Yes":
        prompt.append("this person occasionally has suicidal thoughts")

    if data["familyHistory"] == "Yes":
        prompt.append("this person also has a family history of mental illeness")

    prompt.append(
        """this person has high chances of being in depression 
    suggest some activities to reduce chances of depression while suggesting activites consider location, gender , age and demographics. 
    repsond back in a conversational tone as directly talking to the person"""
    )

    return {
        'prediction' : int(prediction[0]),
        'prompt' : "\n".join(prompt)
    }



def cleanup_diet(val) :
    if val == 'Unhealthy':
        return 0
    elif val == 'Healthy':
        return 1
    else:
        return 0.5


def clean_str_data(df):

    df2 = df.copy()
    df2['Gender'] =  df['Gender'].apply(lambda x:  1.0 if x == 'Male' else 0.0)
    df2['Family_History'] = df['Family_History'].apply(lambda x:  1.0 if x == 'Yes' else 0.0)
    df2['Suicidal_Thoughts'] = df['Suicidal_Thoughts'].apply(lambda x:  1.0 if x == 'Yes' else 0.0) 
    df2['Profession_Status'] = df['Profession_Status'].apply(lambda x:  1.0 if x != 'Student' else 0.0) 
    df2['Dietary_Habits'] = df['Dietary_Habits'].apply(cleanup_diet)
    cols = df2.select_dtypes(include= ['object']).columns.to_list()
    cols.append('id')
    return df2.drop(columns=cols) 
