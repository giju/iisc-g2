"""
Following are utility functions created for listing files in a directory and generate a usable key when random values are present in column names

"""
import os
import re
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


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
    'Financial Stress' : float ,
    'Depression':str
}
    



column_encoder = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(),  STR_COLS) 
    ],
)

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

def load_training_data(version) :
    train_list = list_files(f'../data/processed/{version}/')
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




def get_options():

    base = os.path.dirname(__file__)


    print(f"Base {base}")

    df_city = pd.read_csv(os.path.join(base,'./labels/city.csv') )
    df_diet = pd.read_csv(os.path.join(base,'./labels/dietary-habits.csv') ) 
    df_prof = pd.read_csv(os.path.join(base,'./labels/profession.csv') ) 
    df_degree = pd.read_csv(os.path.join(base,'./labels/degree.csv') )  
     
    return {
        'profession': df_prof['name'].to_list(),
        'degree': df_degree['name'].to_list(),
        'city': df_city['name'].to_list(),
        'diet' : df_diet['name'].to_list(),
    }

 
def generate_prompt(data) :
    prompt = []
    if(data['professionType'] == 'working'):

        prompt.append(f"{data['name']} is a {data['age']} year old  {data['gender']} a working professional from the city: {data['city']}")
        prompt.append(f"the person is working as a {data['profession']} who works on average {data['timeSpent']} hours a day.")

        prompt.append(f"the person rates thenselves on work pressure has a {data['workPressure']}/5, 5 beign highest")
        prompt.append(f"the person rates thenselves on job satisfaction pressure has a {data['jobSatisfaction']}/5,  5 beign highest")
    else: 
        prompt.append(f"{data['name']} is a {data['age']} year old  {data['gender']} from {data['city']}")
        prompt.append(f"who is doing a {data['degree']} course with a CGPA of {data['cgpa']} who studies on average {data['timeSpent']} hours a day.")

        prompt.append(f"the person rates thenselves on academic pressure has a {data['academicPressure']}/5, 5 being highest")
        prompt.append(f"the person rates thenselves on study satisfaction pressure has a {data['studySatisfaction']}/5,  5 being highest")
        prompt.append(f"the person rates thenselves on financial pressure on {data['studySatisfaction']}/5,  5 being highest")
 

    prompt.append(f"with {data['dietaryHabits']} eating habits  ")
    prompt.append(f"and average {data['sleepDuration']} hours of sleep")

    if (data['suicidalThoughts'] == 'Yes') :
        prompt.append('this person occasionally has suicidal thoughts')
    
    if (data['familyHistory'] == 'Yes'): 
        prompt.append('this person also has a family history of mental illeness')
    
    prompt.append("""this person has high chances of being in depression 
    suggest some activities to reduce chances of depression while suggesting activites consider location, gender , age and demographics. 
    repsond back in a conversational tone as directly talking to the person""")
 
    return "\n".join(prompt)

def get_gpt_response(prompt): 
    key = os.getenv('OPENAI_API_KEY')
    print(f"Promot : {prompt}")
    response = client.chat.completions.create(
        model="o1-preview",
        messages=[
            {
                "role": "user", 
                "content": prompt
            }
        ]
    )
    # data = json.loads(content)
    # print(data['status'])
    return response.choices[0].message.content