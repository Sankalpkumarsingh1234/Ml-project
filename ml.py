
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import joblib
import os

df = pd.read_csv('Training.csv')
X = df.drop('prognosis', axis=1)
y = df['prognosis']

le = LabelEncoder()
y_encoded = le.fit_transform(y)
diseases_list = le.classes_

x_train, x_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


model_file = "svc_model.pkl"
if os.path.exists(model_file):
    svc = joblib.load(model_file)
else:
    svc = SVC(kernel='linear', probability=True)
    svc.fit(x_train, y_train)
    joblib.dump(svc, model_file)


def load_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.title()
    return df

description = load_csv('description.csv')
diets = load_csv('diets.csv')
precautions = load_csv('precautions_df.csv')
medications = load_csv('medications.csv')
workout = load_csv('workout_df.csv')

symptoms_dict = {col: idx for idx, col in enumerate(X.columns)}


def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join(desc.values) if not desc.empty else "No description available"

    pre = precautions[precautions['Disease'] == dis][['Precaution_1','Precaution_2','Precaution_3','Precaution_4']]
    pre = pre.values.flatten().tolist() if not pre.empty else ["No precautions available"]

    med = medications[medications['Disease'] == dis]['Medication']
    med = med.values.tolist() if not med.empty else ["No medication available"]

    die = diets[diets['Disease'] == dis]['Diet']
    die = die.values.tolist() if not die.empty else ["No diet available"]

    wrkout = workout[workout['Disease'] == dis]['Workout']
    wrkout = wrkout.values.tolist() if not wrkout.empty else ["No workout available"]

    return desc, pre, med, die, wrkout
