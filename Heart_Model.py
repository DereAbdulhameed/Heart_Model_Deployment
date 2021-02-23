import streamlit as st
import pandas as pd
import numpy as np
import time 
import pickle

def convert_sex(sex_value):
    sex_dict = {'Male': 1, 'Female': 0}
    sex_val = sex_dict[sex_value]
    return sex_val

def pipelines(age, sex, rest_bp, chol, max_hr, chest_pain, thal, rest_ecg):
    cols = ['age', 'sex', 'rest_bp', 'chol', 'max_hr', 'chest_pain' 'thal', 'rest_ecg']

    feature_list_df = pd.DataFrame(columns=cols)
    feature_list_df['age'] = [age]
    feature_list_df['sex'] = [convert_sex(sex)]
    feature_list_df['rest_bp'] = [rest_bp]
    feature_list_df['chol'] = [chol]
    feature_list_df['max_hr'] = [max_hr]

    if chest_pain == 'Typical':
        feature_list_df['chest_pain_typical'] = 1
    elif chest_pain == 'Non-Anginal':
        feature_list_df['chest_pain_nonanginal'] = 1
    elif chest_pain == 'Non-Typical':
        feature_list_df['chest_pain_nontypical'] = 1
    elif chest_pain == 'Atypical':
        feature_list_df['chest_pain_asymptomatic'] = 1
    if thal == 'Normal':
        feature_list_df['thal_normal'] = 1
    elif thal == 'Fixed':
        feature_list_df['thal_fixed'] = 1
    elif thal == 'Reversable':
        feature_list_df['thal_reversable'] = 1
    if rest_ecg == 'Left Ventricular Hypertrophy ':
        feature_list_df['rest_ecg_left ventricular hypertrophy'] = 1
    elif rest_ecg == 'Normal':
        feature_list_df['rest_ecg_normal'] = 1
    elif rest_ecg == 'ST-T Wave Abnormality':
        feature_list_df['rest_ecg_ST-T wave abnormality'] = 1

    feature_list_df = pd.get_dummies(data=feature_list_df, columns=['chest_pain', 'thal', 'rest_ecg'])
    
    return feature_list_df.fillna(0)


def main():
    st.title('A simple Health Disease Susceptibility Check')
    
    st.subheader('Get to know your health status now!!')
    st.write("This App takes in basic clinical features and uses it to predict a patient's susceptibility to a heart disease")    
    choice = st.radio("Do you agree to fill in your personal health data?", ('Agree', 'Disagree'))

    if choice == 'Agree':
        st.write('Great! Welcome')
        st.subheader('Please Fill in appropriately!')
        left_column, right_column = st.beta_columns(2)

        chest_pain = right_column.selectbox('Chest Pain Type', ('Non-Anginal','Typical', 'Non-Typical','Atypical'))
        rest_bp = left_column.number_input('Resting Blood Pressure', value= int(80), min_value= 80)
        cholesterol = right_column.number_input('Cholesterol Value in mg/dl')
        rest_ecg = left_column.selectbox('Resting ECG', ('Normal', 'Left Ventricular Hypertrophy', 'ST-T Wave Abnormality'))
        max_hr = right_column.number_input('Maximum Heart Rate', value = int(70), min_value = 70, max_value = 205)
        thal = left_column.selectbox('Thalassemia', ('Normal', 'Fixed', 'Reversable'))
        sex= right_column.selectbox('How would you like to be addressed', ('Male','Female'))
        age= left_column.number_input('Age in Figures',value= int(5), min_value = 5, max_value= 150)

        df = pipelines(age, sex, rest_bp, cholesterol, max_hr, chest_pain, thal, rest_ecg)

        model_name = 'Heart_Pred_model'
        model = pickle.load(open(model_name, 'rb'))

        diagnosis = st.button('Diagnosis')
        if diagnosis:
            prediction = int(model.predict(df))
            prediction_proba = model.predict_proba(df) [:,0]

            st.write(prediction)
            st.write(prediction_proba)
    else:
        st.write("## Thank You for Visiting")



    option = st.selectbox('Who would you like to be contact?',
    ('Your Doctor', 'Relative', 'Husband/wife', 'Friend'))



    st.progress(0)
    #st.success()
    #st.info()

    st.write('You selected:', option)

if __name__ == "__main__":
    main()