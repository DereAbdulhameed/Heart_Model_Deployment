import streamlit as st
import pandas as pd
import numpy as np
import time 
import pickle

def convert_sex(sex):
    sex_dict = {'Male': 1, 'Female': 0}
    sex_val = sex_dict[sex]
    return sex_val

def convert_chest_pain(chest_pain):
    chest_pain_dict = {'Non-Typical': 2, 'Typical': 3, 'Non-Anginal': 1, 'Asymptomatic': 0}
    chest_pain_val = chest_pain_dict[chest_pain]
    return chest_pain_val

def convert_thal(thal):
    thal_dict = {'Normal': 1, 'Fixed': 0, 'Reversable': 2}
    thal_val = thal_dict[thal]
    return thal_val

def convert_rest_ecg(rest_ecg):
    rest_ecg_dict = {'Normal': 1, 'Left Ventricular Hypertrophy': 0, 'ST-T Wave Abnormality': 2}
    rest_ecg_val = rest_ecg_dict[rest_ecg]
    return rest_ecg_val

def pipelines(age, sex, rest_bp, chol, max_hr, chest_pain, thal, rest_ecg):
    cols = ['age', 'sex', 'rest_bp', 'chol', 'max_hr', 'chest_pain', 'thal', 'rest_ecg']

    feature_list_df = pd.DataFrame(columns=cols)
    feature_list_df['age'] = [age]

    feature_list_df['sex'] = [convert_sex(sex)]
    feature_list_df['rest_bp'] = [rest_bp]
    feature_list_df['chol'] = [chol]
    feature_list_df['max_hr'] = [max_hr]

    feature_list_df['chest_pain'] = [convert_chest_pain(chest_pain)]
    feature_list_df['thal'] = [convert_thal(thal)]
    feature_list_df['rest_ecg'] = [convert_rest_ecg(rest_ecg)]

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

        chest_pain = right_column.selectbox('Chest Pain Type', ('Non-Anginal', 'Typical', 'Non-Typical', 'Asymptomatic'))
        rest_bp = left_column.number_input('Resting Blood Pressure', value= int(80), min_value= 80, max_value=180)
        cholesterol = right_column.number_input('Cholesterol Value in mg/dl', value=int(30))
        rest_ecg = left_column.selectbox('Resting ECG', ('Normal', 'Left Ventricular Hypertrophy', 'ST-T Wave Abnormality'))
        max_hr = right_column.number_input('Maximum Heart Rate', value = int(70), min_value = 70, max_value = 205)
        thal = left_column.selectbox('Thalassemia', ('Normal', 'Fixed', 'Reversable'))
        sex= right_column.selectbox('How would you like to be addressed', ('Male', 'Female'))
        age= left_column.number_input('Age in Figures',value= int(5), min_value = 5, max_value= 130)

        df = pipelines(age, sex, rest_bp, cholesterol, max_hr, chest_pain, thal, rest_ecg)
        st.dataframe(df)

        model_name = 'Heart_Pred_model'
        model = pickle.load(open(model_name, 'rb'))

        diagnosis = st.button('Diagnosis')
        if diagnosis:
            prediction = np.array_str(model.predict(df))
            # prediction_integer = "".join(prediction)
            prediction_proba = model.predict_proba(df)[:,0]
            # prediction_proba_integer = "".join(prediction_proba)

            st.write(f"Your Diagnosis result return {prediction} with a {prediction_proba}")

            if prediction == ['Disease']:
                option = st.selectbox('Who would you like us to contact?',
                ('Your Pastor/Alfa', 'Husband/wife', 'Friend', 'No One'))
                if option == 'No One':
                    st.write('Take Good care of yourself and stay Blessed')
                else:
                    # Add a placeholder
                    calling = st.empty()
                    bar = st.progress(0)

                    for i in range(50):
                      # Update the progress bar with each iteration.
                      calling.text(f'Calling {option}')
                      bar.progress(i + 1)
                      time.sleep(0.2)
                    st.write('Contact Made have a nice day')
    else:
        st.write("## Thank You for Visiting")

if __name__ == "__main__":
    main()