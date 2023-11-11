import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and encoders
def load_model_and_encoders():
    with open('model_student_data.pkl', 'rb') as file:
        model, *encoders = pickle.load(file)
    return model, *encoders

# Categorical Data Encoding
def encode_categorical_data(df, encoders):
    categorical_columns = [ 'school', 'sex', 'age', 'address', 'Medu', 'Fedu', 'studytime', 'G1', 'G2', 'absences', 
                           'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'Dalc', 'Walc', 
                           'activities_yes', 'failures', 'famrel', 'famsup_yes', 'freetime', 'goout', 'health', 'higher_yes']
    for column, encoder in zip(categorical_columns, encoders):
        # Ensure that the column exists in the DataFrame
        if column in df.columns:
            df[column] = encoder.transform(df[column])
    return df


# Predict ScoreG3
def predict_scoreG3(model, user_input, encoders):
    user_input = encode_categorical_data(user_input, encoders)
    prediction = model.predict(user_input)
    return prediction[0]

# Main Streamlit App
def main():
    st.title('Student Performance Form')

    # Load model and encoders
    model, *encoders = load_model_and_encoders()

    # Create a form for students to fill out
    st.subheader('Student Information')
    school = st.selectbox('School', [0, 1])
    sex = st.selectbox('Sex', [0, 1])
    age = st.slider('Age', 15, 22, 18)
    address = st.selectbox('Address', [0, 1])
    Medu = st.slider('Mother Education (Medu)', 1, 4, 1)
    Fedu = st.slider('Father Education (Fedu)', 1, 4, 1)
    studytime = st.slider('Study Time (hours)', 0, 50, 25)
    G1 = st.slider('คะแนนครั้งที่ 1', 0, 100, 50)
    G2 = st.slider('คะแนนครั้งที่ 2', 0, 100, 50)
    absences = st.slider('Number of Absences', 0, 50, 25)
    famsize = st.selectbox('Family size', [0, 1])
    Pstatus = st.selectbox('Pstatus', [0, 1])
    Mjob = st.selectbox('Mother job', [0, 1, 2, 3, 4])
    Fjob =  st.selectbox('Father job',[0, 1, 2, 3, 4])
    reason = st.selectbox('Reason', [0, 1, 2, 3])
    guardian = st.selectbox('Guardian', [0, 1, 2])
    Dalc = st.selectbox('Dalc', [0, 1])
    Walc = st.selectbox('Walc', [0, 1])
    activities_yes = st.selectbox('activities_yes', [0, 1])
    failures = st.selectbox('failures', [0, 1])
    famrel = st.selectbox('famrel', [0, 1])
    famsup_yes = st.selectbox('famsup_yes', [0, 1])
    freetime = st.selectbox('freetime', [0, 1])
    goout = st.selectbox('goout', [0, 1])
    health = st.selectbox('health', [0, 1])
    higher_yes = st.selectbox('higher_yes', [0, 1])
    # ... (similar input fields for other features)
    # ... (similar input fields for other features)

    # Form submission
    if st.button('Submit'):
        # Create a DataFrame for the user input
        user_input = pd.DataFrame({
            'school': [school],
            'sex': [sex],
            'age': [age],
            'address': [address],
            'Medu': [Medu],
            'Fedu': [Fedu],
            'studytime': [studytime],
            'G1': [G1],
            'G2': [G2],
            'absences': [absences],
            'famsize': [famsize], 
            'Pstatus': [Pstatus], 
            'Mjob': [Mjob], 
            'Fjob': [Fjob],
            'reason': [reason], 
            'guardian': [guardian],
            'Dalc': [Dalc],
            'Walc': [Walc],
            'activities_yes': [activities_yes],
            'failures': [failures],
            'famrel': [famrel],
            'famsup_yes': [famsup_yes], 
            'freetime': [freetime], 
            'goout': [goout], 
            'health': [health], 
            'higher_yes': [higher_yes]
            # ... (similar entries for other features)
        })

        # Predict scpredict_scoreG3
        prediction = predict_scoreG3(model, user_input, encoders)

        # Display prediction result
        st.subheader('Prediction Result:')
        st.write('G3:', prediction)

if __name__ == '__main__':
    main()
