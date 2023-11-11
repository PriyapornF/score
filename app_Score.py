import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and encoders
def load_model_and_encoders():
    with open('model_student_data.pkl', 'rb') as file:
        model, *encoders = pickle.load(file)
    return model, encoders

# Categorical Data Encoding
def encode_categorical_data(df, encoders):
    categorical_columns = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
                            'reason', 'guardian']
    for column, encoder in zip(categorical_columns, encoders):
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
    school = st.selectbox('School', ['GP', 'MS'])
    sex = st.selectbox('Sex', ['F', 'M'])
    age = st.slider('Age', 15, 22, 18)
    address = st.selectbox('Address', ['U', 'R'])
    Medu = st.slider('Mother Education (Medu)', 1, 4, 1)
    Fedu = st.slider('Father Education (Fedu)', 1, 4, 1)
    studytime = st.slider('Study Time (hours)', 0, 50, 25)
    G1 = st.slider('คะแนนครั้งที่ 1', 0, 100, 50)
    G2 = st.slider('คะแนนครั้งที่ 2', 0, 100, 50)
    absences = st.slider('Number of Absences', 0, 50, 25)
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
            'absences': [absences]
            # ... (similar entries for other features)
        })

        # Predict scpredict_scoreG3
        prediction = predict_scoreG3(model, user_input, encoders)

        # Display prediction result
        st.subheader('Prediction Result:')
        st.write('G3:', prediction)

if __name__ == '__main__':
    main()
