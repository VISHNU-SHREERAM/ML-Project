# import joblib
# # Load the model
# regr = joblib.load('regr.pkl')
# clf = joblib.load('clf.pkl')
# feature_columns = joblib.load('feature_columns.pkl')
# scaler_dict = joblib.load('scaler_dict.pkl')
# catData = joblib.load('catData.pkl')
# label_encoding_dict = joblib.load('label_encoding_dict.pkl')
# numData = joblib.load('numData.pkl')
# unique_values = joblib.load('unique_values.pkl')


# # take the input from the user for each of the features
# input_data = {}
# import streamlit as st
# import pandas as pd

# input_data = {}
# st.title('Stress and Sleep Disorder Predictor')
# # gender
# input_data['Gender']=st.selectbox('Gender', unique_values['Gender'])

# input_data['Age'] = st.number_input('Age', min_value=0, max_value=150, value=None, step=1,placeholder='Enter Age')
# # Occupation
# input_data['Occupation']=st.selectbox('Occupation', unique_values['Occupation'])
# # sleep duration
# input_data['Sleep Duration'] = st.number_input('Sleep Duration', min_value=0, max_value=24, value=None, step=1,placeholder='Enter Sleep Duration')
# #quality of sleep out of 10
# input_data['Quality of Sleep'] = st.number_input('Sleep Quality out of 10', min_value=0, max_value=10, value=None, step=1,placeholder='Enter Sleep Quality')
# #Physical activity Level out of 10
# input_data['Physical Activity Level'] = st.number_input('Physical Activity Level out of 100', min_value=0, max_value=100, value=None, step=1,placeholder='Enter Physical Activity Level')
# # BMI Category
# input_data['BMI Category'] = st.selectbox('BMI Category', unique_values['BMI Category'])
# # Heart Rate
# input_data['Heart Rate'] = st.number_input('Heart Rate', min_value=0, max_value=200, value=None, step=1,placeholder='Enter Heart Rate')
# # Daily Step Count
# input_data['Daily Steps'] = st.number_input('Daily Step Count', min_value=0, max_value=50000, value=None, step=1,placeholder='Enter Daily Step Count')
# # Diasolic Pressure (low blood pressure)
# input_data['Diastolic Pressure'] = st.number_input('Diastolic Pressure (lower limit)', min_value=0, max_value=200, value=None, step=1,placeholder='Enter Diastolic Pressure')
# # Systolic Pressure (high blood pressure)
# input_data['Systolic Pressure'] = st.number_input('Systolic Pressure (upper limit)', min_value=0, max_value=200, value=None, step=1,placeholder='Enter Systolic Pressure')
# # Validation
# valid_inputs = True
# for key, value in input_data.items():
#     if value is None:
#         st.error(f"Please enter a valid value for {key}.")
#         valid_inputs = False

# # try catch for key error, that is when the user does not enter a value for a feature
# if valid_inputs:
#     input_df = pd.DataFrame([input_data])
#     st.write("Input DataFrame:", input_df)

#     if st.checkbox('Predict Stress'):
#         input_df = pd.DataFrame([input_data])
#         for category in catData:
#             if category!='Sleep Disorder':
#                 input_df[category] = label_encoding_dict[category].transform(input_df[category])
#         for f in numData:
#             input_df[f] = scaler_dict[f].transform(input_df[[f]])

#         stress_level = regr.predict(input_df)
#         st.write('Predicted Stress Level:', stress_level[0])

#     if st.checkbox('Predict Sleep Disorder'):
#         input_df = pd.DataFrame([input_data])
#         for category in catData:
#             if category!='Sleep Disorder':
#                 input_df[category] = label_encoding_dict[category].transform(input_df[category])
#         for f in numData:
#             input_df[f] = scaler_dict[f].transform(input_df[[f]])

#         sleep_disorder = clf.predict(input_df)
#         st.write('Predicted Sleep Disorder:', label_encoding_dict['Sleep Disorder'].inverse_transform([sleep_disorder])[0])
# # Run the web app
# # streamlit run final_code.py













import streamlit as st
import pandas as pd
import joblib

# Load models and data
regr = joblib.load('regr.pkl')
clf = joblib.load('clf.pkl')
feature_columns = joblib.load('feature_columns.pkl')
scaler_dict = joblib.load('scaler_dict.pkl')
catData = joblib.load('catData.pkl')
label_encoding_dict = joblib.load('label_encoding_dict.pkl')
numData = joblib.load('numData.pkl')
unique_values = joblib.load('unique_values.pkl')

# App Layout
st.set_page_config(
    page_title="Stress & Sleep Disorder Predictor",
    page_icon="🌜",
    layout="centered"
)

st.title("Stress and Sleep Disorder Predictor 🔮")
st.markdown("""
### Understand your health better
This tool helps predict your stress levels and the likelihood of sleep disorders based on lifestyle and health data.

Please fill in the details on the sidebar, and let's get started!
""")
# use image from directory mindful.png
st.image('mindful.png', use_container_width =True)
# Input Section with user-friendly placeholders and emojis
st.sidebar.header("Input Your Details")

input_data = {}
input_data['Gender'] = st.sidebar.selectbox(
    'Gender 👩‍♂️‍♀️', unique_values['Gender']
)

input_data['Age'] = st.sidebar.number_input(
    'Age 📈', min_value=0, max_value=150, step=1, value=None, placeholder='Enter Age'
)

input_data['Occupation'] = st.sidebar.selectbox(
    'Occupation 💼', unique_values['Occupation']
)

input_data['Sleep Duration'] = st.sidebar.number_input(
    'Sleep Duration (hours) 💤', min_value=0, max_value=24, step=1, value=None, placeholder='Enter Sleep Duration'
)

input_data['Quality of Sleep'] = st.sidebar.number_input(
    'Sleep Quality (out of 10) 💭', min_value=0, max_value=10, step=1, value=None, placeholder='Rate Your Sleep Quality'
)

input_data['Physical Activity Level'] = st.sidebar.number_input(
    'Physical Activity Level (out of 100) 🏃', min_value=0, max_value=100, step=1, value=None, placeholder='Enter Physical Activity Level'
)

input_data['BMI Category'] = st.sidebar.selectbox(
    'BMI Category 🍎', unique_values['BMI Category']
)

input_data['Heart Rate'] = st.sidebar.number_input(
    'Heart Rate (BPM) 💓', min_value=0, max_value=200, step=1, value=None, placeholder='Enter Heart Rate'
)

input_data['Daily Steps'] = st.sidebar.number_input(
    'Daily Step Count 👣', min_value=0, max_value=50000, step=1, value=None, placeholder='Enter Daily Step Count'
)

input_data['Diastolic Pressure'] = st.sidebar.number_input(
    'Diastolic Pressure (mmHg) 🔻', min_value=0, max_value=200, step=1, value=None, placeholder='Enter Diastolic Pressure'
)

input_data['Systolic Pressure'] = st.sidebar.number_input(
    'Systolic Pressure (mmHg) 🔻', min_value=0, max_value=200, step=1, value=None, placeholder='Enter Systolic Pressure'
)

# Validate Input
valid_inputs = all(value is not None for value in input_data.values())

if valid_inputs:
    input_df = pd.DataFrame([input_data])
    st.write("### Your Input Details")
    st.dataframe(input_df)

    # Stress Prediction
    
    if st.button('Predict Stress Level 🔮'):  
        for category in catData:
            if category != 'Sleep Disorder':
                input_df[category] = label_encoding_dict[category].transform(input_df[category])
        for f in numData:
            input_df[f] = scaler_dict[f].transform(input_df[[f]])
        stress_level = regr.predict(input_df)
        if stress_level < 4:
            st.success(f'Predicted Stress Level: {stress_level[0]:.2f} 🌈 Excellent, keep it up!')
        elif 4 <= stress_level < 5:
            st.info(f'Predicted Stress Level: {stress_level[0]:.2f} 🌟 Good, stay balanced!')
        elif 5 <= stress_level < 6:
            st.info(f'Predicted Stress Level: {stress_level[0]:.2f} ⭐ Absolutely Normal for working professionals.')
        elif 6 <= stress_level < 7:
            st.warning(f'Predicted Stress Level: {stress_level[0]:.2f} ⚠ Need improvement in stress management.')
        elif 7 <= stress_level < 8:
            st.error(f'Predicted Stress Level: {stress_level[0]:.2f} ⚠ Take care of mental health.')
        else:
            st.error(f'Predicted Stress Level: {stress_level[0]:.2f} ⚠ Oh, It seems You have high stress, Please Take care of yourself and seek support.')

    # Sleep Disorder Prediction
    if st.button('Predict Sleep Disorder 🔮'):
        for category in catData:
            if category != 'Sleep Disorder':
                input_df[category] = label_encoding_dict[category].transform(input_df[category])
        for f in numData:
            input_df[f] = scaler_dict[f].transform(input_df[[f]])

        sleep_disorder = clf.predict(input_df)
        prediction = label_encoding_dict['Sleep Disorder'].inverse_transform([sleep_disorder])[0]

        if prediction == 'None':
            st.success(f'Predicted Sleep Disorder: None 🌈 Great! You seem to have no issues with sleep.')
        elif prediction == 'Sleep Apnea':
            st.warning(f'Predicted Sleep Disorder: Sleep Apnea ⚠ Please consult a doctor.')
        elif prediction == 'Insomnia':
            st.warning(f'Predicted Sleep Disorder: Insomnia ⚠ Consider seeking professional help.')
else:
    st.error("Please fill in all the input fields to proceed!")

st.markdown("---")
st.markdown("_Disclaimer: This is a predictive tool and not a substitute for professional medical advice. Please consult a healthcare provider for accurate diagnosis and treatment._")
