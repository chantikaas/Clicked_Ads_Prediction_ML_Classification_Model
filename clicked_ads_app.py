import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

# Image
image_url = 'Image1'

# Add custom CSS to set the background color
st.markdown(
    """
    <style>
        body {
            background-color: #ADD8E6;  /* lightblue */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Display image above title
st.image(image_url, use_column_width=True)

# Set Web App Title
st.title("🖱️ Clicked Ad Prediction App 🖱️")
st.write("""
This application **forecasts whether an individual is likely to clicked or ignored an ad**.
""")

# Sidebar header with emojis
st.sidebar.header('🧑🏻 User Input Features 👩🏻')
# Collects user input features into a DataFrame
def user_input_features():
    # Input widgets
    daily_time_spent = st.sidebar.number_input('⏰Average Daily Time Spent on Company Website (Minutes)', value=30)
    daily_internet_usage = st.sidebar.number_input('🌐Average Daily Internet Usage (MB)', value=100)
    age = st.sidebar.number_input('🎂Age', value=40)
    area_income = st.sidebar.number_input('💸Average Area Income (IDR)', value=100000000)
    data = {'Daily Time Spent on Site' : daily_time_spent,
            'Age' : age,
            'Area Income' : area_income,
            'Daily Internet Usage' : daily_internet_usage
            }
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

# Combines user input features with entire clicked ads dataset
# This will be useful for the encoding phase
clicked_raw = pd.read_csv('clicked_ads_cleaned.csv')
clicked_ads = clicked_raw.drop(columns=['Clicked on Ad'])

# Concatenate the cleaned 'clicked_ads' DataFrame with 'input_df'
df = pd.concat([input_df, clicked_ads], axis=0)

# Columns to normalize
columns_to_normalize = ['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']

# Normalize specified columns
scaler = MinMaxScaler()
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

df = df[:1] # Select only the first row of the data

# Reads in saved classification model
load_model = pickle.load(open('clicked_ads_lr_model.pkl', 'rb'))

# Apply model to make predictions
prediction = load_model.predict(df)
prediction_proba = load_model.predict_proba(df)[:, 1]  # Probability of positive class (clicked ad)

# Display predictions and confidence scores
st.subheader('Clicked Prediction and Confidence Score')
st.markdown('**Will the individual clicked on the ad?** 🤔')

for prob in prediction_proba:
    clicked_ads_prediction = 'Clicked Ads' if prob > 0.5 else 'Ignored Ads'
    confidence_ignored = 1 - prob  # Confidence that the customer will ignore the ad
    confidence_score = prob 

    styled_prediction_clicked = f"The model expresses a confidence level of **{confidence_score:.2%}** that the customer will **<span style='color:green;'>click the ad</span>**!😊"
    styled_prediction_ignored = f"The model express a confidence level of **{confidence_ignored:.2%}** that the customer will remain engaged and **<span style='color:red;'>ignore the ad</span>**.😢"

    if clicked_ads_prediction == 'Clicked Ads':
        st.markdown(styled_prediction_clicked, unsafe_allow_html=True)
    else:
        st.markdown(styled_prediction_ignored, unsafe_allow_html=True)
