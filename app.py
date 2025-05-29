import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer

# Set page config as the first Streamlit command
st.set_page_config(page_title="Personality Predictor", page_icon="😊")

# Load dataset for fitting preprocessors
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'personality_dataset.csv'))
        return df
    except FileNotFoundError:
        return None

# Load pretrained model
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open(os.path.join(os.path.dirname(__file__), 'model.pickle'), 'rb'))
        return model
    except FileNotFoundError:
        return None

# Load data and model
df = load_data()
model = load_model()

# Define preprocessing objects
numeric_columns = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                   'Friends_circle_size', 'Post_frequency']
categorical_columns = ['Stage_fear', 'Drained_after_socializing']

# Initialize preprocessors
numeric_imputer = SimpleImputer(strategy='median')
categorical_imputer = SimpleImputer(strategy='most_frequent')
scaler = StandardScaler()
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
le = LabelEncoder()

# Fit preprocessors on the dataset if available
if df is not None:
    numeric_imputer.fit(df[numeric_columns])
    categorical_imputer.fit(df[categorical_columns])
    # Compute bin edges for Time_spent_Alone
    time_spent_alone_bin_edges = pd.qcut(df['Time_spent_Alone'].dropna(), q=3, retbins=True)[1]
    # Prepare features for scaling and polynomial transformation
    df_temp = df.copy()
    df_temp[numeric_columns] = numeric_imputer.transform(df_temp[numeric_columns])
    df_temp[categorical_columns] = categorical_imputer.transform(df_temp[categorical_columns])
    df_temp = pd.get_dummies(df_temp, columns=categorical_columns, drop_first=True)
    df_temp['Alone_to_Social_Ratio'] = df_temp['Time_spent_Alone'] / (df_temp['Social_event_attendance'] + 1)
    df_temp['Social_Comfort_Index'] = (
        df_temp['Friends_circle_size'] + df_temp['Post_frequency'] - df_temp['Stage_fear_Yes']
    ) / 3
    df_temp['Social_Overload'] = df_temp['Drained_after_socializing_Yes'] * df_temp['Social_event_attendance']
    df_temp['Time_spent_Alone_Binned'] = pd.cut(
        df_temp['Time_spent_Alone'], 
        bins=time_spent_alone_bin_edges, 
        labels=['Low', 'Medium', 'High'], 
        include_lowest=True
    )
    df_temp = pd.get_dummies(df_temp, columns=['Time_spent_Alone_Binned'], drop_first=True)
    poly_features = poly.fit_transform(df_temp[['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size']])
    poly_feature_names = poly.get_feature_names_out(['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size'])
    df_temp[poly_feature_names] = poly_features
    feature_columns = numeric_columns + ['Stage_fear_Yes', 'Drained_after_socializing_Yes', 
                                        'Alone_to_Social_Ratio', 'Social_Comfort_Index', 
                                        'Social_Overload', 'Time_spent_Alone_Binned_Medium', 
                                        'Time_spent_Alone_Binned_High'] + list(poly_feature_names)
    scaler.fit(df_temp[feature_columns])
else:
    # Fallback if dataset is unavailable
    time_spent_alone_bin_edges = [0, 3, 7, 11]  # Approximated from dataset quantiles
    feature_columns = numeric_columns + ['Stage_fear_Yes', 'Drained_after_socializing_Yes', 
                                        'Alone_to_Social_Ratio', 'Social_Comfort_Index', 
                                        'Social_Overload', 'Time_spent_Alone_Binned_Medium', 
                                        'Time_spent_Alone_Binned_High', 
                                        'Time_spent_Alone Social_event_attendance',
                                        'Time_spent_Alone Friends_circle_size',
                                        'Social_event_attendance Friends_circle_size']
    # Adjust feature_columns to match model's expected feature count
    if model is not None and hasattr(model, 'coef_'):
        expected_features = model.coef_[0].shape[0]
        if len(feature_columns) > expected_features:
            feature_columns = feature_columns[:expected_features]
            st.warning(f"Adjusted feature_columns to {expected_features} features to match model expectations.")
        elif len(feature_columns) < expected_features:
            feature_columns += [f'Placeholder_{i}' for i in range(expected_features - len(feature_columns))]
            st.warning(f"Padded feature_columns to {expected_features} features with placeholders to match model expectations.")

# Prediction function
def predict_extrovert_introvert(time_spent_alone, social_event_attendance, going_outside, 
                               friends_circle_size, post_frequency, stage_fear, drained_after_socializing):
    if model is None:
        return "Error: Model not loaded", None
    
    # Input validation
    if any(x < 0 for x in [time_spent_alone, social_event_attendance, going_outside, 
                           friends_circle_size, post_frequency]):
        return "Error: Input values cannot be negative", None
    
    # Create input dictionary
    input_dict = {
        'Time_spent_Alone': time_spent_alone,
        'Social_event_attendance': social_event_attendance,
        'Going_outside': going_outside,
        'Friends_circle_size': friends_circle_size,
        'Post_frequency': post_frequency,
        'Stage_fear': stage_fear,
        'Drained_after_socializing': drained_after_socializing
    }
    
    # Convert dict to DataFrame
    df_input = pd.DataFrame([input_dict])
    
    # Impute missing values
    df_input[numeric_columns] = numeric_imputer.transform(df_input[numeric_columns])
    df_input[categorical_columns] = categorical_imputer.transform(df_input[categorical_columns])
    
    # Encode categorical variables
    df_input = pd.get_dummies(df_input, columns=categorical_columns, drop_first=True)
    
    # Add missing dummy columns
    for col in ['Stage_fear_Yes', 'Drained_after_socializing_Yes']:
        if col not in df_input.columns:
            df_input[col] = 0
    
    # Cap outliers (based on dataset IQR)
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25) if df is not None else df_input[col].quantile(0.25)
        Q3 = df[col].quantile(0.75) if df is not None else df_input[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + (2.5 * IQR if col in ['Social_event_attendance', 'Friends_circle_size', 'Post_frequency'] else 1.5 * IQR)
        df_input[col] = df_input[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Feature engineering
    df_input['Alone_to_Social_Ratio'] = df_input['Time_spent_Alone'] / (df_input['Social_event_attendance'] + 1)
    df_input['Social_Comfort_Index'] = (
        df_input['Friends_circle_size'] + df_input['Post_frequency'] - df_input['Stage_fear_Yes']
    ) / 3
    df_input['Social_Overload'] = df_input['Drained_after_socializing_Yes'] * df_input['Social_event_attendance']
    
    # Binned feature
    df_input['Time_spent_Alone_Binned'] = pd.cut(
        df_input['Time_spent_Alone'], 
        bins=time_spent_alone_bin_edges, 
        labels=['Low', 'Medium', 'High'], 
        include_lowest=True
    )
    df_input = pd.get_dummies(df_input, columns=['Time_spent_Alone_Binned'], drop_first=True)
    
    # Add missing bin columns
    for col in ['Time_spent_Alone_Binned_Medium', 'Time_spent_Alone_Binned_High']:
        if col not in df_input.columns:
            df_input[col] = 0
    
    # Polynomial features
    poly_features = poly.transform(df_input[['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size']])
    poly_feature_names = poly.get_feature_names_out(['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size'])
    df_input[poly_feature_names] = poly_features
    
    # Ensure all training columns are present
    for col in feature_columns:
        if col not in df_input.columns:
            df_input[col] = 0
    
    # Arrange columns in training order
    df_input = df_input[feature_columns]
    
    # Scale features
    df_input_scaled = scaler.transform(df_input)
    
    # Adjust input to match model's expected feature count
    expected_features = model.coef_[0].shape[0] if hasattr(model, 'coef_') else len(feature_columns)
    if df_input_scaled.shape[1] != expected_features:
        if df_input_scaled.shape[1] > expected_features:
            df_input_scaled = df_input_scaled[:, :expected_features]
        else:
            padding = np.zeros((df_input_scaled.shape[0], expected_features - df_input_scaled.shape[1]))
            df_input_scaled = np.hstack([df_input_scaled, padding])
    
    # Predict
    try:
        prob = model.predict_proba(df_input_scaled)[0]
        prediction = model.predict(df_input_scaled)[0]
        confidence = max(prob) * 100
        return 'Extrovert' if prediction == 0 else 'Introvert', confidence
    except Exception as e:
        return f"Error during prediction: {str(e)}", None

# Check if model loaded successfully
if model is None:
    st.error("Model file not found. Please ensure 'model.pickle' is in the correct directory.")
elif df is None:
    st.error("Dataset file not found. Please ensure 'personality_dataset.csv' is in the correct directory.")
else:
    st.title("Personality Predictor: Are You an Introvert or Extrovert?")
    st.markdown("Enter your social behavior details below to find out your personality type!")
    
    # Input explanations
    with st.expander("What do these inputs mean?"):
        st.markdown("""
        - **Hours spent alone per day**: Time spent alone (0-24 hours).
        - **Social events per month**: Number of social gatherings you attend monthly.
        - **Times going outside per week**: How often you leave your home weekly.
        - **Number of close friends**: Count of close friends you interact with regularly.
        - **Social media posts per week**: How often you post on social media weekly.
        - **Stage fear**: Whether you feel nervous speaking in public (Yes/No).
        - **Feel drained after socializing**: Whether you feel tired after social interactions (Yes/No).
        """)
    
    # Create two columns for input fields
    col1, col2 = st.columns(2)
    
    with col1:
        time_spent_alone = st.slider("Hours spent alone per day", 0,12, 1,
                                    help="How many hours do you spend alone daily?")
        social_event_attendance = st.slider("Social events per month", 0,10,1,
                                          help="How many social events do you attend monthly?")
        going_outside = st.slider("Times going outside per week", 0,7,1,
                                 help="How many times do you go outside weekly?")
        stage_fear = st.selectbox("Do you have stage fear?", ["No", "Yes"], 
                                 help="Do you feel nervous speaking in public?")
    
    with col2:
        friends_circle_size = st.slider("Number of close friends", 0,15,1,
                                      help="How many close friends do you have?")
        post_frequency = st.slider("Social media posts per week", 0,10,1,
                                  help="How many times do you post on social media weekly?")
        drained_after_socializing = st.selectbox("Feel drained after socializing?", ["No", "Yes"], 
                                               help="Do you feel tired after social interactions?")
    
    # Create columns for buttons
    col_btn1, col_btn2 = st.columns([1, 1])
    
    with col_btn1:
        predict_clicked = st.button("Predict My Personality", type="primary")
    
    with col_btn2:
        if st.button("Reset Inputs"):
            st.session_state.clear()
            st.rerun()
    
    # Predict button action
    if predict_clicked:
        with st.spinner("Predicting..."):
            result, confidence = predict_extrovert_introvert(
                time_spent_alone, social_event_attendance, going_outside, 
                friends_circle_size, post_frequency, stage_fear, drained_after_socializing
            )
            if "Error" in result:
                st.error(result)
            else:
                emoji = "🤗" if result == "Extrovert" else "🤫"
                st.markdown(f"### You are an **{result}**! {emoji}")
                st.write(f"Prediction Confidence: {confidence:.2f}%")
    
    # Example prediction
    st.subheader("Example Prediction")
    example_result, example_confidence = predict_extrovert_introvert(
        time_spent_alone=11.0,
        social_event_attendance=2,
        going_outside=3,
        friends_circle_size=4,
        post_frequency=2,
        stage_fear="Yes",
        drained_after_socializing="Yes"
    )
    st.write(f"Example prediction for typical inputs: **{example_result}** (Confidence: {example_confidence:.2f}%)")
