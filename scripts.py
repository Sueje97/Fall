import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

model = joblib.load("Total_model.pkl")

## Define feature names
feature_names = ["Age", "Sex", "Fall_history", "Difficulty_in_bending", "Difficulty_getting_up_after_prolonged_sitting",
                 "Pain_Severe", "Health_Poor", "Sleep", "Depression"]

# Streamlit user interface
st.title("Falls Risk Prediction Among Community-Dwelling Older Adults")

## Defining variables

# Age: numerical input
Age = st.number_input("Age:", min_value=60, max_value=120, value=60)

# Sex: categorical selection
Sex = st.selectbox("Sex (0=Female, 1=Male):", options=[0, 1], format_func=lambda x: 'Female (0)' if x == 0 else 'Male (1)')

# Fall history: categorical selection
Fall_history = st.selectbox("Fall history:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# Difficulty in bending: categorical selection
Difficulty_in_bending = st.selectbox("Difficulty in bending:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# Difficulty getting up after prolonged sitting: categorical selection
Difficulty_getting_up_after_prolonged_sitting = st.selectbox("Difficulty getting up after prolonged sitting:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# Pain_Severe: categorical selection
Pain_Severe = st.selectbox("Pain(Severe):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# Difficulty in bending: categorical selection
Health_Poor = st.selectbox("Self-reported general health(Pool):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# Sleep duration(h): numerical input
Sleep = st.number_input("Sleep duration(h):", min_value=0, max_value=24, value=6)

# Depression: numerical input
Depression = st.number_input("Depression:", min_value=0, max_value=30, value=0)

# Process inputs and make predictions
feature_values = [Age,Sex, Fall_history, Difficulty_in_bending, Difficulty_getting_up_after_prolonged_sitting,
                 Pain_Severe, Health_Poor, Sleep, Depression]
features = pd.DataFrame([feature_values], columns=feature_names)  

from sklearn.preprocessing import StandardScaler
columns_to_scale = ['Depression', 'Sleep', 'Age']
scaler = StandardScaler()
features[columns_to_scale] = scaler.fit_transform(features[columns_to_scale])
features = features.round(3)


if st.button("Predict"):    
    
    # Predict class and probabilities    
    features_array = features.values

# 预测类别和概率
    predicted_class = model.predict(features_array)[0]
    predicted_proba = model.predict_proba(features_array)[0][1]  # 假设是二分类问题

    # Display prediction results     
    st.write(f"**Predicted Class:** {predicted_class}")    
    st.write(f"**Prediction Probabilities:** {predicted_proba:.1f}")

    # Generate advice based on prediction results   
    probability = predicted_proba[predicted_class] * 100
    
    if predicted_class == 1:        
         advice = (f"According to our model, you have a high risk of falling. "
                   f"The model predicts that your probability of having risk of falling is {probability:.1f}%. "            
                   "While this is just an estimate, it suggests that you may be at significant risk. "    
                 )    
    else:
         advice = ( f"According to our model, you have a low risk of falling. "
                   f"The model predicts that your probability of not having falling is {probability:.1f}%. "       
                  )

    st.write(advice)
    
    # Calculate SHAP values and display force plot 
    ## Load the model
    model = joblib.load("Total_model.pkl")
    def predict_proba(X):
        return model.predict_proba(X)
    
    explainer = shap.LinearExplainer(model, features)
    shap_values = explainer.shap_values(features)
    shap.force_plot(base_value=explainer.expected_value[1],  shap_values=shap_values[1], features=features.iloc[0], matplotlib=True)
    
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")


   