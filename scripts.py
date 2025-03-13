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
features = np.array([feature_values])

if st.button("Predict"):    
    
    # Predict class and probabilities    
    predicted_class = model.predict(features)[0]    
    predicted_proba = model.predict_proba(features)[0]
    
    # Display prediction results   
    if predicted_class == 1:
        st.write(f"**Predicted Class:** {predicted_class}")
        st.write(f"**Prediction Probabilities:** {predicted_proba}")
    else:
        st.write("**Predicted Class is not 1.**") 

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
    
    explainer = shap.Explainer(predict_proba, masker=shap.maskers.Independent(features))  # Use the prediction function
    shap_values = explainer(pd.DataFrame([feature_values], columns=feature_names))

# Display SHAP force plot
    shap.force_plot(
        explainer.expected_value,shap_values[0],pd.DataFrame([feature_values], columns=feature_names),matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")


