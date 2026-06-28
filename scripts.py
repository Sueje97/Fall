import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ================= Page Basic Configuration =================
st.set_page_config(
    page_title="Fall Risk Prediction System",
    page_icon="👵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 原生标题，消除多余空白框
st.title("👵 Fall Risk Prediction Tool for Community-Dwelling Older Adults")
st.markdown("---")

# Custom CSS 卡片样式
st.markdown("""
<style>
.form-card {
    background-color: #f8fafc;
    padding: 25px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    height: 100%;
}
.result-card {
    padding: 25px;
    border-radius: 12px;
    margin-bottom: 20px;
}
.shap-card {
    padding: 25px;
    border-radius: 12px;
    background-color: #ffffff;
    border:1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# ================= Loading model + SHAP interpreter =================
@st.cache_resource
def load_model_and_explainer():
    try:
        import lightgbm
        model = joblib.load("model.pkl")
        explainer = shap.TreeExplainer(model)
        return model, explainer
    except ModuleNotFoundError:
        st.error("❌ Dependency Missing: lightgbm is not installed. Please run: pip install lightgbm in terminal")
        st.stop()
    except Exception as e:
        st.error(f"❌ Model Loading Failed: {e}")
        st.stop()

model, explainer = load_model_and_explainer()

# 【1、Original feature order of the model (fixed and cannot be changed)】
FEATURE_NAMES_RAW = [
    "Depression", "Fall_history_Yes", "Difficulty_in_bending_Yes",
    "Gender_1", "difficulty_rising_after_sitting_Yes", "Age",
    "Pain_Severe", "Health_Poor"
]

# 【2、mapping: Original variable name】
FEATURE_LABEL_MAPPING = {
    "Depression": "CESD-10 Depression Symptom Score",
    "Fall_history_Yes": "Previous Fall History",
    "Difficulty_in_bending_Yes": "Difficulty Bending Down",
    "Gender_1": "Gender (0=Female,1=Male)",
    "difficulty_rising_after_sitting_Yes": "Difficulty Rising From Sitting Position",
    "Age": "Age",
    "Pain_Severe": "Severe Pain Level (Quite a bit pain?)",
    "Health_Poor": "Poor Self-Rated Health Status (Health=Poor?)"
}
# Generate a list of labels for the waterfall chart display 
FEATURE_NAMES_DISPLAY = [FEATURE_LABEL_MAPPING[name] for name in FEATURE_NAMES_RAW]

# ====================== Left and right column layout ======================
col_left, col_right = st.columns([1, 1.3]) 

# ========== Left side: Input form area==========
with col_left:
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    st.subheader("Input Individual Clinical Indicators")
    with st.form("prediction_form"):
        # 1. Age
        age = st.number_input("Age", min_value=45, max_value=110, value=70)
        # 2. CESD-10 Depression Score
        depression = st.number_input("CESD-10 Depression Symptom Score (0–30)", min_value=0, max_value=30, value=8, step=1)
        # 3. Gender
        gender = st.radio("Gender (0 = Female, 1 = Male)", options=[0, 1], horizontal=True, format_func=lambda x: "Female" if x == 0 else "Male")
        # 4. Severe Pain Level
        pain_severe = st.radio("Severe Pain Level (Quite a bit pain?)", options=[0, 1], horizontal=True, format_func=lambda x: "No" if x == 0 else "Yes")

        fall_history = st.radio("Previous Fall History", options=[0, 1], horizontal=True, format_func=lambda x: "No" if x == 0 else "Yes")
        bending_diff = st.radio("Difficulty Bending Down", options=[0, 1], horizontal=True, format_func=lambda x: "No" if x == 0 else "Yes")
        rising_diff = st.radio("Difficulty Rising From Sitting Position", options=[0, 1], horizontal=True, format_func=lambda x: "No" if x == 0 else "Yes")
        # 5. Poor Self-Rated Health
        health_poor = st.radio("Poor Self-Rated Health Status (Health=Poor?)", options=[0, 1], horizontal=True, format_func=lambda x: "No" if x == 0 else "Yes")

        submitted = st.form_submit_button("🔍 Calculate Fall Risk Probability", use_container_width=True, type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

# ========== On the right: Top = Predicted result; Bottom = SHAP waterfall chart ==========
with col_right:
    if submitted:
        gender_val = gender
        input_data = pd.DataFrame([[
            depression, fall_history, bending_diff,
            gender_val,
            rising_diff, age, pain_severe, health_poor
        ]], columns=FEATURE_NAMES_RAW)

        prob = model.predict_proba(input_data)[0][1]
        cutoff_threshold = 0.220
        if prob >= cutoff_threshold:
            risk_level = "🔴 High Fall Risk"
            bg_color = "#fee2e2"
        else:
            risk_level = "🟢 Low Fall Risk"
            bg_color = "#dcfce7"

        st.markdown(f'<div class="result-card" style="background-color:{bg_color}">', unsafe_allow_html=True)
        st.subheader("Prediction Result")
        st.metric(
            label="Predicted Probability of Fall",
            value=f"{prob:.1%}",
            delta=risk_level
        )
        st.progress(prob)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="shap-card">', unsafe_allow_html=True)
        st.subheader("SHAP Waterfall Plot: Individual Prediction Explanation")
        shap_values = explainer.shap_values(input_data)
        exp = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_data.iloc[0].values,
            feature_names=FEATURE_NAMES_DISPLAY 
        )

        fig, ax = plt.subplots(figsize=(14, 6.8))
        shap.plots.waterfall(exp, show=False, max_display=12)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Fill in the indicators on the left and click the calculation button to view prediction results and SHAP interpretation chart")