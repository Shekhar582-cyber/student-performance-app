import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Student Performance Predictor", layout="centered")
st.title("🎓 Student Performance Predictor")

st.markdown("""
Enter student information below to predict whether they will **Pass ✅** or **Fail ❌** based on academic and personal factors.
""")

# --- Input form
with st.form("student_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 15, 22, 17)
        studytime = st.selectbox("Study Time (per week)", [1, 2, 3, 4], format_func=lambda x: f"{x} - {'<2h' if x==1 else '2-5h' if x==2 else '5-10h' if x==3 else '>10h'}")
        failures = st.selectbox("Past Class Failures", [0, 1, 2, 3])
        absences = st.slider("Number of Absences", 0, 30, 3)

    with col2:
        sex = st.radio("Sex", ["Male", "Female"])
        goout = st.slider("Going Out Frequency", 1, 5, 3)
        health = st.slider("Health (1=worst, 5=best)", 1, 5, 3)
        schoolsup = st.radio("School Support", ["Yes", "No"])

    submitted = st.form_submit_button("Predict Performance")

# --- Load the model (for demo purposes we'll train it here)
@st.cache_resource
def load_model():
    # Load the original dataset (used for training)
    df = pd.read_csv("student_data.csv")

    # Encode categorical
    df["sex"] = df["sex"].map({"F": 0, "M": 1})
    df["schoolsup"] = df["schoolsup"].map({"yes": 1, "no": 0})

    # Fill missing categorical values with mode using a safe approach
    df.loc[:, "schoolsup"] = df["schoolsup"].fillna(0)

    # Select features and target
    features = ["age", "studytime", "failures", "absences", "goout", "health", "sex", "schoolsup"]
    X = df[features]
    y = np.where(df["G3"] >= 10, 1, 0)  # 1 = Pass, 0 = Fail

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    return model

model = load_model()

# --- Predict
if submitted:
    # Map inputs
    input_data = pd.DataFrame([{
        "age": age,
        "studytime": studytime,
        "failures": failures,
        "absences": absences,
        "goout": goout,
        "health": health,
        "sex": 1 if sex == "Male" else 0,
        "schoolsup": 1 if schoolsup == "Yes" else 0
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][prediction]

    if prediction == 1:
        st.success(f"✅ The student is likely to PASS! (Confidence: {probability:.2%})")
    else:
        st.error(f"❌ The student is likely to FAIL. (Confidence: {probability:.2%})")
