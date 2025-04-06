import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import io

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

# --- Load and preprocess the data
@st.cache_resource
def load_model():
    df = pd.read_csv("student_data.csv")

    # Encode categorical
    df["sex"] = df["sex"].map({"F": 0, "M": 1})
    df["schoolsup"] = df["schoolsup"].map({"yes": 1, "no": 0})
    df.loc[:, "schoolsup"] = df["schoolsup"].fillna(0)

    # Create binary target column: Pass (1) if G3 >= 10 else Fail (0)
    df["G3_result"] = (df["G3"] >= 10).astype(int)

    # Handle class imbalance with upsampling
    df_majority = df[df.G3_result == 1]
    df_minority = df[df.G3_result == 0]

    df_minority_upsampled = resample(df_minority,
                                     replace=True,
                                     n_samples=len(df_majority),
                                     random_state=42)

    df_balanced = pd.concat([df_majority, df_minority_upsampled])

    features = ["age", "studytime", "failures", "absences", "goout", "health", "sex", "schoolsup"]
    X = df_balanced[features]
    y = df_balanced["G3_result"]

    model = RandomForestClassifier(class_weight='balanced', random_state=42)
    model.fit(X, y)
    return model

model = load_model()

# --- Predict
if submitted:
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

    result_text = "PASS ✅" if prediction == 1 else "FAIL ❌"
    confidence_text = f"Confidence: {probability:.4%}"

    if prediction == 1:
        st.success(f"✅ The student is likely to PASS! ({confidence_text})")
    else:
        st.error(f"❌ The student is likely to FAIL. ({confidence_text})")

    # --- Downloadable report
    input_data["Prediction"] = result_text
    input_data["Confidence"] = f"{probability:.4%}"

    csv_buffer = io.StringIO()
    input_data.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📄 Download Prediction Report",
        data=csv_buffer.getvalue(),
        file_name="student_prediction_report.csv",
        mime="text/csv"
    )
