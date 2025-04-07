import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import openai

# Set your OpenAI API key
openai.api_key = "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"

st.set_page_config(page_title="Student Performance Predictor", layout="centered")
st.title("🎓 Student Performance Predictor")

st.markdown("""
Enter student information below to predict whether they will **Pass ✅** or **Fail ❌** and get improvement tips.
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

    use_ai = st.checkbox("💡 Use AI to generate improvement tips", value=True)

    submitted = st.form_submit_button("Predict Performance")

# --- Load the model
@st.cache_resource
def load_model():
    df = pd.read_csv("student_data.csv")

    df["sex"] = df["sex"].map({"F": 0, "M": 1})
    df["schoolsup"] = df["schoolsup"].map({"yes": 1, "no": 0})
    df.loc[:, "schoolsup"] = df["schoolsup"].fillna(0)

    features = ["age", "studytime", "failures", "absences", "goout", "health", "sex", "schoolsup"]
    X = df[features]
    y = np.where(df["G3"] >= 10, 1, 0)

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    return model, df

model, full_data = load_model()

# --- AI Tip Generator
def generate_ai_tips(data):
    try:
        prompt = f"""
        A student has the following profile:
        - Age: {data['age']}
        - Study Time: {data['studytime']}
        - Past Failures: {data['failures']}
        - Absences: {data['absences']}
        - Social Activity: {data['goout']}
        - Health: {data['health']}
        - School Support: {'Yes' if data['schoolsup'] == 1 else 'No'}
        
        Suggest 3 personalized tips to help this student improve their performance and pass.
        """
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return "⚠️ Unable to fetch AI tips at the moment. Please try again later."

# --- Prediction logic
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

    if prediction == 1:
        st.success(f"✅ The student is likely to PASS! (Confidence: {probability:.2%})")
    else:
        st.error(f"❌ The student is likely to FAIL. (Confidence: {probability:.2%})")

        # Show improvement tips
        st.subheader("📘 Improvement Plan")
        if use_ai:
            tips = generate_ai_tips(input_data.iloc[0])
        else:
            tips = "1. Increase study hours to at least 5 hours/week.\n2. Reduce absences and maintain regular attendance.\n3. Seek school support and focus on health."

        st.info(tips)
        
        # Allow downloading tips
        st.download_button("Download Tips as Text File", data=tips, file_name="improvement_tips.txt")

# --- Data visualization
st.divider()
st.subheader("📊 Data Overview")

col1, col2 = st.columns(2)

with col1:
    sex_counts = full_data["sex"].map({0: "Female", 1: "Male"}).value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(sex_counts, labels=sex_counts.index, autopct="%1.1f%%", startangle=90)
    ax1.set_title("Gender Distribution")
    st.pyplot(fig1)

with col2:
    support_counts = full_data["schoolsup"].map({1: "Yes", 0: "No"}).value_counts()
    fig2, ax2 = plt.subplots()
    ax2.pie(support_counts, labels=support_counts.index, autopct="%1.1f%%", startangle=90)
    ax2.set_title("School Support Distribution")
    st.pyplot(fig2)
