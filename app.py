import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# ===== PAGE CONFIG =====
st.set_page_config(page_title="Medical Data Analyst", layout="wide", page_icon="🏥")

# ===== TRAIN MODEL =====
@st.cache_resource
def train_model():
    np.random.seed(42)
    data = pd.DataFrame({
        "Age": np.random.randint(18, 80, 300),
        "BP": np.random.randint(90, 180, 300),
        "Glucose": np.random.randint(70, 200, 300),
        "BMI": np.random.uniform(18, 40, 300),
        "Risk": np.random.choice([0, 1], 300)
    })
    X = data[["Age", "BP", "Glucose", "BMI"]]
    y = data["Risk"]
    model = RandomForestClassifier()
    model.fit(X, y)
    return model, data, X

model, data, X = train_model()

# ===== SESSION STATE =====
if "history" not in st.session_state:
    st.session_state.history = []

if "page" not in st.session_state:
    st.session_state.page = "Home"

# ===== SIDEBAR =====
st.sidebar.title("🏥 Medical AI Dashboard")
pages = [
    "Home", "Dashboard", "Prediction",
    "Upload Patient Data", "History",
    "Graphs", "Explainable AI", "Health Tips"
]

st.session_state.page = st.sidebar.radio("Navigate", pages, index=pages.index(st.session_state.page))
page = st.session_state.page

st.sidebar.info("Advanced Medical Data Analyst System")

# ===== GLOBAL CSS =====
st.markdown("""
<style>
.hero {
    text-align: center;
    padding: 70px 40px;
    background: linear-gradient(135deg, #4e73df, #6f42c1, #00c9a7);
    border-radius: 30px;
    color: white;
    box-shadow: 0 0 35px rgba(0,0,0,0.25);
    animation: fadeIn 1.2s ease;
}

.glass-card {
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(12px);
    padding: 25px;
    border-radius: 20px;
    color: white;
    text-align: center;
    box-shadow: 0 0 15px rgba(255,255,255,0.15);
    transition: 0.3s;
    font-weight: 600;
    font-size: 18px;
}

.glass-card:hover {
    transform: scale(1.05);
    box-shadow: 0 0 25px rgba(255,255,255,0.3);
}

.section-title {
    text-align: center;
    font-size: 30px;
    margin-top: 35px;
    font-weight: bold;
}

.stButton>button {
    background: linear-gradient(135deg,#00c9a7,#4e73df);
    color:white;
    border-radius:14px;
    height:50px;
    width:100%;
    font-size:18px;
}

@keyframes fadeIn {
    from {opacity: 0; transform: translateY(25px);}
    to {opacity: 1; transform: translateY(0);}
}
</style>
""", unsafe_allow_html=True)

# ================= HOME =================
if page == "Home":
    st.markdown("""
    <div class="hero">
        <h1 style="font-size:48px;">🏥 Medical AI Analyst Platform</h1>
        <h3>Next-Gen AI Healthcare Risk Prediction & Analytics</h3>
        <p style="font-size:18px;">Smart • Secure • Intelligent • Doctor-Friendly</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='glass-card'>🧠 AI Disease Risk Prediction</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='glass-card'>📊 Medical Dashboard & Insights</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='glass-card'>🤖 Explainable AI System</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-title">🔬 About This System</div>
    <p style="text-align:center; font-size:18px;">
    Machine Learning based healthcare analytics platform for disease risk prediction and patient insights.
    </p>
    """, unsafe_allow_html=True)

    col4, col5 = st.columns(2)
    with col4:
        st.success("✅ Early Disease Detection")
        st.success("✅ AI Doctor Support")
        st.success("✅ Healthcare Analytics")
    with col5:
        st.success("✅ Student Learning Platform")
        st.success("✅ Hospital Intelligence")
        st.success("✅ Secure Patient Insights")

    if st.button("👉 Start AI Risk Prediction"):
        st.session_state.page = "Prediction"

# ================= DASHBOARD =================
elif page == "Dashboard":
    st.title("📊 Medical Dashboard")
    col1, col2, col3 = st.columns(3)
    col1.metric("👨‍⚕️ Patients", len(data))
    col2.metric("🎂 Avg Age", round(data["Age"].mean(), 1))
    col3.metric("⚠️ High Risk %", f"{round(data['Risk'].mean()*100, 1)}%")
    st.dataframe(data.head(20))

# ================= PREDICTION + CSV DOWNLOAD =================
elif page == "Prediction":
    st.title("🧠 Disease Risk Prediction")

    age = st.slider("Age", 18, 80, 30)
    bp = st.slider("Blood Pressure", 80, 200, 120)
    glucose = st.slider("Glucose", 60, 250, 100)
    bmi = st.slider("BMI", 15.0, 45.0, 22.0)

    if st.button("Predict Risk"):
        input_data = pd.DataFrame([[age, bp, glucose, bmi]], columns=X.columns)
        prob = model.predict_proba(input_data)[0][1]

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob*100,
            title={'text': "Risk Probability %"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "red" if prob>0.7 else "orange" if prob>0.4 else "green"}}
        ))
        st.plotly_chart(fig, use_container_width=True)

        if prob < 0.4:
            risk_label = "LOW RISK ✅"
            st.success(risk_label)
        elif prob < 0.7:
            risk_label = "MEDIUM RISK ⚠️"
            st.warning(risk_label)
        else:
            risk_label = "HIGH RISK ❗"
            st.error(risk_label)

        new_record = {
            "Age": age,
            "BP": bp,
            "Glucose": glucose,
            "BMI": bmi,
            "Risk": risk_label,
            "Probability (%)": round(prob*100, 2)
        }

        st.session_state.history.append(new_record)

        st.subheader("📋 Latest Prediction")
        latest_df = pd.DataFrame([new_record])
        st.dataframe(latest_df)

        csv = latest_df.to_csv(index=False)
        st.download_button("⬇ Download This Prediction (CSV)", csv, "latest_prediction.csv")

    if len(st.session_state.history) > 0:
        st.subheader("📥 Download Full Prediction History")
        history_df = pd.DataFrame(st.session_state.history)
        csv_all = history_df.to_csv(index=False)
        st.download_button("⬇ Download Full History (CSV)", csv_all, "prediction_history.csv")

# ================= UPLOAD PATIENT DATA =================
elif page == "Upload Patient Data":
    st.title("📂 Upload Patient CSV")
    file = st.file_uploader("Upload CSV File", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.success("File Uploaded Successfully!")
        st.dataframe(df.head(50))

# ================= HISTORY =================
elif page == "History":
    st.title("📜 Prediction History")

    if len(st.session_state.history) == 0:
        st.warning("No prediction history yet")
    else:
        st.dataframe(pd.DataFrame(st.session_state.history))

# ================= GRAPHS =================
elif page == "Graphs":
    st.title("📈 Medical Graph Analytics")

    fig = px.histogram(data, x="Age", nbins=15, title="Age Distribution")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.scatter(data, x="BMI", y="Glucose", color="Risk", title="BMI vs Glucose")
    st.plotly_chart(fig2, use_container_width=True)

# ================= EXPLAINABLE AI =================
elif page == "Explainable AI":
    st.title("🤖 Explainable AI")

    importance = model.feature_importances_
    feature_df = pd.DataFrame({"Feature": X.columns, "Importance": importance})

    fig = px.bar(feature_df, x="Importance", y="Feature", orientation="h", title="Feature Importance")
    st.plotly_chart(fig, use_container_width=True)

    st.info("Higher Glucose & BP increase health risk.")

# ================= HEALTH TIPS =================
elif page == "Health Tips":
    st.title("💡 Smart Health Advice")

    tips = [
        "Drink enough water daily",
        "Exercise at least 30 minutes",
        "Reduce sugar intake",
        "Sleep 7-8 hours",
        "Eat fruits and vegetables",
        "Manage stress and relax"
    ]
    st.success(np.random.choice(tips))
