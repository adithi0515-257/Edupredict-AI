import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="EduPredict AI", page_icon="🎓", layout="wide")

# -----------------------------
# Data and model
# -----------------------------
@st.cache_data
def make_demo_dataset(n=1200, seed=42):
    """Create a reproducible DEMONSTRATION dataset. Replace with real, consented data for research."""
    rng = np.random.default_rng(seed)
    study_hours = rng.uniform(0, 8, n)
    attendance = rng.uniform(40, 100, n)
    previous_marks = rng.uniform(25, 100, n)
    assignment_completion = rng.uniform(0, 100, n)
    sleep_hours = rng.uniform(3.5, 10, n)
    # Illustrative relationship, not a validated educational model.
    noise = rng.normal(0, 7, n)
    final_score = (
        0.30 * previous_marks
        + 2.6 * study_hours
        + 0.16 * attendance
        + 0.12 * assignment_completion
        + 1.0 * np.clip(sleep_hours - 5, -2, 3)
        + noise
    )
    final_score = np.clip(final_score, 0, 100)
    return pd.DataFrame({
        "Study hours/day": study_hours,
        "Attendance (%)": attendance,
        "Previous marks (%)": previous_marks,
        "Assignments completed (%)": assignment_completion,
        "Sleep hours/night": sleep_hours,
        "Final score (%)": final_score
    })

FEATURES = [
    "Study hours/day", "Attendance (%)", "Previous marks (%)",
    "Assignments completed (%)", "Sleep hours/night"
]
TARGET = "Final score (%)"

@st.cache_resource
def train_model():
    data = make_demo_dataset()
    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=180, max_depth=10, min_samples_leaf=3, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2": r2_score(y_test, predictions),
        "test_size": len(y_test)
    }
    return model, data, metrics

model, data, metrics = train_model()

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0b1220 0%, #111b31 55%, #18223c 100%); }
    [data-testid="stSidebar"] { background: #0b1220; }
    h1, h2, h3, p, label, span { color: #eef4ff; }
    .hero { padding: 1.4rem 1.6rem; border: 1px solid #334766; border-radius: 20px;
            background: linear-gradient(120deg, rgba(59,130,246,.20), rgba(139,92,246,.15)); }
    .muted { color: #a8b8d5; }
    .stMetric { background: rgba(255,255,255,.045); padding: 14px; border-radius: 14px; border: 1px solid #263752; }
    div[data-testid="stForm"] { border: 1px solid #334766; border-radius: 16px; padding: 18px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🎓 EduPredict AI")
    st.caption("Student performance insights")
    page = st.radio("Navigate", ["Predictor", "Model Insights", "About the Project"])
    st.divider()
    st.caption("GDG on Campus • ML Project")
    st.caption("Demo data only — not for high-stakes decisions.")

if page == "Predictor":
    st.markdown("""
    <div class="hero">
      <h1>Learn smarter. Grow stronger.</h1>
      <p class="muted">Estimate a possible academic score and receive practical, personalized study suggestions.</p>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1.0, 1.1], gap="large")

    with left:
        st.subheader("📋 Student profile")
        with st.form("prediction_form"):
            study = st.slider("Study hours per day", 0.0, 12.0, 3.0, 0.5)
            attendance = st.slider("Attendance", 0, 100, 80, format="%d%%")
            previous = st.slider("Previous exam marks", 0, 100, 70, format="%d%%")
            assignments = st.slider("Assignments completed", 0, 100, 75, format="%d%%")
            sleep = st.slider("Sleep per night", 2.0, 12.0, 7.0, 0.5)
            submitted = st.form_submit_button("✨ Predict performance", use_container_width=True)

    with right:
        st.subheader("📈 Your prediction")
        if submitted:
            row = pd.DataFrame([{
                "Study hours/day": study,
                "Attendance (%)": attendance,
                "Previous marks (%)": previous,
                "Assignments completed (%)": assignments,
                "Sleep hours/night": sleep
            }])
            score = float(np.clip(model.predict(row)[0], 0, 100))
            category = "Excellent range" if score >= 85 else "Strong range" if score >= 70 else "Developing range" if score >= 50 else "Needs support"
            c1, c2 = st.columns(2)
            c1.metric("Estimated final score", f"{score:.1f}/100")
            c2.metric("Performance band", category)
            st.progress(int(round(score)))
            st.caption("This is an estimate from a demonstration model, not a guarantee or a formal academic assessment.")

            st.markdown("#### 💡 Your study action plan")
            tips = []
            if study < 2:
                tips.append("Build a consistent routine: try two focused 25-minute sessions, then increase gradually.")
            elif study < 4:
                tips.append("Try one extra focused study block on difficult topics, with short breaks.")
            else:
                tips.append("Protect the quality of your study time with practice questions and spaced revision.")
            if attendance < 75:
                tips.append("Aim to attend more classes and review missed notes with a classmate or instructor.")
            if previous < 60:
                tips.append("Revisit foundational topics and solve past questions to identify recurring gaps.")
            if assignments < 70:
                tips.append("Break assignments into small deadlines and submit each part before the due date.")
            if sleep < 7:
                tips.append("If possible, maintain a consistent sleep schedule; rest can support attention and learning.")
            if not tips:
                tips.append("Keep your routine consistent, review mistakes weekly, and use practice tests to track progress.")
            for tip in tips:
                st.markdown(f"- {tip}")

            with st.expander("Why did the model predict this?"):
                st.write("The model learned patterns from the demo training data. Feature importance below describes the model's overall use of features; it does not prove cause and effect.")
                importance = pd.DataFrame({"Feature": FEATURES, "Importance": model.feature_importances_}).sort_values("Importance", ascending=True)
                fig = px.bar(importance, x="Importance", y="Feature", orientation="h", title="Model feature importance")
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Adjust the student profile and select **Predict performance** to see an estimate and study suggestions.")
            st.markdown("##### What you'll get")
            st.markdown("- Predicted score estimate\n- Performance band\n- Actionable study suggestions\n- Explanation of model feature importance")

elif page == "Model Insights":
    st.markdown("<div class='hero'><h1>Model Insights</h1><p class='muted'>A transparent look at the model and its demonstration evaluation.</p></div>", unsafe_allow_html=True)
    st.write("")
    a, b, c = st.columns(3)
    a.metric("Mean Absolute Error", f"{metrics['MAE']:.2f} points")
    b.metric("RMSE", f"{metrics['RMSE']:.2f} points")
    c.metric("R² score", f"{metrics['R2']:.3f}")
    st.caption(f"Evaluation used a held-out test split of {metrics['test_size']} synthetic examples. Metrics describe this generated demo data only; they are not evidence of real-world accuracy.")
    st.subheader("Feature importance")
    importance = pd.DataFrame({"Feature": FEATURES, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
    fig = px.bar(importance, x="Feature", y="Importance", title="Which features does the model rely on?")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_title="", yaxis_title="Relative importance")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Demo dataset preview")
    st.dataframe(data.head(10), use_container_width=True)
    fig2 = px.scatter(data.sample(250, random_state=42), x="Study hours/day", y=TARGET, color="Attendance (%)", title="Illustrative relationship between study time and score")
    fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)

else:
    st.markdown("<div class='hero'><h1>About EduPredict AI</h1><p class='muted'>An educational machine-learning prototype for learning and discussion.</p></div>", unsafe_allow_html=True)
    st.subheader("Problem statement")
    st.write("Students often receive marks after an assessment, but may not have an accessible way to explore how study habits and prior performance relate to outcomes.")
    st.subheader("Proposed solution")
    st.write("EduPredict AI demonstrates a regression model, a simple prediction interface, model evaluation, feature importance, and rule-based suggestions for study habits.")
    st.subheader("ML workflow")
    st.markdown("""
    1. Generate a reproducible demonstration dataset.
    2. Separate input features and the target score.
    3. Split data into training and test sets.
    4. Train a Random Forest Regressor.
    5. Evaluate using MAE, RMSE, and R².
    6. Make predictions for a new student profile.
    7. Display suggestions and explain model limitations.
    """)
    st.subheader("Limitations and responsible use")
    st.warning("This version uses synthetic data generated from an illustrative formula. It is a functional demo of an ML pipeline, not a validated student-scoring tool. Do not use it to grade, rank, admit, discipline, or make other high-stakes decisions about real students. For a research version, use an appropriate real dataset with permission, document its source, test for bias and data leakage, and validate on unseen data.")
    st.markdown("**Possible future work:** use a public educational dataset, compare Linear Regression with Random Forest, add cross-validation, improve explainability with SHAP, and deploy the app.")
