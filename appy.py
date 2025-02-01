import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor

# Load trained CatBoost Model
@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model("catboost_neet_model.cbm")  # Load saved model
    return model

model = load_model()

# Web App Title
st.title("🎯 NEET Rank Predictor & Student Performance Analyzer")

# Upload CSV File
uploaded_file = st.file_uploader("📂 Upload your quiz performance data (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("🛠 Available Columns in Uploaded Data:")
    st.write(df.columns.tolist())  # Prints all column names
    st.write("📊 Sample Data Preview:")
    st.dataframe(df.head())

    # 🎯 Generate Insights: Identify Weak Areas & Performance Gaps
    st.subheader("📉 Insights: Weak Areas & Performance Gaps")

    # Identify weak subjects
   # Check if "subject" exists, otherwise use an alternative column
    if "subject" in df.columns:
      weak_subjects = df[df["accuracy"] < 50]["subject"].unique()
    elif "topic" in df.columns:  # If "subject" is missing, try "topic"
      weak_subjects = df[df["accuracy"] < 50]["topic"].unique()
    else:
      weak_subjects = []  # Empty if neither column exists

    if len(weak_subjects) > 0:
        st.warning(f"❌ Weak Subjects: {', '.join(weak_subjects)} (Accuracy < 50%)")
    else:
        st.success("✅ No major weak subjects detected!")

    # Identify most common mistake topics
    mistake_counts = df.groupby("topic")["mistakes"].sum().sort_values(ascending=False)
    st.write("📌 Topics where most mistakes happen:")
    st.dataframe(mistake_counts.head(5))

    # Plot accuracy trends over time
    st.subheader("📈 Accuracy Trends Over Time")
    plt.figure(figsize=(8, 5))
    sns.lineplot(x=df.index, y=df["accuracy"], marker="o")
    plt.xlabel("Quiz Number")
    plt.ylabel("Accuracy (%)")
    plt.title("Student Accuracy Improvement Over Time")
    st.pyplot(plt)

    # 🎯 Personalized Study Recommendations
    st.subheader("📌 Personalized Study Recommendations")
    recommendations = []

    # Check for common weak topics
    if len(weak_subjects) > 0:
        recommendations.append(f"⚠ Focus more on weak subjects: {', '.join(weak_subjects)}.")
    
    # Check accuracy vs difficulty level
    if "difficulty" in df.columns:
        hard_accuracy = df[df["difficulty"] == "Hard"]["accuracy"].mean()
        if hard_accuracy < 50:
            recommendations.append("🔥 Practice more difficult-level questions to improve confidence.")

    # General improvement recommendations
    if df["accuracy"].mean() < 70:
        recommendations.append("📚 Revise core concepts and practice more mock tests.")
    
    if len(recommendations) > 0:
        for rec in recommendations:
            st.warning(rec)
    else:
        st.success("✅ You're on track! Keep practicing.")

    # 🧑‍🎓 Student Persona Analysis
    st.subheader("🧑‍🎓 Student Persona Analysis")

    avg_accuracy = df["accuracy"].mean()
    avg_speed = df["speed"].mean() if "speed" in df.columns else None
    avg_mistakes = df["mistakes"].mean()

    # Define persona based on accuracy & speed
    if avg_accuracy >= 85:
        persona = "🔥 High Achiever - Excels in Accuracy"
    elif avg_accuracy >= 70:
        persona = "📚 Concept Builder - Strong, but Needs More Practice"
    else:
        persona = "⚡ Risk Taker - Needs to Improve Accuracy"

    if avg_speed and avg_speed >= 90:
        persona += " 🏃 (Fast Learner)"
    elif avg_speed and avg_speed < 60:
        persona += " 🐢 (Needs to Improve Speed)"
    
    if avg_mistakes > 10:
        persona += " ❌ (Tends to Make Many Mistakes)"

    st.success(f"**Student Persona: {persona}**")

    # 🎯 Probabilistic Model for NEET Rank Prediction
    st.subheader("🎯 Predicting NEET Rank with Probabilistic Model")

    def predict_neet_rank(score, accuracy):
        base_rank = 50000  # Approximate base rank
        score_factor = (1 - (score / 200)) * 20000  # Score impact
        accuracy_factor = (1 - accuracy) * 20000  # Accuracy impact
        predicted_rank = base_rank - score_factor - accuracy_factor
        return max(1, int(predicted_rank))

    df["predicted_neet_rank"] = df.apply(lambda row: predict_neet_rank(row["score"], row["accuracy"]), axis=1)
    st.dataframe(df[["score", "accuracy", "predicted_neet_rank"]])

    # Show predicted vs actual ranks (if available)
    if "actual_rank" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=df["actual_rank"], y=df["predicted_neet_rank"], color="blue")
        plt.plot([0, 100000], [0, 100000], linestyle="--", color="red")  # Reference line
        plt.title("Predicted vs Actual NEET Rank")
        plt.xlabel("Actual NEET Rank")
        plt.ylabel("Predicted NEET Rank")
        st.pyplot(plt)

    st.success("✅ NEET Rank Prediction Completed!")
