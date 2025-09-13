import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import plotly.express as px

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("out/model.joblib")

model = load_model()

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Hospital LOS Predictor", layout="wide")
st.title("🏥 Hospital Length of Stay (LOS) Predictor with Explainability & Fairness")

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Predict Patient LOS",
    "📊 Feature Importance Dashboard",
    "⚖️ Fairness Report",
    "📈 Model Evaluation"
])

# -------------------------------
# Tab 1: Prediction + Interactive SHAP
# -------------------------------
with tab1:
    st.subheader("Enter Patient Details")
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    gender = st.selectbox("Gender", ["Male", "Female"])
    admission_type = st.selectbox("Admission Type", ["Emergency", "Elective", "Urgent"])
    comorbidities = st.number_input("Comorbidities", min_value=0, max_value=10, value=2)
    procedures = st.number_input("Procedures", min_value=0, max_value=20, value=3)
    labscore = st.slider("Lab Score", min_value=0, max_value=100, value=50)

    if st.button("Predict LOS"):
        new_data = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "AdmissionType": admission_type,
            "Comorbidities": comorbidities,
            "Procedures": procedures,
            "LabScore": labscore
        }])

        # Prediction
        pred = model.predict(new_data)[0]

        # Custom Alert
        if pred > 15:
            st.error(f"⚠️ High Risk: Predicted LOS = {pred:.1f} days")
        else:
            st.success(f"Predicted LOS: **{pred:.1f} days**")

                    # -------------------------------
        # SHAP Explainability (Interactive)
        # -------------------------------
        st.subheader("🔍 Why this prediction?")
        try:
            preproc = model.named_steps["preproc"]
            core_model = model.named_steps["model"]

            # Transform input
            transformed = preproc.transform(new_data)

            # Get feature names
            try:
                feature_names = preproc.get_feature_names_out()
            except:
                feature_names = new_data.columns

            # SHAP explainer
            explainer = shap.TreeExplainer(core_model)
            shap_values = explainer.shap_values(transformed)

            # Prepare DataFrame for Plotly
            shap_df = pd.DataFrame({
                "Feature": feature_names,
                "SHAP Value": shap_values[0]
            }).sort_values("SHAP Value", key=abs, ascending=False)

            # Interactive Plotly horizontal bar
            fig = px.bar(
                shap_df,
                x="SHAP Value",
                y="Feature",
                orientation='h',
                color="SHAP Value",
                color_continuous_scale=['#4caf50', '#ffffff', '#ff4c4c'],
                hover_data={"Feature": True, "SHAP Value": ':.2f'}
            )
            fig.update_layout(
                title="Feature Contributions to LOS Prediction",
                yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.warning(f"SHAP explanation not available: {e}")

# -------------------------------
# Tab 2: Feature Importance
# -------------------------------
with tab2:
    st.subheader("📊 Global Feature Importance")
    try:
        importances = model.named_steps["model"].feature_importances_
        feature_names = model.named_steps["preproc"].get_feature_names_out()
        importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        importance_df = importance_df.sort_values("Importance", ascending=False)

        # Horizontal bar chart with gradient
        colors = plt.cm.viridis(importance_df["Importance"] / importance_df["Importance"].max())
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(importance_df["Feature"], importance_df["Importance"], color=colors)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Feature importance not available: {e}")

# -------------------------------
# Tab 3: Fairness Report
# -------------------------------
with tab3:
    st.subheader("⚖️ Fairness Report")
    try:
        df = pd.read_csv("data/hospital_los.csv")
        X = df.drop(columns=["LOS"])
        y = df["LOS"]
        preds = model.predict(X)

        results_df = pd.DataFrame({
            "Gender": X["Gender"],
            "Age": X["Age"],
            "TrueLOS": y,
            "PredLOS": preds
        })

        # Gender-wise MAE
        st.markdown("### 📌 By Gender")
        gender_results = (
            results_df.groupby("Gender", group_keys=False)
            .apply(lambda g: mean_absolute_error(g["TrueLOS"], g["PredLOS"]))
            .reset_index(name="MAE")
        )
        max_mae = gender_results["MAE"].max()
        gender_colors = ["#ff4c4c" if val == max_mae else "#4caf50" for val in gender_results["MAE"]]
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(gender_results["Gender"], gender_results["MAE"], color=gender_colors)
        ax.set_ylabel("MAE")
        ax.set_title("MAE by Gender")
        st.pyplot(fig)

        # Age group-wise MAE
        st.markdown("### 📌 By Age Groups (<40, 40–60, >60)")
        bins = [0, 40, 60, 200]
        labels = ["<40", "40-60", ">60"]
        results_df["AgeGroup"] = pd.cut(results_df["Age"], bins=bins, labels=labels, right=False)

        age_results = (
            results_df.groupby("AgeGroup", observed=False, group_keys=False)
            .apply(lambda g: mean_absolute_error(g["TrueLOS"], g["PredLOS"]))
            .reset_index(name="MAE")
        )
        max_age_mae = age_results["MAE"].max()
        age_colors = ["#ff4c4c" if val == max_age_mae else "#4caf50" for val in age_results["MAE"]]
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(age_results["AgeGroup"].astype(str), age_results["MAE"], color=age_colors)
        ax.set_ylabel("MAE")
        ax.set_title("MAE by Age Group")
        st.pyplot(fig)

        # Warning if high differences
        if abs(gender_results["MAE"].max() - gender_results["MAE"].min()) > 2 or \
           abs(age_results["MAE"].max() - age_results["MAE"].min()) > 2:
            st.warning("⚠️ Model shows performance differences across groups.")
        else:
            st.success("✅ Model performs fairly evenly across Gender and Age groups.")

    except Exception as e:
        st.warning(f"Fairness analysis not available: {e}")

# -------------------------------
# Tab 4: Interactive Model Evaluation
# -------------------------------
with tab4:
    st.subheader("📊 Overall Model Performance & Group-wise MAE")

    try:
        df = pd.read_csv("data/hospital_los.csv")
        X = df.drop(columns=["LOS"])
        y_true = df["LOS"]
        y_pred = model.predict(X)

        # Overall metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # fixed
        r2 = r2_score(y_true, y_pred)

        st.markdown("### 🔹 Overall Metrics")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.2f} days")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f} days")
        st.write(f"**R² Score:** {r2:.2f}")

        # Prepare results DataFrame for group-wise MAE
        results_df = pd.DataFrame({
            "Gender": X["Gender"],
            "Age": X["Age"],
            "TrueLOS": y_true,
            "PredLOS": y_pred
        })

        # Gender-wise MAE interactive
        gender_results = (
            results_df.groupby("Gender", group_keys=False)
            .apply(lambda g: mean_absolute_error(g["TrueLOS"], g["PredLOS"]))
            .reset_index(name="MAE")
        )
        max_mae = gender_results["MAE"].max()
        gender_results["Color"] = gender_results["MAE"].apply(lambda x: '#ff4c4c' if x == max_mae else '#4caf50')

        fig = px.bar(
            gender_results,
            x="Gender",
            y="MAE",
            color="Color",
            color_discrete_map="identity",
            text="MAE",
            hover_data={"Gender": True, "MAE": ':.2f'}
        )
        fig.update_layout(title="MAE by Gender", yaxis_title="MAE", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Age group-wise MAE interactive
        bins = [0, 40, 60, 200]
        labels = ["<40", "40-60", ">60"]
        results_df["AgeGroup"] = pd.cut(results_df["Age"], bins=bins, labels=labels, right=False)

        age_results = (
            results_df.groupby("AgeGroup", observed=False, group_keys=False)
            .apply(lambda g: mean_absolute_error(g["TrueLOS"], g["PredLOS"]))
            .reset_index(name="MAE")
        )
        max_age_mae = age_results["MAE"].max()
        age_results["Color"] = age_results["MAE"].apply(lambda x: '#ff4c4c' if x == max_age_mae else '#4caf50')

        fig = px.bar(
            age_results,
            x="AgeGroup",
            y="MAE",
            color="Color",
            color_discrete_map="identity",
            text="MAE",
            hover_data={"AgeGroup": True, "MAE": ':.2f'}
        )
        fig.update_layout(title="MAE by Age Group", yaxis_title="MAE", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Highlight performance issues
        if abs(gender_results["MAE"].max() - gender_results["MAE"].min()) > 2 or \
           abs(age_results["MAE"].max() - age_results["MAE"].min()) > 2:
            st.warning("⚠️ Model shows differences in prediction error across groups.")
        else:
            st.success("✅ Model performs fairly evenly across groups.")

    except Exception as e:
        st.warning(f"Model evaluation not available: {e}")
