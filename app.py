import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

st.set_page_config(page_title="Aircraft Engine Risk Dashboard", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

df = pd.read_csv(DATA_DIR / "engine_predictions.csv")

st.sidebar.header("Model Settings")
model_choice = st.sidebar.selectbox("Choose Prediction Model", ["GRU", "LSTM"])

if model_choice == "GRU":
    df["Predicted_RUL"] = df["Predicted_RUL_GRU"]
else:
    df["Predicted_RUL"] = df["Predicted_RUL_LSTM"]

def classify_risk(rul: float) -> str:
    if rul < 30:
        return "🔴 HIGH"
    elif rul < 60:
        return "🟡 MEDIUM"
    return "🟢 LOW"

def recommendation(risk: str) -> str:
    if "HIGH" in risk:
        return "⚠️ Immediate Maintenance"
    elif "MEDIUM" in risk:
        return "🛠 Schedule Inspection"
    return "✅ Normal Operation"

df["Risk_Score"] = 1 - (df["Predicted_RUL"] / 125.0)
df["Risk_Score"] = df["Risk_Score"].clip(0, 1).round(4)
df["Risk_Level"] = df["Predicted_RUL"].apply(classify_risk)
df["Action"] = df["Risk_Level"].apply(recommendation)
df = df.sort_values("Predicted_RUL", ascending=True).reset_index(drop=True)

def create_pdf_report(dataframe: pd.DataFrame, selected_model: str) -> bytes:
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    _, height = letter

    y = height - 40
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(40, y, "Aircraft Engine Safety & Risk Report")

    y -= 22
    pdf.setFont("Helvetica", 10)
    pdf.drawString(40, y, f"Model Used: {selected_model}")
    y -= 15
    pdf.drawString(40, y, f"Total Engines: {len(dataframe)}")
    y -= 15
    pdf.drawString(40, y, f"Average Predicted RUL: {round(dataframe['Predicted_RUL'].mean(), 2)}")
    y -= 15
    pdf.drawString(40, y, f"High Risk Engines: {len(dataframe[dataframe['Risk_Level'].str.contains('HIGH')])}")
    y -= 15
    pdf.drawString(40, y, f"Average Risk Score: {round(dataframe['Risk_Score'].mean(), 2)}")

    y -= 25
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, y, "Top 15 Risk Engines")

    y -= 18
    pdf.setFont("Helvetica-Bold", 9)
    pdf.drawString(40, y, "Engine")
    pdf.drawString(95, y, "True RUL")
    pdf.drawString(155, y, "Pred RUL")
    pdf.drawString(225, y, "Risk")
    pdf.drawString(300, y, "Action")

    y -= 14
    pdf.setFont("Helvetica", 9)

    for _, row in dataframe.head(15).iterrows():
        if y < 50:
            pdf.showPage()
            y = height - 40

        pdf.drawString(40, y, str(int(row["Engine_ID"])))
        pdf.drawString(95, y, str(round(float(row["True_RUL"]), 2)))
        pdf.drawString(155, y, str(round(float(row["Predicted_RUL"]), 2)))
        pdf.drawString(225, y, row["Risk_Level"])
        pdf.drawString(300, y, row["Action"][:28])
        y -= 14

    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()

st.title("✈️ Aircraft Engine Safety & Risk Monitoring System")
st.caption(f"Active Model: {model_choice}")

st.subheader("📊 Fleet Health Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Engines", len(df))
col2.metric("Avg Predicted RUL", int(df["Predicted_RUL"].mean()))
col3.metric("High Risk Engines", len(df[df["Risk_Level"] == "🔴 HIGH"]))
col4.metric("Avg Risk Score", round(df["Risk_Score"].mean(), 2))

st.subheader("🚨 Critical Alerts (Immediate Action Required)")
alerts = df[df["Risk_Level"] == "🔴 HIGH"]
st.dataframe(alerts.head(10), use_container_width=True)

st.subheader("📊 Fleet Risk Distribution")
st.bar_chart(df["Risk_Level"].value_counts())

st.subheader("🔥 Top Risk Engines")
st.dataframe(df.head(15), use_container_width=True)

st.subheader("🔍 Engine Drill-Down Analysis")
engine_id = st.selectbox("Select Engine", df["Engine_ID"])
engine = df[df["Engine_ID"] == engine_id]

st.write(engine)

st.subheader("📈 Prediction vs Actual")
fig1, ax1 = plt.subplots()
ax1.bar(
    ["True RUL", "Predicted RUL"],
    [engine["True_RUL"].values[0], engine["Predicted_RUL"].values[0]]
)
ax1.set_ylabel("Cycles")
ax1.set_title(f"Engine {engine_id} RUL Comparison ({model_choice})")
st.pyplot(fig1)

st.subheader("📈 Predicted RUL Distribution")
fig2, ax2 = plt.subplots()
ax2.hist(df["Predicted_RUL"], bins=30)
ax2.set_title(f"Predicted RUL Distribution ({model_choice})")
ax2.set_xlabel("Predicted RUL")
ax2.set_ylabel("Count")
st.pyplot(fig2)

st.subheader("💾 Export Reports")
csv_data = df.to_csv(index=False).encode("utf-8")
pdf_data = create_pdf_report(df, model_choice)

col_csv, col_pdf = st.columns(2)
with col_csv:
    st.download_button(
        label="Download Fleet Report (CSV)",
        data=csv_data,
        file_name=f"fleet_risk_report_{model_choice.lower()}.csv",
        mime="text/csv"
    )

with col_pdf:
    st.download_button(
        label="Download Fleet Report (PDF)",
        data=pdf_data,
        file_name=f"fleet_risk_report_{model_choice.lower()}.pdf",
        mime="application/pdf"
    )
