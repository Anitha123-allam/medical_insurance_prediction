# Medical Insurance Charges Prediction - simple Streamlit app
# Run:  streamlit run app.py
import json
from datetime import datetime
from html import escape
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

BASE = Path(__file__).parent
st.set_page_config(page_title="Insurance Charges Prediction", page_icon="🏥", layout="wide")

# ---------- Simple styling ----------
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg, #eef4ff, #f3ecff, #e8fbf8);}
[data-testid="stSidebar"] {background: linear-gradient(180deg, #1e3a8a, #6d28d9);}
[data-testid="stSidebar"] * {color: white !important;}
div[data-testid="stMetric"] {background: white; padding: 15px; border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,.08); border-left: 6px solid #7c3aed;}
.stButton > button, .stDownloadButton > button {background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white; border: none; border-radius: 10px; padding: .5rem 1.5rem;}
.title {background: linear-gradient(120deg, #2563eb, #7c3aed, #0d9488); color: white;
    padding: 25px; border-radius: 20px; font-size: 2rem; font-weight: 700; margin-bottom: 20px;}
.result {background: linear-gradient(120deg, #0d9488, #2563eb); color: white; padding: 25px;
    border-radius: 20px; text-align: center; font-size: 2.2rem; font-weight: 700; margin-top: 15px;}
.home-hero {position: relative; overflow: hidden; background: linear-gradient(125deg, #172554, #5b21b6 58%, #0f766e);
    color: white; padding: 38px; border-radius: 26px; margin: 8px 0 28px;
    box-shadow: 0 18px 40px rgba(49, 46, 129, .2);}
.home-hero:after {content: ""; position: absolute; width: 220px; height: 220px; right: 7%; top: -95px;
    border-radius: 50%; background: rgba(255,255,255,.1);}
.hero-eyebrow {display: inline-block; padding: 7px 12px; border-radius: 99px;
    background: rgba(255,255,255,.16); color: #e0e7ff; font-size: .78rem;
    font-weight: 700; letter-spacing: .08em; text-transform: uppercase;}
.home-hero h1 {font-size: clamp(2rem, 4vw, 3.2rem); line-height: 1.08; margin: 18px 0 12px;
    color: white;}
.home-hero p {max-width: 700px; color: #e0e7ff; font-size: 1.08rem; margin: 0;}
.hero-tags {display: flex; flex-wrap: wrap; gap: 9px; margin-top: 22px;}
.hero-tags span {padding: 7px 11px; border: 1px solid rgba(255,255,255,.28);
    border-radius: 99px; color: #f8fafc; font-size: .85rem;}
.home-section-label {color: #475569; font-size: .8rem; font-weight: 800;
    letter-spacing: .1em; margin: 8px 0 12px; text-transform: uppercase;}
.home-stat {min-height: 132px; padding: 17px; border-radius: 18px; color: white;
    box-shadow: 0 10px 24px rgba(15,23,42,.12); margin-bottom: 14px;}
.home-stat.blue {background: linear-gradient(135deg, #2563eb, #06b6d4);}
.home-stat.violet {background: linear-gradient(135deg, #7c3aed, #c026d3);}
.home-stat.teal {background: linear-gradient(135deg, #0f766e, #14b8a6);}
.home-stat.orange {background: linear-gradient(135deg, #ea580c, #f59e0b);}
.home-stat.rose {background: linear-gradient(135deg, #be123c, #f43f5e);}
.home-stat.indigo {background: linear-gradient(135deg, #4338ca, #818cf8);}
.home-stat.cyan {background: linear-gradient(135deg, #0e7490, #22d3ee);}
.home-stat-top {display: flex; align-items: center; gap: 9px; color: rgba(255,255,255,.88);
    font-size: .88rem; font-weight: 600;}
.home-stat-icon {font-size: 1.2rem;}
.home-stat-value {font-size: clamp(1.35rem, 2vw, 1.8rem); font-weight: 800;
    line-height: 1.2; margin-top: 15px; overflow-wrap: anywhere;}
.home-note {padding: 15px 18px; border: 1px solid #c7d2fe; border-radius: 14px;
    background: linear-gradient(100deg, #eef2ff, #ecfeff); color: #334155; margin-top: 5px;}
.footer {text-align: center; color: #64748b; padding-top: 30px;}
</style>
""", unsafe_allow_html=True)

# ---------- Load files ----------
from sklearn.model_selection import train_test_split

import io

def read_any_csv(path):
    text = Path(path).read_text(encoding="utf-8-sig").replace('"', '')
    first = text.splitlines()[0]
    sep = ";" if first.count(";") > first.count(",") else ","
    return pd.read_csv(io.StringIO(text), sep=sep, decimal="," if sep == ";" else ".")

try:
    df = read_any_csv(BASE / "insurance_clean.csv")
    compare = read_any_csv(BASE / "model_comparison.csv")
    model = joblib.load(BASE / "best_insurance_model.pkl")
    info = json.load(open(BASE / "model_info.json"))
except FileNotFoundError as e:
    st.error(f"File not found: {e.filename}. Please run the notebook first (it creates these files).")
    st.stop()

# build the test predictions here (same 80:20 split and random_state as the notebook)
_, X_test, _, y_test = train_test_split(df.drop("charges", axis=1), df["charges"], test_size=0.2, random_state=42)
preds = pd.DataFrame({"Actual": y_test.values, "Predicted": model.predict(X_test)})

# ---------- Session state (for reset button and history) ----------
defaults = {"age": 30, "sex": "male", "bmi": 25.0, "children": 0, "smoker": "no", "region": "southwest"}
for k, v in defaults.items():
    st.session_state.setdefault("in_" + k, v)
st.session_state.setdefault("history", [])
st.session_state.setdefault("result", None)


def reset_form():
    for k, v in defaults.items():
        st.session_state["in_" + k] = v
    st.session_state["result"] = None


def title(text):
    st.markdown(f'<div class="title">{text}</div>', unsafe_allow_html=True)


def home_stat(label, value, icon, color):
    st.markdown(
        f'<div class="home-stat {color}"><div class="home-stat-top">'
        f'<span class="home-stat-icon">{escape(icon)}</span>{escape(label)}</div>'
        f'<div class="home-stat-value">{escape(str(value))}</div></div>',
        unsafe_allow_html=True,
    )


def footer():
    st.markdown('<div class="footer">🏥 Medical Insurance Charges Prediction | Python, Scikit-learn, Streamlit</div>',
                unsafe_allow_html=True)


# ---------- Sidebar navigation ----------
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Data Analysis", "🤖 Model Performance",
                                  "💰 Prediction", "ℹ️ About Project"])

# ============ PAGE 1: HOME ============
if page == "🏠 Home":
    st.markdown("""
    <section class="home-hero">
        <div class="hero-eyebrow">🏥 Medical insurance insights</div>
        <h1>Understand your insurance estimate</h1>
        <p>Explore the data, compare machine-learning models, and estimate yearly medical
        insurance charges using a few simple details.</p>
        <div class="hero-tags">
            <span>Machine-learning powered</span>
            <span>Interactive data insights</span>
            <span>Fast estimates</span>
        </div>
    </section>
    """, unsafe_allow_html=True)
    st.markdown('<div class="home-section-label">Project snapshot</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        home_stat("Total records", f"{len(df):,}", "📋", "blue")
    with c2:
        home_stat("Input features", df.shape[1] - 1, "🧩", "violet")
    with c3:
        home_stat("Best model", info["best_model"], "🏆", "teal")
    with c4:
        home_stat("Test R² score", f'{info["test_r2"]:.3f}', "🎯", "orange")
    c5, c6, c7 = st.columns(3)
    with c5:
        home_stat("Test MAE ($)", f'{info["test_mae"]:,.0f}', "💡", "rose")
    with c6:
        home_stat("Test RMSE ($)", f'{info["test_rmse"]:,.0f}', "📈", "indigo")
    with c7:
        home_stat("CV RMSE ($)", f'{info["cv_rmse"]:,.0f}', "🔎", "cyan")
    st.markdown(
        '<div class="home-note"><strong>Ready to explore?</strong> '
        'Use the sidebar to analyze the dataset, review model performance, or make a prediction.</div>',
        unsafe_allow_html=True,
    )
    footer()

# ============ PAGE 2: DATA ANALYSIS ============
elif page == "📊 Data Analysis":
    title("📊 Data Analysis")
    c1, c2, c3 = st.columns(3)
    smoker = c1.multiselect("Smoker", df["smoker"].unique(), default=list(df["smoker"].unique()))
    sex = c2.multiselect("Gender", df["sex"].unique(), default=list(df["sex"].unique()))
    region = c3.multiselect("Region", df["region"].unique(), default=list(df["region"].unique()))
    data = df[df["smoker"].isin(smoker) & df["sex"].isin(sex) & df["region"].isin(region)]

    if data.empty:
        st.warning("No data for these filters. Please select at least one option in each filter.")
    else:
        st.subheader("Dataset preview")
        st.dataframe(data.head(10))
        a, b = st.columns(2)
        a.subheader("Data types")
        a.dataframe(data.dtypes.astype(str).rename("type"))
        b.subheader("Missing values")
        b.dataframe(data.isnull().sum().rename("missing"))
        st.subheader("Descriptive statistics")
        st.dataframe(data.describe())

        st.subheader("Charts")
        a, b = st.columns(2)
        a.plotly_chart(px.histogram(data, x="charges", nbins=40, title="Charges distribution"))
        b.plotly_chart(px.box(data, x="smoker", y="charges", color="smoker", title="Charges by smoker"))
        a.plotly_chart(px.box(data, x="region", y="charges", color="region", title="Charges by region"))
        b.plotly_chart(px.scatter(data, x="age", y="charges", color="smoker", title="Age vs charges"))
        st.plotly_chart(px.imshow(data.select_dtypes("number").corr(), text_auto=".2f",
                                  title="Correlation heatmap"))
    footer()

# ============ PAGE 3: MODEL PERFORMANCE ============
elif page == "🤖 Model Performance":
    title("🤖 Model Performance")
    st.success(f'Best model: **{info["best_model"]}** (lowest cross-validation RMSE)')
    st.dataframe(compare)
    st.download_button("⬇️ Download model comparison (CSV)", compare.to_csv(index=False), "model_comparison.csv")

    chart_layout = {
        "template": "plotly_white",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#24324a"},
        "title_font": {"color": "#172554", "size": 20},
        "margin": {"l": 20, "r": 30, "t": 60, "b": 20},
    }
    rmse_chart = px.bar(
        compare.sort_values("CV RMSE"),
        x="CV RMSE",
        y="Model",
        orientation="h",
        color="Model",
        color_discrete_sequence=px.colors.qualitative.Bold,
        title="Cross-validation RMSE (smaller is better)",
        labels={"CV RMSE": "CV RMSE ($)", "Model": ""},
        text_auto=",.0f",
    )
    rmse_chart.update_layout(**chart_layout, showlegend=False)
    rmse_chart.update_traces(marker_line_color="white", marker_line_width=1.5, textposition="outside")
    rmse_chart.update_xaxes(showgrid=True, gridcolor="#e2e8f0", zeroline=False)
    rmse_chart.update_yaxes(showgrid=False)
    st.plotly_chart(rmse_chart, width="stretch")

    preds["Residual"] = preds["Actual"] - preds["Predicted"]
    a, b = st.columns(2)
    actual_chart = px.scatter(
        preds, x="Actual", y="Predicted", title="Actual vs Predicted (test data)",
        color_discrete_sequence=["#7c3aed"],
        labels={"Actual": "Actual charges ($)", "Predicted": "Predicted charges ($)"},
    )
    actual_chart.update_layout(**chart_layout)
    actual_chart.update_traces(marker={"size": 9, "opacity": 0.75, "line": {"width": 1, "color": "white"}})
    a.plotly_chart(actual_chart, width="stretch")

    residual_chart = px.scatter(
        preds, x="Predicted", y="Residual", title="Residual plot",
        color_discrete_sequence=["#0891b2"],
        labels={"Predicted": "Predicted charges ($)", "Residual": "Residual ($)"},
    )
    residual_chart.update_layout(**chart_layout)
    residual_chart.update_traces(marker={"size": 9, "opacity": 0.75, "line": {"width": 1, "color": "white"}})
    b.plotly_chart(residual_chart, width="stretch")

    residual_histogram = px.histogram(
        preds, x="Residual", nbins=30, title="Residual distribution",
        color_discrete_sequence=["#f97316"], labels={"Residual": "Residual ($)"},
    )
    residual_histogram.update_layout(**chart_layout)
    residual_histogram.update_traces(marker_line_color="white", marker_line_width=1)
    st.plotly_chart(residual_histogram, width="stretch")
    footer()

# ============ PAGE 4: PREDICTION ============
elif page == "💰 Prediction":
    title("💰 Insurance Charges Prediction")
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age", min_value=18, max_value=100, step=1, key="in_age")
    sex = c2.selectbox("Gender", ["male", "female"], key="in_sex")
    bmi = c3.number_input("BMI", min_value=10.0, max_value=60.0, step=0.1, key="in_bmi")
    c4, c5, c6 = st.columns(3)
    children = c4.number_input("Children", min_value=0, max_value=10, step=1, key="in_children")
    smoker = c5.selectbox("Smoker", ["no", "yes"], key="in_smoker")
    region = c6.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"], key="in_region")
    rate = st.number_input("USD to INR rate (change to today's rate)", min_value=1.0, value=88.0, step=0.5)

    b1, b2, _ = st.columns([1.5, 1, 4])
    predict = b1.button("🔮 Predict Insurance Charges")
    b2.button("🔄 Reset", on_click=reset_form)

    if predict:
        if not (18 <= age <= 100 and 10 <= bmi <= 60 and 0 <= children <= 10):
            st.error("Please enter valid values: age 18-100, BMI 10-60, children 0-10.")
        else:
            row = pd.DataFrame([{"age": age, "sex": sex, "bmi": bmi, "children": children,
                                 "smoker": smoker, "region": region}])
            usd = float(model.predict(row)[0])
            st.session_state["result"] = (usd, usd * rate)
            st.session_state["history"].append({
                "Time": datetime.now().strftime("%H:%M:%S"), **row.iloc[0].to_dict(),
                "Model": info["best_model"], "Predicted USD": round(usd, 2), "Predicted INR": round(usd * rate, 2)})

    if st.session_state["result"]:
        usd, inr = st.session_state["result"]
        st.markdown(f'<div class="result">Estimated charges: ₹ {inr:,.0f}<br>'
                    f'<span style="font-size:1rem">≈ $ {usd:,.0f} | Model: {info["best_model"]}</span></div>',
                    unsafe_allow_html=True)

    st.warning("This is only an estimate based on a dataset. It is NOT an actual insurance quote.")

    if st.session_state["history"]:
        st.subheader("Prediction history (this session)")
        history = pd.DataFrame(st.session_state["history"])
        st.dataframe(history)
        st.download_button("⬇️ Download predictions (CSV)", history.to_csv(index=False), "predictions.csv")
    footer()

# ============ PAGE 5: ABOUT ============
else:
    title("ℹ️ About Project")
    st.markdown("""
**Objective:** estimate medical insurance charges from personal details.

**Dataset:** insurance data with 7 columns (age, sex, bmi, children, smoker, region, charges).

**Algorithms:** Linear Regression, Polynomial Regression (degree 2 and 3), Decision Tree, KNN, Random Forest.

**Technologies:** Python, Pandas, Matplotlib, Seaborn, Plotly, Scikit-learn, Streamlit.

**Evaluation:** data split 80:20. Models compared with 5-fold cross-validation RMSE on training data.
The test data was used only once, for the best model.

**Limitations:** small dataset, charges are in US dollars (converted to ₹ with the rate you enter),
and the result is an estimate, not a real quote.

**Future work:** more features, gradient boosting, real Indian insurance data.
""")
    footer()
