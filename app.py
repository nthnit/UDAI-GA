import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Wine Quality Prediction", layout="wide")

# ----------------- ĐỌC FILE TRANSLATIONS -----------------
@st.cache_data
def load_translations():
    with open("lang/translations.json", "r", encoding="utf-8") as f:
        return json.load(f)

translations = load_translations()

# ----------------- CHỌN NGÔN NGỮ -----------------
selected_lang_display = st.sidebar.selectbox("Select Language / Chọn Ngôn ngữ", ["English", "Tiếng Việt"])
language_mapping = {"English": "en", "Tiếng Việt": "vi"}
selected_language = language_mapping[selected_lang_display]

def tr(key):
    return translations[selected_language].get(key, key)

# ----------------- LOAD DỮ LIỆU -----------------
@st.cache_data
def load_data():
    data = pd.read_csv("Data/winequality-red.csv", sep=";")
    return data

data = load_data()

# ----------------- NAVIGATION -----------------
nav = st.sidebar.radio("Navigation", [tr("nav1"), tr("nav2"), tr("nav3"), tr("nav4")])

# ----------------- SECTION 1: INTRODUCTION -----------------
if nav == tr("nav1"):
    st.title(tr("title"))
    st.header(tr("section1_header"))
    st.markdown(tr("section1_body"))

# ----------------- SECTION 2: EDA -----------------
elif nav == tr("nav2"):
    st.header(tr("section2_header"))
    st.subheader(tr("section2_overview"))
    st.write(tr("section2_shape"), data.shape)
    st.write(tr("section2_columns"), list(data.columns))
    if st.checkbox(tr("section2_head")):
        st.write(data.head())
    
    st.sidebar.header(tr("section2_sidebar_header"))
    eda_option = st.sidebar.radio("Choose Visualization:", 
                                  (tr("section2_sidebar_option1"), tr("section2_sidebar_option2"), tr("section2_sidebar_option3")))
    
    if eda_option == tr("section2_sidebar_option1"):
        st.subheader(tr("section2_sidebar_option1"))
        st.write(data.head())
    elif eda_option == tr("section2_sidebar_option2"):
        st.subheader(tr("section2_sidebar_option2"))
        col_to_plot = st.selectbox("Choose a column:", data.columns)
        fig, ax = plt.subplots()
        ax.hist(data[col_to_plot].dropna(), bins=30, color='skyblue', edgecolor='black')
        ax.set_title(f"Distribution of {col_to_plot}")
        ax.set_xlabel(col_to_plot)
        ax.set_ylabel("Frequency")
        st.pyplot(fig, use_container_width=False)

    elif eda_option == tr("section2_sidebar_option3"):
        st.subheader(tr("section2_sidebar_option3"))
        corr = data.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig, use_container_width=False)


# ----------------- SECTION 3: PREPROCESSING & TRAINING -----------------
elif nav == tr("nav3"):
    st.header(tr("section3_header"))
    if "quality" in data.columns:
        X = data.drop("quality", axis=1)
        y = data["quality"]
    else:
        st.error(tr("error_quality_missing"))
        st.stop()
    
    test_size_percent = st.slider(tr("section3_test_size"), min_value=10, max_value=50, value=20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_percent/100, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    k = st.slider(tr("section3_k"), min_value=1, max_value=20, value=5, step=1)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    
    y_pred = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    st.subheader(tr("section3_model_eval"))
    st.write("Accuracy:", accuracy)
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    
    st.session_state.scaler = scaler
    st.session_state.knn = knn

# ----------------- SECTION 4: PREDICTION -----------------
elif nav == tr("nav4"):
    st.header(tr("section4_header"))
    if "knn" not in st.session_state or "scaler" not in st.session_state:
        st.warning("Please go to Section 3: Preprocessing & Training first.")
    else:
        scaler = st.session_state.scaler
        knn = st.session_state.knn
        
        st.markdown(tr("section4_description"))
        fixed_acidity = st.number_input("Fixed Acidity", value=float(data["fixed acidity"].mean()))
        volatile_acidity = st.number_input("Volatile Acidity", value=float(data["volatile acidity"].mean()))
        citric_acid = st.number_input("Citric Acid", value=float(data["citric acid"].mean()))
        residual_sugar = st.number_input("Residual Sugar", value=float(data["residual sugar"].mean()))
        chlorides = st.number_input("Chlorides", value=float(data["chlorides"].mean()))
        free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=float(data["free sulfur dioxide"].mean()))
        total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=float(data["total sulfur dioxide"].mean()))
        density = st.number_input("Density", value=float(data["density"].mean()))
        pH = st.number_input("pH", value=float(data["pH"].mean()))
        sulphates = st.number_input("Sulphates", value=float(data["sulphates"].mean()))
        alcohol = st.number_input("Alcohol", value=float(data["alcohol"].mean()))
        
        if st.button(tr("predict_button")):
            input_features = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                                        free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])
            input_scaled = scaler.transform(input_features)
            prediction = knn.predict(input_scaled)
            st.success(f"Predicted Wine Quality: {prediction[0]}")
