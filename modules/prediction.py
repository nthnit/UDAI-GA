import streamlit as st
import numpy as np

def show(data, tr):
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
