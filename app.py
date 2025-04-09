import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Hàm load dữ liệu, sử dụng cache của Streamlit để không phải load lại nhiều lần
@st.cache_data 
def load_data(): 
    data = pd.read_csv("Data/winequality-red.csv", sep=";") 
    return data

# Load dữ liệu
data = load_data()

# Giới thiệu ứng dụng
st.title("Wine Quality Prediction using k-NN")
st.write("This application predicts the quality of red wine using a k-Nearest Neighbors (k-NN) classifier.")
st.write("Dataset features include:")
st.write(list(data.columns))

# Hiển thị dữ liệu thô nếu người dùng muốn
if st.checkbox("Show Raw Data"):
    st.write(data.head())

# Tách các biến đặc trưng (features) và biến mục tiêu (quality)
X = data.drop("quality", axis=1)
y = data["quality"]

# Cho phép người dùng chọn tỷ lệ phần trăm dữ liệu dùng cho tập test
test_size_percent = st.slider("Choose Test Set Size (%)", min_value=10, max_value=50, value=20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_percent/100, random_state=42)

# Chuẩn hóa dữ liệu với StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Cho phép người dùng điều chỉnh tham số k (số lân cận) cho k-NN
k = st.slider("Choose k (number of neighbors)", min_value=1, max_value=20, value=5, step=1)

# Huấn luyện mô hình k-NN
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)

# Dự đoán trên tập test
y_pred = knn.predict(X_test_scaled)

# Hiển thị các chỉ số đánh giá
accuracy = accuracy_score(y_test, y_pred)
st.subheader("Model Evaluation on Test Data")
st.write("Accuracy:", accuracy)
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))
st.write("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Phần dự đoán mới dựa trên đầu vào từ người dùng
st.header("Predict Wine Quality from New Input")
st.write("Enter wine properties below:")

# Lấy giá trị trung bình của từng thuộc tính để làm giá trị mặc định cho người dùng
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

# Khi nhấn nút dự đoán
if st.button("Predict Quality"):
    # Tạo mảng đầu vào cho mô hình
    input_features = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                                free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])
    # Áp dụng bộ scaler
    input_scaled = scaler.transform(input_features)
    prediction = knn.predict(input_scaled)
    st.success(f"Predicted Wine Quality: {prediction[0]}")
