import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Memuat data dan model
file_path = "weather.csv" 
df = pd.read_csv(file_path)

# Drop date
df = df.drop('date', axis=1)

# Normalisasi data
columns_to_normalize = ['temp_max', 'temp_min', 'precipitation', 'wind']
scaler = MinMaxScaler()
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Pisahkan fitur dan target
X = df.drop('weather', axis=1)
y = df['weather']

# Dropdown untuk memilih rasio split data
split_ratio = st.selectbox("Pilih rasio split data:", ["70/30", "80/20", "90/10"], index=1)

# Menentukan test_size berdasarkan pilihan dropdown
if split_ratio == "70/30":
    test_size = 0.3
elif split_ratio == "80/20":
    test_size = 0.2
else:
    test_size = 0.1

# Split data untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Model Decision Tree
dt_model = DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=42)
dt_model.fit(X_train, y_train)

# Model KNN
knn_model = KNeighborsClassifier(n_neighbors=6)
knn_model.fit(X_train, y_train)

# Model Gradient Boosting
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

# Fungsi untuk melakukan prediksi
def predict_weather(precipitation, temp_max, temp_min, wind):
    features = np.array([[precipitation, temp_max, temp_min, wind]])
    features = scaler.transform(features)
    # Prediksi dari masing-masing model
    dt_pred = dt_model.predict(features)
    knn_pred = knn_model.predict(features)
    gb_pred = gb_model.predict(features)

    return dt_pred[0], knn_pred[0], gb_pred[0]


st.sidebar.title("Pilih Tab")

if 'tab' not in st.session_state:
    st.session_state.tab = "Home"

# Tombol untuk Data Asli
if st.sidebar.button("Home"):
    st.session_state.tab = "Home"

# Tombol untuk Prediksi Cuaca
if st.sidebar.button("Prediksi Cuaca"):
    st.session_state.tab = "Prediksi Cuaca"

if st.sidebar.button("Grafik"):
    st.session_state.tab = "Grafik"

# Fungsi untuk menghitung akurasi setelah prediksi
def calculate_accuracy(model, model_name):
    if model_name == "Decision Tree":
        y_pred = dt_model.predict(X_test)
        return accuracy_score(y_test, y_pred)
    elif model_name == "KNN":
        y_pred = knn_model.predict(X_test)
        return accuracy_score(y_test, y_pred)
    elif model_name == "Gradient Boosting":
        y_pred = gb_model.predict(X_test)
        return accuracy_score(y_test, y_pred)

# Konten untuk Prediksi Cuaca
if st.session_state.tab == "Prediksi Cuaca":
    st.title("Prediksi Cuaca")
    # Form input untuk pengguna
    precipitation = st.number_input('Precipitation', min_value=0.0, step=0.1)
    temp_max = st.number_input('Temp Max', min_value=-50.0, max_value=50.0, step=0.1)
    temp_min = st.number_input('Temp Min', min_value=-50.0, max_value=50.0, step=0.1)
    wind = st.number_input('Wind', min_value=0.0, step=0.1)

    if st.button('Prediksi'):
        # Lakukan prediksi berdasarkan input
        dt_pred, knn_pred, gb_pred = predict_weather(precipitation, temp_max, temp_min, wind)

        # Tampilkan prediksi
        predict = {
            'Model': ['Decision Tree', 'KNN', 'Gradient Boosting'],
            'Prediksi': [dt_pred, knn_pred, gb_pred]
        }
        st.write('Prediksi Cuaca :')
        st.dataframe(predict)

        # Tampilkan akurasi dari masing-masing model setelah prediksi
        dt_accuracy = calculate_accuracy(dt_pred, "Decision Tree")
        knn_accuracy = calculate_accuracy(knn_pred, "KNN")
        gb_accuracy = calculate_accuracy(gb_pred, "Gradient Boosting")

        accuracy_data = {
            'Model': ['Decision Tree', 'KNN', 'Gradient Boosting'],
            'Akurasi': [dt_accuracy, knn_accuracy, gb_accuracy]
        }
        st.write("Akurasi Model :")
        st.table(accuracy_data)

# Konten untuk Data Asli
elif st.session_state.tab == "Home":
    st.title("Halaman Utama")
    st.write("Selamat Datang di Aplikasi Prediksi Cuaca, Silahkan pilih tab yang tersedia dan aplikasi akan memberikan prediksi cuaca berdasarkan input yang diberikan.")
    st.header("Data Asli")
    data_awal = pd.read_csv(file_path)
    data_awal = data_awal.drop('date', axis=1)
    st.dataframe(data_awal)

    # Normalisasi data
    st.header("Data Setelah Normalisasi")
    st.dataframe(scaler.fit_transform(df[columns_to_normalize]))

    # Klasifikasi Decision Tree
    st.header("Klasifikasi Decision Tree")
    st.write("Akurasi Model Decision Tree : ", calculate_accuracy(dt_model, "Decision Tree"))
    # Evaluate model
    dt_y_pred = dt_model.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_y_pred)
    dt_precision = precision_score(y_test, dt_y_pred, average='weighted')
    dt_recall = recall_score(y_test, dt_y_pred, average='weighted')
    dt_f1 = f1_score(y_test, dt_y_pred, average='weighted')
    metrics_df = pd.DataFrame({
        'Matriks': ['Akurasi', 'Presisi', 'Recall', 'F1 Score'],
        'Nilai': [dt_accuracy, dt_precision, dt_recall, dt_f1]
    })
    st.write(metrics_df)

    # Klasifikasi KNN
    st.header("Klasifikasi KNN")
    st.write("Akurasi Model KNN : ", calculate_accuracy(knn_model, "KNN"))
    # Evaluate model
    knn_y_pred = knn_model.predict(X_test)
    knn_accuracy = accuracy_score(y_test, knn_y_pred)
    knn_precision = precision_score(y_test, knn_y_pred, average='weighted')
    knn_recall = recall_score(y_test, knn_y_pred, average='weighted')
    knn_f1 = f1_score(y_test, knn_y_pred, average='weighted')
    metrics_df = pd.DataFrame({
        'Matriks': ['Akurasi', 'Presisi', 'Recall', 'F1 Score'],
        'Nilai': [knn_accuracy, knn_precision, knn_recall, knn_f1]
    })
    st.write(metrics_df)

    # Klasifikasi Gradient Boosting
    st.header("Klasifikasi Gradient Boosting")
    st.write("Akurasi Model Gradient Boosting : ", calculate_accuracy(gb_model, "Gradient Boosting"))
    # Evaluate model
    gb_y_pred = gb_model.predict(X_test)
    gb_accuracy = accuracy_score(y_test, gb_y_pred)
    gb_precision = precision_score(y_test, gb_y_pred, average='weighted')
    gb_recall = recall_score(y_test, gb_y_pred, average='weighted')
    gb_f1 = f1_score(y_test, gb_y_pred, average='weighted')
    metrics_df = pd.DataFrame({
        'Matriks': ['Akurasi', 'Presisi', 'Recall', 'F1 Score'],
        'Nilai': [gb_accuracy, gb_precision, gb_recall, gb_f1]
    })
    st.write(metrics_df)

# Konten untuk Grafik
elif st.session_state.tab == "Grafik":
    st.title("Grafik")
    st.write("Grafik Akurasi Model")
    models = ['Decision Tree', 'KNN', 'Gradient Boosting']
    accuracy = [0.0, 0.0, 0.0]
    accuracy[0] = calculate_accuracy(dt_model, "Decision Tree")
    accuracy[1] = calculate_accuracy(knn_model, "KNN")
    accuracy[2] = calculate_accuracy(gb_model, "Gradient Boosting")
    plt.bar(models, accuracy)
    st.pyplot(plt)
