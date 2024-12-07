import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# Memuat data dan model
file_path = "D:\Dani\PSD\weather.csv"  # Ganti path dengan path file CSV Anda
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

# Split data untuk training dan testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    st.session_state.tab = "Prediksi Cuaca"

# Tombol untuk Data Asli
if st.sidebar.button("Data Asli"):
    st.session_state.tab = "Data Asli"

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
        # st.write('Prediksi Cuaca (Decision Tree):', dt_pred)
        # st.write('Prediksi Cuaca (KNN):', knn_pred)
        # st.write('Prediksi Cuaca (Gradient Boosting):', gb_pred)

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
elif st.session_state.tab == "Data Asli":
    st.title("Data Asli")
    st.dataframe(df)

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
