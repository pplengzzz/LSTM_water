import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ตั้งค่าหน้าเว็บ Streamlit
st.set_page_config(page_title='การพยากรณ์ด้วย LSTM', page_icon=':ocean:')

# ชื่อของแอป
st.title("และการพยากรณ์ด้วย LSTM")

# ฟังก์ชันสร้างข้อมูลสำหรับ LSTM
def create_dataset(data, look_back=15):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

# ฟังก์ชันคำนวณความแม่นยำ
def calculate_accuracy(filled_data, original_data, original_nan_indexes):
    actual_values = original_data.loc[original_nan_indexes, 'wl_up']
    predicted_values = filled_data.loc[original_nan_indexes, 'wl_up']
    
    # คำนวณ MAE และ RMSE
    mae = mean_absolute_error(actual_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))

    st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
    st.write(f"Root Mean Square Error (RMSE): {rmse:.4f}")

# ฟังก์ชันพยากรณ์ค่าระดับน้ำด้วย LSTM
def predict_water_level_lstm(df, model_path, look_back=15):
    # โหลดโมเดล LSTM ที่ฝึกแล้ว
    model = load_model(model_path)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[['wl_up']])

    # เตรียมข้อมูลสำหรับ LSTM
    X, _ = create_dataset(df_scaled, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # พยากรณ์ข้อมูล
    predictions = model.predict(X)

    # Inverse scaling
    predictions = scaler.inverse_transform(predictions)

    # สร้าง DataFrame สำหรับค่าที่ถูกพยากรณ์
    df_predictions = df.copy()
    df_predictions.iloc[look_back:, 0] = predictions.flatten()  # เติมค่าที่ทำนายได้ ไม่ใช่ค่าจริง

    return df_predictions

# อัปโหลดไฟล์ CSV ข้อมูลจริง
uploaded_file = st.file_uploader("เลือกไฟล์ CSV ข้อมูลจริง", type="csv")

if uploaded_file is not None:
    # โหลดข้อมูลจริง
    data = pd.read_csv(uploaded_file)
    data['datetime'] = pd.to_datetime(data['datetime'])
    
    # ทำให้ datetime เป็น tz-naive (ไม่มี timezone)
    data['datetime'] = data['datetime'].dt.tz_localize(None)
    
    data.set_index('datetime', inplace=True)

    # ให้ผู้ใช้เลือกช่วงวันที่ที่ต้องการตัดข้อมูล
    st.subheader("เลือกช่วงวันที่ที่ต้องการตัดข้อมูล")
    start_date = st.date_input("เลือกวันเริ่มต้น", pd.to_datetime(data.index.min()).date())
    end_date = st.date_input("เลือกวันสิ้นสุด", pd.to_datetime(data.index.max()).date())

    # รวมวันและเวลาที่เลือกเข้าด้วยกันเป็นช่วงเวลา
    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date) + pd.DateOffset(days=1) - pd.Timedelta(seconds=1)  # ให้ครอบคลุมทั้งวันสิ้นสุด

    # ตรวจสอบว่ามีข้อมูลในช่วงวันที่ที่เลือกหรือไม่
    if not data.index.isin(pd.date_range(start=start_datetime, end=end_datetime)).any():
        st.error("ไม่มีข้อมูลในช่วงวันที่ที่เลือก กรุณาเลือกช่วงวันที่ที่มีข้อมูล")
    else:
        if st.button("ตัดข้อมูล"):
            # ข้อมูลก่อนและหลังช่วงเวลาที่ตัดออก (สำหรับฝึกโมเดล)
            train_data = data[(data.index < start_datetime) | (data.index > end_datetime)]

            # ข้อมูลช่วงเวลาที่ถูกตัดออก (สำหรับเติมข้อมูล)
            missing_data = data[(data.index >= start_datetime) & (data.index <= end_datetime)]

            # สร้างสำเนาของข้อมูลก่อนถูกตัดออกเพื่อพล็อตกราฟ
            original_missing_data = missing_data.copy()

            # พยากรณ์ข้อมูลที่ถูกตัดออกด้วยโมเดล LSTM
            filled_missing_data = predict_water_level_lstm(missing_data, "lstm_2024_50epochs.keras")

            # รวมข้อมูลทั้งหมด
            final_data = pd.concat([train_data, filled_missing_data]).sort_index()

            # เก็บตำแหน่ง NaN ก่อนเติมค่า
            original_nan_indexes = filled_missing_data.iloc[15:].index

            # คำนวณความแม่นยำ
            calculate_accuracy(filled_missing_data, original_missing_data, original_nan_indexes)

            # แสดงกราฟข้อมูลที่ถูกเติม
            plt.figure(figsize=(14, 7))

            # ข้อมูลที่มีอยู่แล้ว
            plt.plot(train_data.index, train_data['wl_up'], label='Existing Data', color='blue')

            # ข้อมูลที่ถูกเติม
            plt.plot(filled_missing_data.index, filled_missing_data['wl_up'], label='Filled Data (LSTM)', color='green')

            # ข้อมูลที่ถูกตัดออก (แสดงค่าจริงก่อนเติม)
            plt.plot(original_missing_data.index, original_missing_data['wl_up'], label='Original Missing Data', color='orange', linestyle='dotted')

            plt.xlabel('Date')
            plt.ylabel('Water Level (wl_up)')
            plt.title('Water Level over Time with LSTM-filled Data')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)

            # แสดงผลลัพธ์การเติมค่าเป็นตาราง
            st.subheader('ตารางข้อมูลที่เติมค่า (datetime, wl_up)')
            st.write(filled_missing_data[['wl_up']])




