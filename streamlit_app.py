import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import altair as alt

# ตั้งค่าหน้าเว็บ Streamlit
st.set_page_config(page_title='Water Level Prediction (LSTM)', page_icon=':ocean:')

# ชื่อของแอป
st.title("การจัดการข้อมูลระดับน้ำและการพยากรณ์ด้วย LSTM")

# อัปโหลดไฟล์ CSV
uploaded_file = st.file_uploader("เลือกไฟล์ CSV", type="csv")

# ฟังก์ชันสำหรับการทำนาย
def predict_water_level_lstm(df, model_path, time_steps=120, n_future=288):  # เปลี่ยน n_future เป็น 288 จุด (3 วัน)
    # แปลงคอลัมน์ 'datetime' ให้เป็น datetime
    df['datetime'] = pd.to_datetime(df['datetime'])

    # เลือกเฉพาะคอลัมน์ 'wl_up' เพื่อทำนาย
    df_selected = df[['wl_up']]

    # สร้าง StandardScaler
    scaler = StandardScaler()

    # ปรับข้อมูล
    df_scaled = scaler.fit_transform(df_selected)

    # ใช้ข้อมูลย้อนหลังล่าสุดสำหรับการทำนาย
    x_input = df_scaled[-time_steps:]
    x_input = np.reshape(x_input, (1, time_steps, 1))  # reshape ให้ตรงตามอินพุตของโมเดล

    # โหลดโมเดล LSTM ที่ถูกฝึกไว้ล่วงหน้า
    model = load_model(model_path)

    # ทำนายทีละขั้น (Step-by-step Prediction)
    predictions = []
    for _ in range(n_future):
        predicted = model.predict(x_input)
        predictions.append(predicted[0, 0])

        # ใช้ค่าทำนายเป็นข้อมูลย้อนหลังสำหรับการทำนายครั้งถัดไป
        predicted = np.reshape(predicted, (1, 1, 1))  # ปรับขนาด predicted เป็น (1, 1, 1)
        x_input = np.append(x_input[:, 1:, :], predicted, axis=1)

    # นำผลลัพธ์กลับไปสู่ขอบเขตเดิม (inverse transform)
    predictions_original_scale = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    # สร้าง DataFrame จากผลลัพธ์ทำนาย
    last_date = pd.to_datetime(df['datetime'].iloc[-1])  # ใช้คอลัมน์ datetime เพื่อหาวันสุดท้าย
    future_dates = pd.date_range(last_date, periods=n_future + 1, freq='15T')[1:]  # สร้างช่วงเวลาสำหรับ 3 วันข้างหน้า

    df_predictions = pd.DataFrame(predictions_original_scale, columns=['prediction_wl_up'])
    df_predictions['datetime'] = future_dates
    df_predictions.set_index('datetime', inplace=True)

    return df_predictions

# ฟังก์ชันสำหรับการพล๊อตกราฟ
def plot_results(df_actual, df_predicted):
    # ข้อมูลจริง (Actual Data)
    data_actual = pd.DataFrame({
        'datetime': df_actual['datetime'],
        'Water Level': df_actual['wl_up'],
        'Type': 'Actual'
    })

    # ข้อมูลทำนาย (Predicted Data)
    data_predicted = pd.DataFrame({
        'datetime': df_predicted.index,
        'Water Level': df_predicted['prediction_wl_up'],
        'Type': 'Predicted'
    })

    # รวมข้อมูลทั้งสองเข้าด้วยกัน
    combined_data = pd.concat([data_actual, data_predicted])

    # สร้างกราฟด้วย Altair และปรับแกน Y ไม่ให้เริ่มจาก 0
    y_min = combined_data['Water Level'].min() - 5  # ลดค่าต่ำสุดเพื่อให้เห็นการเปลี่ยนแปลงชัดเจนขึ้น
    y_max = combined_data['Water Level'].max() + 5  # เพิ่มค่าสูงสุด

    chart = alt.Chart(combined_data).mark_line().encode(
        x='datetime:T',
        y=alt.Y('Water Level:Q', scale=alt.Scale(domain=[y_min, y_max])),  # ปรับแกน Y
        color='Type:N'
    ).properties(
        title='Water Level Prediction for Next 3 Days',
        height=400
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

# การประมวลผลหลังจากอัปโหลดไฟล์
if uploaded_file is not None:
    # อ่านไฟล์ CSV ที่อัปโหลด
    df = pd.read_csv(uploaded_file)

    # รันการทำนายด้วยโมเดล LSTM
    st.markdown("---")
    st.write("ทำนายระดับน้ำ 3 วันข้างหน้าหลังจากข้อมูลล่าสุด")

    # รันการทำนายด้วยโมเดล LSTM
    df_predictions = predict_water_level_lstm(df, "lstm_2024_50epochs.keras")

    # พล๊อตผลลัพธ์การทำนายและข้อมูลจริง
    plot_results(df, df_predictions)

    # แสดงผลลัพธ์การทำนายเป็นตาราง
    st.subheader('ตารางข้อมูลที่ทำนาย')
    st.write(df_predictions)

else:
    st.write("กรุณาอัปโหลดไฟล์ CSV เพื่อเริ่มการทำนาย")

