import streamlit as st
import numpy as np
import joblib

def run_ml_app():
    st.subheader('자동차 금액 예측')
    
    # 성별, 나이, 연봉, 카드빚, 자산을 유저한테 모두 입력받아서
    # 자동차 구매 금액 예측하세요.

    new_data = np.array([0, 50, 40000, 50000, 200000])

    new_data = new_data.reshape(1, 5)

    regressor = joblib.load('regressor.pkl')

    regressor.predict(new_data)

