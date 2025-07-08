from ml_test import predict_top3_product
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import pickle
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import os

# 공통으로 쓰는 province_name_map 미리 뺌
province_name_map = {
    '전북특별자치도': '전북',
    '전라남도': '전남',
}

def convert_province_city(province, city, town):
    province_short = province_name_map.get(province, province)  # 기본값은 province
    return f"{province_short} {city} {town}"

# --- 1. CONFIG 설정 ---
CONFIG = {
    # 기온 예측 모델 설정
    'temp': {
        'model_dir': r"C:\Users\User\farmpilot\models",
        'model_path_template': 'lstm_model_{region}.h5',
        'scaler_path_template': 'scaler_{region}.pkl',
        'target_col': '평균기온(℃)',
        'feature_cols': ['평균기온(℃)', '평균 습도(%)', '강수량(mm)', 'season_sin', 'season_cos'],
        'seq_length': 90,
    },
    # 불쾌지수 예측 모델 설정
    'discomfort': {
        'model_dir': r"C:\Users\User\farmpilot\models2",
        'model_path_template': 'model_di_{region}.h5',
        'scaler_features_path_template': 'scaler_features_di_{region}.pkl',
        'scaler_target_path_template': 'scaler_target_di_{region}.pkl',
        'target_col': '불쾌지수',
        'feature_cols': ['평균기온(℃)', '최고기온(℃)', '평균 습도(%)', '강수량(mm)', 
                         'season_sin', 'season_cos', 'summer_weight'],
        'seq_length': 60,
    }
}

DATA_PATH = r"C:\Users\User\farmpilot\SYNM_jeollado_koppen_discomfort.csv"
MIN_PREDICTION_YEAR = 2025

# --- 2. WeatherPredictor 클래스 정의 ---
class WeatherPredictor:
    """
    기상 데이터 예측을 위한 통합 클래스 (2025년 이후 지원)
    """
    def __init__(self, model_type, region):
        if model_type not in CONFIG:
            raise ValueError(f"잘못된 모델 타입입니다: {model_type}. 가능한 타입: {list(CONFIG.keys())}")
        self.model_type = model_type
        self.region = region
        self.config = CONFIG[model_type]
        self.model = None
        self.scalers = {}
        self._load_assets()

    def _load_assets(self):
        model_dir = self.config['model_dir']
        model_path = os.path.join(model_dir, self.config['model_path_template'].format(region=self.region))
        try:
            self.model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})
        except Exception as e:
            raise RuntimeError(f"모델 로딩 실패: {model_path}\n{e}")

        if 'scaler_path_template' in self.config:
            scaler_path = os.path.join(model_dir, self.config['scaler_path_template'].format(region=self.region))
            with open(scaler_path, 'rb') as f:
                self.scalers['main'] = pickle.load(f)
        elif 'scaler_features_path_template' in self.config:
            features_scaler_path = os.path.join(model_dir, self.config['scaler_features_path_template'].format(region=self.region))
            with open(features_scaler_path, 'rb') as f:
                self.scalers['features'] = pickle.load(f)
            target_scaler_path = os.path.join(model_dir, self.config['scaler_target_path_template'].format(region=self.region))
            with open(target_scaler_path, 'rb') as f:
                self.scalers['target'] = pickle.load(f)

    def _prepare_historical_data(self, df):
        station_df = df[df['지역'] == self.region].copy()
        train_df = station_df[(station_df['날짜'] >= '2022-01-01') & (station_df['날짜'] <= '2024-12-31')].copy()
        train_df.set_index('날짜', inplace=True)
        numeric_cols = [col for col in self.config['feature_cols'] if 'sin' not in col and 'cos' not in col and 'weight' not in col]
        if self.model_type == 'discomfort':
            numeric_cols.insert(0, self.config['target_col'])
        for col in numeric_cols:
            train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
        train_df = train_df[list(set(numeric_cols))].resample('D').mean()
        train_df['day_of_year'] = train_df.index.dayofyear
        train_df['month'] = train_df.index.month
        train_df['season_sin'] = np.sin(2 * np.pi * train_df['day_of_year'] / 365.25)
        train_df['season_cos'] = np.cos(2 * np.pi * train_df['day_of_year'] / 365.25)
        if 'summer_weight' in self.config['feature_cols']:
            train_df['summer_weight'] = np.where(train_df['month'].isin([6, 7, 8]), 1.5, 1.0)
        return train_df.interpolate().fillna(method='ffill').fillna(method='bfill').fillna(0)

    def predict(self, historical_df, target_date):
        feature_cols = self.config['feature_cols']
        seq_length = self.config['seq_length']
        scaler_features = self.scalers.get('features', self.scalers.get('main'))

        scaled_data = scaler_features.transform(historical_df[feature_cols])
        current_sequence = scaled_data[-seq_length:].reshape(1, seq_length, len(feature_cols))

        future_dates = pd.date_range('2025-01-01', '2025-12-31')
        future_preds_scaled = []

        for future_date in future_dates:
            next_pred_scaled = self.model.predict(current_sequence, verbose=0)[0, 0]
            future_preds_scaled.append(next_pred_scaled)

            if self.model_type == 'temp':
                day_of_year = future_date.dayofyear
                month = future_date.month
                hist_humidity = historical_df[historical_df.index.month == month]['평균 습도(%)'].mean()
                hist_rainfall = historical_df[historical_df.index.month == month]['강수량(mm)'].mean()

                temp_features = np.array([[
                    0,
                    historical_df['평균 습도(%)'].mean() if pd.isna(hist_humidity) else hist_humidity,
                    historical_df['강수량(mm)'].mean() if pd.isna(hist_rainfall) else hist_rainfall,
                    np.sin(2 * np.pi * day_of_year / 365.25),
                    np.cos(2 * np.pi * day_of_year / 365.25)
                ]])
                scaled_new_features = scaler_features.transform(temp_features)[0]
                scaled_new_features[0] = next_pred_scaled
                new_step = scaled_new_features.reshape(1, 1, len(feature_cols))
            else:  # discomfort
                day_of_year, month = future_date.dayofyear, future_date.month
                new_features = {
                    'season_sin': np.sin(2 * np.pi * day_of_year / 365.25),
                    'season_cos': np.cos(2 * np.pi * day_of_year / 365.25),
                    'summer_weight': 1.5 if month in [6,7,8] else 1.0
                }
                for col in ['평균기온(℃)', '최고기온(℃)', '평균 습도(%)', '강수량(mm)']:
                    val = historical_df.loc[historical_df['month'] == month, col].mean()
                    new_features[col] = historical_df[col].mean() if pd.isna(val) else val
                new_features_array = np.array([[new_features[col] for col in feature_cols]])
                new_step = scaler_features.transform(new_features_array).reshape(1, 1, len(feature_cols))

            current_sequence = np.append(current_sequence[:, 1:, :], new_step, axis=1)

        if self.model_type == 'discomfort':
            predictions_inv = self.scalers['target'].inverse_transform(np.array(future_preds_scaled).reshape(-1, 1))
        else:
            preds_scaled = np.array(future_preds_scaled).reshape(-1, 1)
            dummy_features = np.zeros((len(preds_scaled), len(feature_cols) - 1))
            combined_preds = np.hstack([preds_scaled, dummy_features])
            predictions_inv = self.scalers['main'].inverse_transform(combined_preds)[:, 0]

        return future_dates, predictions_inv.flatten()


def crop_weather_page():
    st.title("🌾 작물 + 기상 예측 결과")

    # 공통 세션값 꺼내기
    if 'expected_date' not in st.session_state or 'province' not in st.session_state or \
       'region' not in st.session_state or 'city' not in st.session_state or 'town' not in st.session_state:
        st.warning("입력 페이지에서 값을 먼저 입력해주세요.")
        return

    # === 1. 작물 예측 ===
    st.subheader("✅ 추천 작물 예측")

    expected_date = st.session_state['expected_date']
    year, month, day = expected_date.year, expected_date.month, expected_date.day

    province = st.session_state['province']
    city = st.session_state['city']
    town = st.session_state['town']
    region = st.session_state['region']

    province_short = province_name_map.get(province, province)
    sanji_nm = f"{province_short} {city} {town}"

    try:
        top3_results = predict_top3_product(sanji_nm, year, month, day)
        for i, (label, score) in enumerate(top3_results, 1):
            st.markdown(f"### {i}. **{label}**")
    except Exception as e:
        st.error(f"작물 예측 중 오류 발생: {e}")

    st.markdown("---")

    # === 2. 기상 예측 ===
    st.subheader("📊 기상 및 불쾌지수 예측")
     
    expected_date_str = st.session_state.get('expected_date', None)
    region = st.session_state.get('region', None)

    if expected_date_str is None or region is None:
        st.warning("먼저 입력 페이지에서 날짜와 지역을 선택해 주세요.")
        return  # 여기서 리턴해서 예측 안 하게 막음

    try:
        target_date = pd.to_datetime(expected_date_str)
    except Exception as e:
        st.error(f"날짜 형식 오류: {e}")
        return

    DATA_PATH = r"C:\Users\User\farmpilot\SYNM_jeollado_koppen_discomfort.csv"
    MIN_PREDICTION_YEAR = 2025

    try:
        df = pd.read_csv(DATA_PATH)
        df['날짜'] = pd.to_datetime(df['날짜'])
    except Exception as e:
        st.error(f"데이터 로드 오류: {e}")
        return

    if target_date.year < MIN_PREDICTION_YEAR:
        st.error(f"{MIN_PREDICTION_YEAR}년 이후의 날짜만 예측 가능합니다.")
        return

    dates_temp, preds_temp = [], []
    dates_discomfort, preds_discomfort = [], []

    try:
        predictor_temp = WeatherPredictor('temp', region)
        hist_data_temp = predictor_temp._prepare_historical_data(df)
        dates_temp, preds_temp = predictor_temp.predict(hist_data_temp, target_date)

        predictor_discomfort = WeatherPredictor('discomfort', region)
        hist_data_discomfort = predictor_discomfort._prepare_historical_data(df)
        dates_discomfort, preds_discomfort = predictor_discomfort.predict(hist_data_discomfort, target_date)

        st.subheader(f"지역: {region} / 예측 날짜: {target_date.strftime('%Y-%m-%d')}")

        # 그래프 출력 코드도 여기에 넣기

    
    except Exception as e:
        st.error(f"예측 중 오류가 발생했습니다: {e}")
        return  # 에러 시 더 진행 안 함

    # 예측 성공 시 그래프 출력
    fig1 = plt.figure(figsize=(12,6))
    
    start,end = target_date - pd.Timedelta(days=15), target_date + pd.Timedelta(days=15)
    plt.xlim(start,end)

    valid_indices = (dates_temp >= start) & (dates_temp <= end)
    predictions_in_xlim = preds_temp[valid_indices]
        # 기온 그래프
   

    if len(predictions_in_xlim) > 0:
        min_val_in_xlim = np.min(predictions_in_xlim)
        max_val_in_xlim = np.max(predictions_in_xlim)
        y_range = max_val_in_xlim - min_val_in_xlim
        padding = y_range * 0.1 if y_range > 0 else 5
        plt.ylim(min_val_in_xlim - padding, max_val_in_xlim + padding)

    plt.plot(dates_temp, preds_temp, label='predictred temperature (℃)', color='red', marker='o')
    plt.axvline(target_date, color='blue', linestyle='--', label='choosen date')
    plt.title(f"{region} temperature prediction")
    plt.xlabel('Date')
    plt.ylabel('Temperature (℃)')
    plt.grid(True)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig1)

    # 불쾌지수 그래프
    fig2 = plt.figure(figsize=(12,6))

    dates_discomfort = pd.to_datetime(dates_discomfort)
    start = pd.to_datetime(target_date - pd.Timedelta(days=15))
    end = pd.to_datetime(target_date + pd.Timedelta(days=15))

    # x축 범위 설정
    plt.xlim(start, end)

    # y축 범위 자동 조정
    valid_indices_di = (dates_discomfort >= start) & (dates_discomfort <= end)
    predictions_in_xlim_di = preds_discomfort[valid_indices_di]

    if len(predictions_in_xlim_di) > 0:
        min_val = np.min(predictions_in_xlim_di)
        max_val = np.max(predictions_in_xlim_di)
        y_range = max_val - min_val
        padding = y_range * 0.1 if y_range > 0 else 5
        plt.ylim(min_val - padding, max_val + padding)

    plt.plot(dates_discomfort, preds_discomfort, label='predicted index', color='orange', marker='o')
    plt.axvline(target_date, color='blue', linestyle='--', label='choosen date')
    plt.title(f"{region} discomfort index prediction")
    plt.xlabel('Date')
    plt.ylabel('discomfort index')
    plt.grid(True)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig2)

    if st.button("입력 페이지로 돌아가기"):
        for key in ['expected_date', 'region']:
            if key in st.session_state:
                del st.session_state[key]
        st.experimental_rerun()


if __name__ == "__main__":
    crop_weather_page()
