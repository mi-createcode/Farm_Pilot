import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# ✅ ✅ ✅ 들여쓰기 없애야 함!
CONFIG = {
    'temp': {
        'model_dir': r"C:\Users\User\farmpilot\models",
        'model_path_template': 'lstm_model_{region}.h5',
        'scaler_path_template': 'scaler_{region}.pkl',
        'target_col': '평균기온(℃)',
        'feature_cols': ['평균기온(℃)', '평균 습도(%)', '강수량(mm)', 'season_sin', 'season_cos'],
        'seq_length': 90,
    },
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

MIN_PREDICTION_YEAR = 2025
DATA_PATH = r"C:\Users\User\farmpilot\SYNM_jeollado_koppen_discomfort.csv"

# 그리고 아래 class ~ def main() 까지도 마찬가지로 맨 왼쪽에 오도록 전체 들여쓰기 없애세요!


class WeatherPredictor:
    def __init__(self, model_type, region):
        if model_type not in CONFIG:
            raise ValueError(f"잘못된 모델 타입입니다: {model_type}.")
        self.model_type = model_type
        self.region = region
        self.config = CONFIG[model_type]
        self.model = None
        self.scalers = {}
        self._load_assets()

    def _load_pickle_with_fallback(self, path):
        # joblib 먼저 시도
        try:
            obj = joblib.load(path)
            return obj
        except Exception as e_joblib:
            # joblib 실패하면 pickle로 시도
            try:
                with open(path, 'rb') as f:
                    obj = pickle.load(f, encoding='latin1')
                return obj
            except Exception as e_pickle:
                raise RuntimeError(
                    f"scaler 로딩 실패\njoblib error: {e_joblib}\npickle error: {e_pickle}"
                )

    def _load_assets(self):
        model_dir = self.config['model_dir']
        model_path = os.path.join(model_dir, 
                                self.config['model_path_template'].format(region=self.region))
        try:
            self.model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})
        except Exception as e:
            st.error(f"모델 로딩 실패: {e}")
            raise e

        if 'scaler_path_template' in self.config:
            scaler_path = os.path.join(model_dir, 
                                    self.config['scaler_path_template'].format(region=self.region))
            try:
                self.scalers['main'] = self._load_pickle_with_fallback(scaler_path)
            except Exception as e:
                st.error(f"스케일러 로딩 실패: {e}")
                raise e
        else:
            features_scaler_path = os.path.join(model_dir, 
                                            self.config['scaler_features_path_template'].format(region=self.region))
            target_scaler_path = os.path.join(model_dir, 
                                            self.config['scaler_target_path_template'].format(region=self.region))
            try:
                self.scalers['features'] = self._load_pickle_with_fallback(features_scaler_path)
                self.scalers['target'] = self._load_pickle_with_fallback(target_scaler_path)
            except Exception as e:
                st.error(f"스케일러(특징/타겟) 로딩 실패: {e}")
                raise e

    def _prepare_historical_data(self, df):
        station_df = df[df['지역'] == self.region].copy()
        train_df = station_df[
            (station_df['날짜'] >= '2022-01-01') & 
            (station_df['날짜'] <= '2024-12-31')
        ].copy()
        train_df.set_index('날짜', inplace=True)
        numeric_cols = [col for col in self.config['feature_cols'] 
                    if 'sin' not in col and 'cos' not in col and 'weight' not in col]
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

        future_dates = pd.date_range(
            start=target_date - pd.Timedelta(days=15), 
            end=target_date + pd.Timedelta(days=15)
        )
        future_preds_scaled = []

        for future_date in future_dates:
            next_pred_scaled = self.model.predict(current_sequence, verbose=0)[0, 0]
            future_preds_scaled.append(next_pred_scaled)

            if self.model_type == 'temp':
                day_of_year = future_date.dayofyear
                month = future_date.month
                hist_humidity = historical_df[historical_df.index.month == month]['평균 습도(%)'].mean()
                hist_rainfall = historical_df[historical_df.index.month == month]['강수량(mm)'].mean()

                temp_features = np.array([[0,
                    historical_df['평균 습도(%)'].mean() if pd.isna(hist_humidity) else hist_humidity,
                    historical_df['강수량(mm)'].mean() if pd.isna(hist_rainfall) else hist_rainfall,
                    np.sin(2 * np.pi * day_of_year / 365.25),
                    np.cos(2 * np.pi * day_of_year / 365.25)
                ]])

                scaled_new_features = scaler_features.transform(temp_features)[0]
                scaled_new_features[0] = next_pred_scaled
                new_step = scaled_new_features.reshape(1, 1, len(feature_cols))

            else:
                day_of_year, month = future_date.dayofyear, future_date.month
                new_features = {
                    'season_sin': np.sin(2 * np.pi * day_of_year / 365.25),
                    'season_cos': np.cos(2 * np.pi * day_of_year / 365.25),
                    'summer_weight': 1.5 if month in [6, 7, 8] else 1.0
                }
                for col in ['평균기온(℃)', '최고기온(℃)', '평균 습도(%)', '강수량(mm)']:
                    val = historical_df.loc[historical_df['month'] == month, col].mean()
                    new_features[col] = historical_df[col].mean() if pd.isna(val) else val

                new_features_array = np.array([[new_features[col] for col in feature_cols]])
                new_step = scaler_features.transform(new_features_array).reshape(1, 1, len(feature_cols))

            current_sequence = np.append(current_sequence[:, 1:, :], new_step, axis=1)

        if self.model_type == 'discomfort':
            predictions_inv = self.scalers['target'].inverse_transform(
                np.array(future_preds_scaled).reshape(-1, 1)
            )
        else:
            preds_scaled = np.array(future_preds_scaled).reshape(-1, 1)
            dummy_features = np.zeros((len(preds_scaled), len(feature_cols) - 1))
            combined_preds = np.hstack([preds_scaled, dummy_features])
            predictions_inv = self.scalers['main'].inverse_transform(combined_preds)[:, 0]

        return future_dates, predictions_inv.flatten()

    def plot_results(self, dates, predictions, target_date):
        target_name = self.config['target_col']

        plt.figure(figsize=(12, 6))
        plt.plot(dates, predictions, 
                label=f'{target_date.year}년 예상 {target_name}', 
                color='red', marker='o', markersize=5)
        plt.axvline(target_date, color='blue', linestyle='--', linewidth=2, 
                label=f'선택 날짜: {target_date.strftime("%Y-%m-%d")}')
        plt.title(f"'{self.region}' 지역 {target_date.strftime('%Y년 %m월 %d일')} 주변 1개월 {target_name} 예측")
        plt.xlabel('날짜', fontsize=12)
        plt.ylabel(target_name, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()

        st.pyplot(plt.gcf())
        plt.close()

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')  # ← 여기 수정!
    df['날짜'] = pd.to_datetime(df['날짜'])
    return df

def main():
    st.title("🌤️ LSTM 기반 기상 예측 시스템")
    st.write("2025년 이후 전라도 지역 기온 및 불쾌지수 예측")

    df = load_data()
    regions = df['지역'].unique().tolist()

    region = st.selectbox("예측할 지역을 선택하세요", regions)

    target_date = st.date_input(
        "예측할 날짜를 선택하세요 (2025년 이후만 가능)", 
        min_value=pd.Timestamp(f'{MIN_PREDICTION_YEAR}-01-01'),
        max_value=pd.Timestamp('2100-12-31')
    )

    if target_date.year < MIN_PREDICTION_YEAR:
        st.error(f"{MIN_PREDICTION_YEAR}년 이후 날짜만 선택 가능합니다.")
        return

    if st.button("예측 실행"):
        with st.spinner("예측 중입니다... 잠시만 기다려주세요"):
            try:
                predictor_temp = WeatherPredictor('temp', region)
                hist_data_temp = predictor_temp._prepare_historical_data(df)
                dates_temp, preds_temp = predictor_temp.predict(hist_data_temp, pd.Timestamp(target_date))
                predictor_temp.plot_results(dates_temp, preds_temp, pd.Timestamp(target_date))

                predictor_dis = WeatherPredictor('discomfort', region)
                hist_data_dis = predictor_dis._prepare_historical_data(df)
                dates_dis, preds_dis = predictor_dis.predict(hist_data_dis, pd.Timestamp(target_date))
                predictor_dis.plot_results(dates_dis, preds_dis, pd.Timestamp(target_date))

                st.success("예측이 완료되었습니다!")

            except Exception as e:
                st.error(f"예측 중 오류가 발생했습니다: {e}")

if __name__ == '__main__':
    main()
