from mlmodel import predict_all_products
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import pickle
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import os

province_name_map = {
    '전북특별자치도': '전북',
    '전라남도': '전남',
}

def convert_province_city(province, city, town):
    province_short = province_name_map.get(province, province)
    return f"{province_short} {city} {town}"

CONFIG = {
    'temp': {
        'model_dir': r"C:\\Users\\User\\farmpilot\\models",
        'model_path_template': 'lstm_model_{region}.h5',
        'scaler_path_template': 'scaler_{region}.pkl',
        'target_col': '평균기온(℃)',
        'feature_cols': ['평균기온(℃)', '평균 습도(%)', '강수량(mm)', 'season_sin', 'season_cos'],
        'seq_length': 90,
    },
    'discomfort': {
        'model_dir': r"C:\\Users\\User\\farmpilot\\models2",
        'model_path_template': 'model_di_{region}.h5',
        'scaler_features_path_template': 'scaler_features_di_{region}.pkl',
        'scaler_target_path_template': 'scaler_target_di_{region}.pkl',
        'target_col': '불쾌지수',
        'feature_cols': ['평균기온(℃)', '최고기온(℃)', '평균 습도(%)', '강수량(mm)', 'season_sin', 'season_cos', 'summer_weight'],
        'seq_length': 60,
    }
}

DATA_PATH = r"C:\\Users\\User\\farmpilot\\SYNM_jeollado_koppen_discomfort.csv"
MIN_PREDICTION_YEAR = 2025

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
    st.title("맞춤형 정보 제공 ")

    if 'expected_date' not in st.session_state:
        st.warning("입력 페이지에서 정보를 먼저 입력해주세요.")
        return

    expected_date = st.session_state['expected_date']
    province = st.session_state['province']
    city = st.session_state['city']
    town = st.session_state['town']
    region = st.session_state['region']
    allergy_raw = st.session_state.get("allergy_info", "")
    allergy_list = [x.strip().lower() for x in allergy_raw.split(",") if x.strip()]
    labor_pref = st.session_state.get("labor_level", "중 (보통 수준)")

    st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)  # 탭과 위 제목 사이 띄우기


    tab_labels = ["🌾 작물 추천", "🌡️ 기상 예측", "📋 정책 정보", "🏠 빈집 정보", "📊 종합 보고서"]
    tab_styles = """
    <style>
        div[role="tablist"] > button {
            margin-right: 30px !important;  /* 탭 버튼 간격 조절 */
        }
    """

    tab_styles += "".join([f"button[data-baseweb='tab']:nth-child({i+1}) div {{font-size: 28px !important;}}" for i in range(len(tab_labels))])
    tab_styles += "</style>"

    st.markdown(tab_styles, unsafe_allow_html=True)
    st.markdown(tab_styles, unsafe_allow_html=True)
    tabs = st.tabs(tab_labels)
        
    with tabs[0]:
        sanji_nm = convert_province_city(province, city, town)
        results = predict_all_products(sanji_nm, expected_date.year, expected_date.month, expected_date.day)
        labels = [label.strip().lower() for label, _ in results]
        labels = [label for label in labels if not any(allergy in label for allergy in allergy_list)]

        df = pd.read_csv("crops.csv")
        df.columns = df.columns.str.strip()  # 공백 제거
        df['lower_name'] = df['Product Name'].str.lower()
        df['Value 3'] = pd.to_numeric(df['Value 3'], errors='coerce')

        level_map = {'하 (가벼운 작업)': 1, '중 (보통 수준)': 2, '상 (활동 많아도 괜찮아요)': 3}
        preferred_level = level_map.get(labor_pref, 2)

        valid = [label for label in labels if label in df['lower_name'].values]
        df = df[df['lower_name'].isin(valid)]

        if preferred_level == 1:
            df = df[df['Value 3'] == 1]
        elif preferred_level == 2:
            df = df[df['Value 3'].isin([1, 2])]

        df = df.sort_values(by='Value 1', ascending=False)
        top3 = df.head(3)

        st.subheader("✔️ 최종 추천 작물")

        font_sizes = ["28px", "28px", "28px"]  # 1위, 2위, 3위

        DESCRIPTION_COL = 'summary'
        IMAGE_LINK_COL = 'image'
        PERIOD_COL = 'period'
        YOUTUBE_LINK_COL = 'youtube'
        REASON_COL = 'reason'

        for i, row in top3.iterrows():
            font_size = font_sizes[top3.index.get_loc(i)]
            description = row[DESCRIPTION_COL]
            image_url = row[IMAGE_LINK_COL]
            period = row[PERIOD_COL]
            youtube_link = row[YOUTUBE_LINK_COL]
            reason = row[REASON_COL]

            st.markdown(f"""
                <div style='background-color:#E0F2F7; padding:20px; border-radius:10px; margin-bottom:20px; display: flex; align-items: flex-start;'>
                    <div style='flex: 1; max-width: 25%; margin-right: 20px;'>
                        <img src="{image_url}" alt="{row['Product Name']}" style="width: 100%; height: auto; border-radius: 8px; object-fit: contain;">
                    </div>
                    <div style='flex: 3;'>
                        <h3 style="color:#005f99; font-size:{font_size}; margin-bottom:10px; font-weight:bold;">
                            {row['Product Name']} <span style="font-size:16px; color:#555;">(노동강도: {row['Value 2 (Korean)']})</span>
                        </h3>
                        <p style="font-size:25px; color:#333; line-height:1.8; margin-bottom:20px;">
                            {description}
                        </p>
                        <ul style="font-size:20px; color:#444; line-height:1.8; padding-left: 24px; margin-top: 0;">
                        <strong>- {town}에서 키우기 좋은 이유:</strong> {reason}
                        </li>
                        <li style="margin-bottom:8px;">
                            <strong>- 농사 기간:</strong> {period}
                        </li>
                        <li>
                        <strong>- 기초교육 링크:</strong> <a href="{youtube_link}" target="_blank" style="color:#005f99; text-decoration:none; font-weight:600;">유튜브 바로가기 ▶️</a></li>
                        </ul>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    with tabs[1]:
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
            start = target_date - pd.Timedelta(days=15)
            end = target_date + pd.Timedelta(days=15)
        

            valid_indices = (dates_temp >= start) & (dates_temp <= end)
            dates_range = dates_temp[valid_indices]
            preds_range = preds_temp[valid_indices]

            
            fig1 = plt.figure(figsize=(12,6))
            col1, col2 = st.columns(2)
            plt.plot(dates_range, preds_range, label='예측 온도', color='red', marker='o')
            plt.axvline(target_date, color='blue', linestyle='--', label='선택 날짜')


            plt.title(f"{region} 온도 예측 (±15일)")
            plt.xlabel('날짜')
            plt.ylabel('온도 (℃)')
            plt.grid(True)
            plt.legend()
            plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig1)
            with col1:
                fig1 = plt.figure(figsize=(6, 4))
                plt.plot(dates_temp, preds_temp, label='predictred temperature (℃)', color='red', marker='o')
                plt.axvline(target_date, color='blue', linestyle='--', label='chosen date')
                target_temp = preds_temp[dates_temp == target_date]
                if target_temp.size > 0:
                    plt.plot(target_date, target_temp[0], 'o', color='blue', markersize=10)
                    plt.text(target_date, target_temp[0] + 0.5, f"{target_temp[0]:.1f}℃", 
                            color='blue', fontsize=12, fontweight='bold', ha='center')
                plt.title(f"{region} Temperature Prediction")
                plt.xlabel('Date')
                plt.ylabel('Temperature (℃)')
                plt.grid(True)
                plt.legend()
                plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig1)

            with col2:
                fig2 = plt.figure(figsize=(6, 4))
                plt.plot(dates_discomfort, preds_discomfort, label='predicted index', color='orange', marker='o')
                plt.axvline(target_date, color='blue', linestyle='--', label='chosen date')
                target_index = preds_discomfort[dates_discomfort == target_date]
                if target_index.size > 0:
                    plt.plot(target_date, target_index[0], 'o', color='blue', markersize=10)
                    plt.text(target_date, target_index[0] + 0.5, f"{target_index[0]:.1f}", 
                            color='blue', fontsize=12, fontweight='bold', ha='center')
                plt.title(f"{region} Discomfort Index Prediction")
                plt.xlabel('Date')
                plt.ylabel('Discomfort Index')
                plt.grid(True)
                plt.legend()
                plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig2)

        except Exception as e:
            st.error(f"예측 중 오류 발생: {e}")

            # try-except 블록 끝난 뒤에는 들여쓰기 없이 작성
st.markdown("---")
if st.button("입력 페이지로 돌아가기"):
    for key in ['expected_date', 'region']:
        if key in st.session_state:
            del st.session_state[key]
    st.experimental_rerun()

if __name__ == '__main__':
    crop_weather_page()