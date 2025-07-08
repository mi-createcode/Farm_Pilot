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
import joblib
from policy import get_policy_text
from emptyhouse import get_house_text
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

# 한글 폰트 설정 (Windows 기준, Mac/Linux면 경로 다름)
font_path = "C:/Windows/Fonts/malgun.ttf"  # 맑은 고딕 예시
fontprop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = fontprop.get_name()



# 모델과 전처리기 로딩
load_dir = "mlmodel"
model = joblib.load(os.path.join(load_dir, 'rf_model.pkl'))
scaler = joblib.load(os.path.join(load_dir, 'rf_scaler.pkl'))
sanji_encoder = joblib.load(os.path.join(load_dir, 'rf_sanji_encoder.pkl'))
season_encoder = joblib.load(os.path.join(load_dir, 'rf_season_encoder.pkl'))
prdlst_encoder = joblib.load(os.path.join(load_dir, 'rf_prdlst_encoder.pkl'))

labor_intensity_dict = {
    '가랏': '중', '가죽나물': '중', '가지': '중', '가지나무순': '중', '갈치': '중', '감귤': '하', '감자': '하',
    '갓': '하', '강낭콩': '하', '강황': '중', '건고추': '상', '겨자': '하', '겨자잎': '중', '겨자채': '중',
    '고구마': '하', '고구마순': '중', '고들빼기': '중', '고등어': '하', '고비': '중', '고사리': '중', '고수': '중',
    '고추잎': '중', '곤달비': '중', '곤드레나물': '중', '과루인(하늘타리)': '중', '구기자': '중', '그레이프푸룻(자몽)': '하',
    '그린빈스': '중', '근대': '중', '금감': '하', '금강초': '중', '깻잎': '상', '꽈리고추': '상',
    '냉이': '중', '넙치': '중', '노루궁뎅이버섯': '상', '농어': '중', '느타리버섯': '상', '다래': '중', '단감': '하',
    '달래': '중', '당근': '하', '대구': '하', '대추': '중', '대파': '하', '더덕': '중', '도라지': '중', '도루묵': '하',
    '도토리': '상', '돌나물': '중', '돔': '중', '돗나물': '중', '동부': '하', '동초': '하', '동충하초': '상',
    '두릅': '중', '들깨': '하', '딸기': '상', '땅콩': '하', '떫은감': '하', '레드쉬': '하', '레몬': '하',
    '로메인': '중', '루꼴라': '중', '마': '중', '마늘': '하', '만가닥': '상', '만감': '하', '매실': '중', '머위대': '중',
    '메론': '중', '메밀': '하', '메밀순': '중', '명아주': '중', '모과': '중', '모시대': '중', '모시잎': '중', '목이': '상',
    '무': '하', '무순': '중', '무청': '중', '무화과': '중', '미나리': '중', '민들레': '중',
    '박': '하', '밤': '중', '방아': '중', '방울양배추': '중', '방울토마토': '상', '방풍': '중', '방풍나물': '중',
    '배': '중', '배추': '하', '버찌': '중', '보리': '하', '보리수': '중', '보리순': '중', '복분자': '중', '복숭아': '중',
    '봄동배추': '하', '부지깽이': '중', '부추': '중', '브로코리(녹색꽃양배추)': '중', '블루베리': '중',
    '비름': '중', '비타민': '중', '비트(붉은사탕무우)': '하', '비파': '중', '빈스': '중','사과': '중',
    '산마늘': '중', '산양삼': '상', '살구': '중', '삼엽채': '중', '삼채': '중', '상추': '중', '상황버섯': '상',
    '새송이': '상', '새싹': '중', '생강': '중', '석류': '중', '셀러리(양미나리)': '중', '솔잎': '상',
    '수박': '하', '수삼': '상', '수세미': '하', '수수': '하', '숙주나물': '중', '순무': '하', '시금치': '중',
    '실파': '중', '쌈채': '중', '쌈추': '중', '쑥': '중', '쑥갓': '중', '씀바귀': '중',
    '아로니아': '중', '아스파라가스': '중', '아욱': '중', '알로애': '중', '알타리무': '하', '애느타리버섯': '상',
    '앵두': '중', '야콘': '하', '양배추': '하', '양상추': '중', '양송이': '상', '양파': '하', '양하': '중',
    '어린잎': '중', '얼갈이배추': '하', '엉게나물': '중', '여주': '하', '연근': '중', '열무': '하', '영지버섯': '상',
    '오가피': '중', '오디': '중', '오렌지': '하', '오미자': '중', '오이': '상', '옥수수': '하', '옻': '중',
    '완두': '하', '우슬': '중', '우엉': '하', '우엉대': '중', '울금': '중', '원추리': '중', '유자': '중', '유채': '하',
    '은행': '상', '음나무순': '중', '익모초': '중', '인삼': '상', '잎새': '중', '자두': '중', '자몽': '하',
    '자연산송이': '상', '질경이': '중', '쪽파': '중', '참깨': '하', '참나물': '중', '참다래': '중', '참다래(키위)': '중',
    '참당귀': '중', '참외': '중', '참죽나무순': '중', '천마': '상', '청경채': '중', '체리': '중', '초석잠': '중', '초피': '중',
    '춘채': '중', '취나물': '중', '치커리': '중', '칼리플라워(꽃양배추)': '중', '케일': '중', '콜라비(순무양배추)': '하',
    '콩': '하', '콩나물': '중', '탄제린': '하', '탱자': '중', '토란': '하', '토란대': '중', '토마토': '상',
    '파세리(향미나리)': '중', '파프리카': '상', '팥': '하', '팽이버섯': '상', '포도': '중', '표고버섯': '상',
    '풋고추': '상', '플럼코트': '중', '피마자잎': '중', '피망(단고추)': '상', '피망잎': '중', '호두': '중', '호박': '하',
    '호박잎': '중', '홍고추': '상',
}
province_name_map = {
    '전북특별자치도': '전북',
    '전라남도': '전남',
}

# 함수: 도 이름 치환 + city 결합
def convert_province_city(province, city, town):
    province_short = province_name_map.get(province, province)  # 기본값은 province
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


def crop_page():
    # 입력값 세션에서 꺼내기
    if 'expected_date' not in st.session_state or 'province' not in st.session_state:
        st.warning("입력 페이지에서 값을 먼저 입력해주세요.")
        return

    # 날짜 정보
    expected_date = st.session_state['expected_date']
    year = expected_date.year
    month = expected_date.month
    day = expected_date.day

# 딕셔너리: 도 이름 줄이기
   


    # 사용 예시 (페이지 코드에서)
    province = st.session_state['province']
    city = st.session_state['city']
    town = st.session_state['town']
    sanji_nm = convert_province_city(province, city, town)


level_map = {'하 (가벼운 작업)': 1, '중 (보통 수준)': 2, '상 (활동 많아도 괜찮아요)': 3}

# 노동 난이도 반영 추천 리스트 재배치 함수
def reorder_all_by_labor_preference(all_results, preference_level):
    preferred_level_num = level_map.get(preference_level, 2)

    # 딕셔너리 키 불일치 해결용 매핑
    labor_level_label_map = {
        '하': '하 (가벼운 작업)',
        '중': '중 (보통 수준)',
        '상': '상 (활동 많아도 괜찮아요)',
    }

    def preference_sort_key(item):
        crop, prob = item
        labor_level_raw = labor_intensity_dict.get(crop, '중')
        labor_level = labor_level_label_map.get(labor_level_raw, '중 (보통 수준)')
        level_num = level_map[labor_level]
        diff = abs(level_num - preferred_level_num)
        return (diff, -prob)

    return sorted(all_results, key=preference_sort_key)

# 알러지 필터 함수
def filter_allergy_items(results, allergy_list):
    return [(crop, prob) for (crop, prob) in results if crop not in allergy_list]

# 농작물 추천을 위한 메인 함수
def predict_all_products(sanji_nm, year, month, day, preference_level='중 (보통 수준)', allergy_list=None):    
    season = {
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'fall', 10: 'fall', 11: 'fall'
    }[month]

    sanji_enc = sanji_encoder.transform([sanji_nm])[0]
    season_enc = season_encoder.transform([season])[0]
    input_df = pd.DataFrame([{
        'SANJI_ENC': sanji_enc,
        'YEAR': year,
        'MONTH': month,
        'DAY': day,
        'SEASON_ENC': season_enc
    }])
    input_scaled = scaler.transform(input_df)

    prob = model.predict_proba(input_scaled)[0]
    all_labels = prdlst_encoder.inverse_transform(np.arange(len(prob)))
    all_results = list(zip(all_labels, prob))

    if allergy_list:
        all_results = filter_allergy_items(all_results, allergy_list)

    sorted_by_prob = sorted(all_results, key=lambda x: x[1], reverse=True)
    top3_labels = [label for label, _ in sorted_by_prob[:3]]

    '''
    print("Top-3 예측 작물 (노동 선호 반영 전):")
    for i, label in enumerate(top3_labels, 1):
        print(f"{i}. {label}")
    '''

    sorted_results = reorder_all_by_labor_preference(all_results, preference_level)

    '''
    print(f"\n입력값: 지역={sanji_nm}, 날짜={year}-{month:02}-{day:02}, 계절={season}, 선호 노동강도={preference_level}")
    print("전체 작물 추천 순위 (노동 선호 반영 후)")
    for i, (label, score) in enumerate(sorted_results, 1):
        labor = labor_intensity_dict.get(label, '중 (보통 수준)')
        print(f"{i}. {label} (노동강도: {labor})")
        if i == 10: break
    '''

    return sorted_results

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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_labels)
    tab_styles = """
    <style>
        div[role="tablist"] > button {
            margin-right: 30px !important;  /* 탭 버튼 간격 조절 */
        }
    """

    tab_styles += "".join([f"button[data-baseweb='tab']:nth-child({i+1}) div {{font-size: 28px !important;}}" for i in range(len(tab_labels))])
    tab_styles += "</style>"

    st.markdown(tab_styles, unsafe_allow_html=True)
    
        
    with tab1:
        sanji_nm = convert_province_city(province, city, town)
        results = predict_all_products(
            sanji_nm,
            expected_date.year, expected_date.month, expected_date.day,
            preference_level=labor_pref,
            allergy_list=allergy_list
        )

        labels = [label for label, _ in results]
        df = pd.read_csv("crops.csv")
        df.columns = df.columns.str.strip()
        df['lower_name'] = df['Product Name'].str.lower()
        df = df[df['lower_name'].isin([l.lower() for l in labels])]

        # 모델 결과 순서대로 정렬
        label_order = [l.lower() for l in labels]
        df['sort_order'] = df['lower_name'].apply(lambda x: label_order.index(x))
        df = df.sort_values('sort_order')
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

    
    

    with tab2:
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
        

            valid_temp = (dates_temp >= start) & (dates_temp <= end)
            dates_temp_range = dates_temp[valid_temp]
            preds_temp_range = preds_temp[valid_temp]

            # 불쾌지수 범위 필터 (이것도 같이)
            valid_discomfort = (dates_discomfort >= start) & (dates_discomfort <= end)
            dates_discomfort_range = dates_discomfort[valid_discomfort]
            preds_discomfort_range = preds_discomfort[valid_discomfort]

            
          
            col1, col2 = st.columns(2)

            with col1:
                fig1 = plt.figure(figsize=(6, 4))
                plt.plot(dates_temp_range, preds_temp_range, label='예측 온도 (℃)', color='lightgreen', marker='o')
                plt.axvline(target_date, color='skyblue', linestyle='--', label='선택 날짜')

                target_temp = preds_temp[dates_temp == target_date]
                if target_temp.size > 0:
                    plt.plot(target_date, target_temp[0], 'o', color='skyblue', markersize=10)
                    plt.text(target_date, target_temp[0] + 1, f"{target_temp[0]:.1f}℃", 
                            color='navy', fontsize=12, fontweight='bold', ha='left', va='center')

                plt.title(f"{region} 온도 예측 (±7일)")
                plt.xlabel('날짜')
                plt.ylabel('온도 (℃)')

                plt.legend()

                # x축 범위 설정
                start_date = target_date - pd.Timedelta(days=7)
                end_date = target_date + pd.Timedelta(days=7)
                plt.xlim(start_date, end_date)

                # x축 날짜 포맷, 눈금선 없애기
                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=3))
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                plt.tick_params(axis='x', which='both', length=0)

                plt.grid(False)  # 격자선 없애기

                plt.xticks(rotation=45)
                plt.tight_layout()

                st.pyplot(fig1)


            with col2:
                fig2 = plt.figure(figsize=(6, 4))
                plt.plot(dates_discomfort_range, preds_discomfort_range, label='불쾌지수 예측', color='orange', marker='o')
                plt.axvline(target_date, color='skyblue', linestyle='--', label='선택 날짜')

                target_index = preds_discomfort[dates_discomfort == target_date]
                if target_index.size > 0:
                    plt.plot(target_date, target_index[0], 'o', color='skyblue', markersize=10)
                    # x 좌표를 약간 오른쪽으로 이동 (+0.5일 정도)
                    plt.text(target_date + pd.Timedelta(days=0.5), target_index[0], f"{target_index[0]:.1f}", 
                            color='navy', fontsize=12, fontweight='bold', ha='left', va='center')

                plt.title(f"{region} 불쾌지수 예측 (±15일)")
                plt.xlabel('날짜')
                plt.ylabel('불쾌지수')

                plt.grid(False)  # 격자선 끄기
                plt.legend()

                start_date = target_date - pd.Timedelta(days=7)
                end_date = target_date + pd.Timedelta(days=7)
                plt.xlim(start_date, end_date)

                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=3))
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(1)) 

                plt.xticks(rotation=45)
                plt.tight_layout()

                st.pyplot(fig2)

            st.markdown(
                "<div style='font-size:26px; color:#444; background-color:#f0f9f5; padding:1rem 1.5rem; border-left:5px solid #4CAF50; border-radius:8px;'>"
                "🌤️ 25.5도에 불쾌지수 82.8은 따뜻하고 습한 날씨로, 여름의 활기를 느낄 수 있는 날씨입니다."
                "</div>",
                unsafe_allow_html=True
            )



        except Exception as e:
                st.error(f"예측 중 오류 발생: {e}") 
    with tab3:
        get_house_text()
    with tab4:        
        get_policy_text()
    with tab5:
        st.markdown("""
            <style>
                .title {
                    font-size: 40px;
                    font-weight: 700;
                    margin-bottom: 20px;
                    color: #2F4F4F;
                }
                .section-title {
                    font-size: 32px;
                    font-weight: 600;
                    margin-top: 30px;
                    margin-bottom: 10px;
                    color: #3A7D44;
                }
                .text, .list-item, .note {
                    font-size: 24px;
                    line-height: 1.5;
                    color: #333333;
                }
                .highlight {
                    color: #4CAF50;
                    font-weight: 600;
                }
                .box {
                    background-color: #f8f9fa;
                    border-left: 5px solid #4CAF50;
                    border-radius: 10px;
                    padding: 20px 25px;
                    margin-bottom: 30px;
                    box-shadow: 1px 1px 6px rgba(0,0,0,0.07);
                }
                a {
                    font-size: 24px;
                    color: #3A7D44;
                    font-weight: 600;
                    text-decoration: none;
                }
                a:hover {
                    text-decoration: underline;
                }
            </style>

            <div class="title">🌱 당신만을 위한 귀농 맞춤 가이드</div>

            <div class="section-title">✔️ 최종 추천 작물: <span class="highlight">가지</span></div>
            <div class="text">
                가지는 여름철을 대표하는 채소로, 부드럽고 수분이 풍부해 다양한 요리에 활용하기 아주 좋아요. 노동강도는 <strong>중간</strong> 정도라 무리 없이 관리할 수 있습니다.
            </div>
            <ul class="list-item">
                <li><strong>왜 순창읍에서 가지가 좋을까요?</strong> 순창읍은 일조량이 풍부하고 여름철 고온을 견디는 데 최적화된 환경이에요. 그래서 고품질 가지를 안정적으로 생산할 수 있답니다.</li>
                <li><strong>농사 기간은 어떻게 될까요?</strong> 가지는 보통 4~5월에 정식을 하고, 7~9월 사이에 수확할 수 있어요. 계절 변화에 잘 맞아, 수확 시기도 무리 없답니다.</li>
                <li><strong>기초 교육도 준비되어 있어요!</strong> 농사 초보자라도 걱정 마세요. 가지 재배를 쉽게 배울 수 있는 기초 교육 영상도 유튜브에서 바로 확인할 수 있습니다. ▶️</li>
            </ul>
            <div class="weather-box">
                🌤️ 25.5도에 불쾌지수 82.8은 따뜻하고 습한 날씨로, 여름의 활기를 느낄 수 있는 날씨입니다.
            </div>
            <div class="section-title">🏘️ 추천 매물 정보 (순창군 순창읍)</div>
            <div class="box">
                <p class="text"><strong>용지 종류:</strong> 대지</p>
                <p class="text"><strong>공부 지목:</strong> 건물</p>
                <p class="text"><strong>실 지목:</strong> 목조</p>
                <p class="text"><strong>판매 구분:</strong> 매매</p>
                <p class="text"><strong>가격:</strong> 9천원</p>
                <p class="text"><strong>면적:</strong> 235㎡ (건축 면적 110㎡)</p>
                <p class="text"><strong>주소:</strong> 순창군 순창읍 순창로 220-9</p>
                <p class="text">판매자 정보는 제공되지 않았으나, 담당자는 순창군 인구정책과 귀농귀촌팀이며 연락처는 063-650-1594입니다.</p>
                <p class="text"><strong>등록일:</strong> 2021년</p>
                <p class="text">특이사항은 별도로 없습니다.</p>
            </div>

            <div class="section-title">📍 순창군 순창읍 귀농 정책 및 보조금 안내</div>
            <div class="text">
                순창군은 귀농·귀촌인을 위한 다양한 지원 정책을 운영하고 있어요.<br><br>
                <strong>귀농·귀촌 지원 사업:</strong> 안정적인 정착을 돕기 위한 교육, 상담, 정보 제공이 활발히 이루어지고 있습니다. 예비 귀농인들은 농업 체험과 현장 실습 기회를 가질 수 있어요.<br><br>
                <strong>멘토링 컨설팅 지원:</strong> 성공한 농업인들의 경험을 공유하며 신규 귀농인들이 정착할 수 있도록 돕는 멘토링 프로그램이 운영됩니다.<br><br>
                <strong>재정적 지원:</strong> 초기 정착 비용인 이사, 환영회, 주택 자금 일부 보조까지 받을 수 있어 경제적 부담을 줄일 수 있어요.<br><br>
                <strong>현장 실습 및 창업 교육:</strong> 직접 체험할 수 있는 실습교육과 창업 지원도 제공되며 차량 임차비나 숙식비 일부도 지원합니다.<br><br>
                <strong>특산물 활성화:</strong> 순창군은 두릅, 블루베리 등 특산물을 중심으로 농업을 활성화시키려 노력하고 있습니다.
            </div>

            <div class="text" style="margin-top:40px;">
                혹시 추가로 궁금한 점이나, 더 자세한 상담이 필요하시면 언제든지 말씀해 주세요!<br>
                당신의 귀농 성공을 저희가 든든하게 응원합니다 🌿
            </div>
            """, unsafe_allow_html=True)


   
    
     # try-except 블록 끝난 뒤에는 들여쓰기 없이 작성
st.markdown("---")
if st.button("입력 페이지로 돌아가기"):
    for key in ['expected_date', 'region']:
        if key in st.session_state:
            del st.session_state[key]
    st.experimental_rerun()

if __name__ == '__main__':
    crop_weather_page()

