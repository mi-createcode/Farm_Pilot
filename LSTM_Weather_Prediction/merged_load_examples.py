# 기상 예측 시스템 - LSTM 기반 기온 및 불쾌지수 예측 (2025년 이후 지원)
# 작성일: 2025년 6월 27일
# 목적: 전라도 지역의 기온과 불쾌지수를 LSTM 모델로 예측하여 시각화

import pandas as pd          # 데이터프레임 처리를 위한 라이브러리
import numpy as np           # 수치 계산을 위한 라이브러리
from tensorflow.keras.models import load_model  # 저장된 Keras 모델 불러오기
from tensorflow.keras.losses import MeanSquaredError  # MSE 손실함수 (모델 로딩 호환성)
import pickle               # 스케일러 객체 직렬화/역직렬화
import matplotlib.pyplot as plt  # 그래프 시각화
from matplotlib.dates import DateFormatter  # 날짜 형식 지정
import os                   # 파일 경로 처리

# --- 1. 설정부 (Configuration) ---
# 전역 설정: 모델별 경로, 피처, 하이퍼파라미터를 중앙에서 관리
# 장점: 새로운 모델 추가나 설정 변경 시 이 부분만 수정하면 됨
CONFIG = {
    # 기온 예측 모델 설정
    'temp': {
        'model_dir': r"C:\Users\User\Downloads\models",           # 기온 모델 저장 디렉토리
        'model_path_template': 'lstm_model_{region}.h5',          # 모델 파일명 템플릿 (지역별)
        'scaler_path_template': 'scaler_{region}.pkl',            # 스케일러 파일명 템플릿
        'target_col': '평균기온(℃)',                              # 예측 대상 컬럼
        # 모델 학습에 사용된 입력 피처들 (순서 중요!)
        'feature_cols': ['평균기온(℃)', '평균 습도(%)', '강수량(mm)', 'season_sin', 'season_cos'],
        'seq_length': 90,                                         # LSTM 입력 시퀀스 길이 (90일)
    },
    # 불쾌지수 예측 모델 설정
    'discomfort': {
        'model_dir': r"C:\Users\User\Downloads\models2",          # 불쾌지수 모델 저장 디렉토리
        'model_path_template': 'model_di_{region}.h5',            # 불쾌지수 모델 파일명 템플릿
        'scaler_features_path_template': 'scaler_features_di_{region}.pkl',  # 피처 스케일러
        'scaler_target_path_template': 'scaler_target_di_{region}.pkl',      # 타겟 스케일러
        'target_col': '불쾌지수',                                 # 예측 대상 컬럼
        # 불쾌지수 모델은 더 많은 피처 사용 (기온, 최고기온, 습도, 강수량, 계절성, 여름가중치)
        'feature_cols': ['평균기온(℃)', '최고기온(℃)', '평균 습도(%)', '강수량(mm)', 
                        'season_sin', 'season_cos', 'summer_weight'],
        'seq_length': 60,                                         # LSTM 입력 시퀀스 길이 (60일)
    }
}

# 전라도 기상 데이터 파일 경로
DATA_PATH = r"C:\Users\User\Downloads\아아아아\SYNM_jeollado_koppen_discomfort.csv"

# 최소 예측 연도 설정 - 수정된 부분
MIN_PREDICTION_YEAR = 2025  # 2025년 이후 예측 허용


class WeatherPredictor:
    """
    기상 데이터 예측을 위한 통합 클래스 (2025년 이후 지원)
    
    주요 기능:
    1. 모델과 스케일러 자동 로딩
    2. 과거 데이터 전처리 및 피처 엔지니어링
    3. LSTM 모델을 이용한 시계열 예측 (입력받은 날짜 ±15일)
    4. 예측 결과 시각화
    
    사용법:
    predictor = WeatherPredictor('temp', '군산')  # 군산 지역 기온 예측기 생성
    """
    
    def __init__(self, model_type, region):
        """
        WeatherPredictor 객체 초기화
        
        Args:
            model_type (str): 예측 모델 타입 ('temp' 또는 'discomfort')
            region (str): 예측할 지역명 (예: '군산', '전주', '목포')
            
        Raises:
            ValueError: 잘못된 모델 타입 입력 시 발생
        """
        # 입력 검증: 지원하는 모델 타입인지 확인
        if model_type not in CONFIG:
            raise ValueError(f"잘못된 모델 타입입니다: {model_type}. "
                           f"다음 중 하나여야 합니다: {list(CONFIG.keys())}")
        
        # 인스턴스 변수 초기화
        self.model_type = model_type        # 모델 타입 ('temp' or 'discomfort')
        self.region = region                # 예측 지역명
        self.config = CONFIG[model_type]    # 해당 모델의 설정 정보
        self.model = None                   # Keras 모델 객체 (초기에는 None)
        self.scalers = {}                   # 스케일러 객체들을 저장하는 딕셔너리

        # 모델과 스케일러 파일들을 자동으로 로드
        self._load_assets()

    def _load_assets(self):
        """
        설정 정보를 바탕으로 모델(.h5)과 스케일러(.pkl) 파일들을 로드
        
        처리 과정:
        1. 설정에서 모델 파일 경로 생성
        2. Keras 모델 로드 (MSE 호환성 문제 해결)
        3. 모델 타입에 따라 다른 스케일러 로딩 방식 적용
        
        Note:
            - 기온 모델: 단일 스케일러 사용
            - 불쾌지수 모델: 피처용, 타겟용 스케일러 분리
        """
        # 1. 모델 파일 경로 생성
        model_dir = self.config['model_dir']
        model_path = os.path.join(model_dir, 
                                 self.config['model_path_template'].format(region=self.region))
        
        try:
            # 2. Keras 모델 로드
            # custom_objects: 'mse' 함수 호환성 문제 해결을 위한 명시적 지정
            self.model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})
            print(f"✓ '{self.region}' 지역의 {self.model_type} 모델을 성공적으로 불러왔습니다.")
            
        except (IOError, FileNotFoundError) as e:
            print(f"❌ 오류: 모델 파일을 찾을 수 없습니다. 경로: {model_path}")
            raise e
        except Exception as e:
            print(f"❌ 모델 로딩 중 예상치 못한 오류 발생: {e}")
            raise e

        # 3. 스케일러 로드 (모델 타입에 따라 다른 방식)
        if 'scaler_path_template' in self.config:
            # 기온 모델: 단일 스케일러 사용
            scaler_path = os.path.join(model_dir, 
                                     self.config['scaler_path_template'].format(region=self.region))
            with open(scaler_path, 'rb') as f:
                self.scalers['main'] = pickle.load(f)
                
        elif 'scaler_features_path_template' in self.config:
            # 불쾌지수 모델: 피처와 타겟 스케일러 분리
            # 피처 스케일러 로드
            features_scaler_path = os.path.join(model_dir, 
                                              self.config['scaler_features_path_template'].format(region=self.region))
            with open(features_scaler_path, 'rb') as f:
                self.scalers['features'] = pickle.load(f)
            
            # 타겟 스케일러 로드
            target_scaler_path = os.path.join(model_dir, 
                                            self.config['scaler_target_path_template'].format(region=self.region))
            with open(target_scaler_path, 'rb') as f:
                self.scalers['target'] = pickle.load(f)
                
        print(f"✓ '{self.region}' 지역의 {self.model_type} 스케일러를 성공적으로 불러왔습니다.")

    def _prepare_historical_data(self, df):
        """
        예측에 필요한 과거 데이터를 전처리하고 피처 엔지니어링 수행
        
        Args:
            df (pd.DataFrame): 전체 기상 데이터
            
        Returns:
            pd.DataFrame: 전처리된 과거 데이터 (2022-2024년, 일별 평균)
            
        처리 단계:
        1. 해당 지역 데이터만 필터링
        2. 학습 기간 데이터 선택 (2022-2024년)
        3. 수치형 컬럼 변환 및 일별 리샘플링
        4. 계절성 피처 생성 (sin/cos 변환)
        5. 여름 가중치 피처 생성 (불쾌지수 모델용)
        6. 결측치 처리 (보간법 적용)
        """
        # 1. 해당 지역 데이터만 필터링
        station_df = df[df['지역'] == self.region].copy()
        
        # 2. 학습 기간 데이터 선택 (2022년 1월 1일 ~ 2024년 12월 31일)
        train_df = station_df[
            (station_df['날짜'] >= '2022-01-01') & 
            (station_df['날짜'] <= '2024-12-31')
        ].copy()
        
        # 날짜를 인덱스로 설정
        train_df.set_index('날짜', inplace=True)
        
        # 3. 수치형 컬럼 식별 (계절성, 가중치 피처 제외)
        numeric_cols = [col for col in self.config['feature_cols'] 
                       if 'sin' not in col and 'cos' not in col and 'weight' not in col]
        
        # 불쾌지수 모델의 경우 타겟 컬럼도 포함
        if self.model_type == 'discomfort':
            numeric_cols.insert(0, self.config['target_col'])

        # 4. 수치형 변환 및 일별 리샘플링
        for col in numeric_cols:
            train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
        
        # 중복 제거 후 일별 평균 계산
        train_df = train_df[list(set(numeric_cols))].resample('D').mean()
        
        # 5. 피처 엔지니어링
        # 날짜 기반 피처 생성
        train_df['day_of_year'] = train_df.index.dayofyear  # 연중 일수 (1-365)
        train_df['month'] = train_df.index.month            # 월 (1-12)
        
        # 계절성을 순환 피처로 변환 (sin/cos 변환)
        # 이유: 12월 31일과 1월 1일의 연속성을 보장하기 위함
        train_df['season_sin'] = np.sin(2 * np.pi * train_df['day_of_year'] / 365.25)
        train_df['season_cos'] = np.cos(2 * np.pi * train_df['day_of_year'] / 365.25)
        
        # 여름 가중치 피처 (불쾌지수 모델 전용)
        if 'summer_weight' in self.config['feature_cols']:
            # 여름철(6,7,8월)에는 1.5, 다른 달에는 1.0의 가중치 적용
            train_df['summer_weight'] = np.where(train_df['month'].isin([6, 7, 8]), 1.5, 1.0)
            
        # 6. 결측치 처리
        # 선형 보간 → 전방 채우기 → 후방 채우기 → 0으로 채우기 순서로 적용
        return train_df.interpolate().fillna(method='ffill').fillna(method='bfill').fillna(0)

    def predict(self, historical_df, target_date):
        """
        LSTM 모델을 사용하여 목표 날짜 ±15일 범위의 기상 데이터를 순차적으로 예측
        
        Args:
            historical_df (pd.DataFrame): 전처리된 과거 데이터
            target_date (pd.Timestamp): 예측 목표 날짜 (2025년 이후)
            
        Returns:
            tuple: (예측 날짜 배열, 예측값 배열)
            
        예측 과정:
        1. 초기 시퀀스 생성 (과거 seq_length일 데이터)
        2. 목표 날짜 ±15일 범위에 대해 순차 예측 (총 31일)
        3. 각 예측 후 시퀀스 업데이트 (sliding window)
        4. 예측값을 원래 스케일로 역변환
        """
        # 설정 정보 추출
        feature_cols = self.config['feature_cols']
        seq_length = self.config['seq_length']
        scaler_features = self.scalers.get('features', self.scalers.get('main'))

        # 1. 초기 시퀀스 생성
        # 과거 데이터를 스케일링하여 LSTM 입력 형태로 변환
        scaled_data = scaler_features.transform(historical_df[feature_cols])
        current_sequence = scaled_data[-seq_length:].reshape(1, seq_length, len(feature_cols))

        # 2. 예측 기간 설정 (목표 날짜 ±15일, 총 31일)
        future_dates = pd.date_range(
            start=target_date - pd.Timedelta(days=15), 
            end=target_date + pd.Timedelta(days=15)
        )
        future_preds_scaled = []  # 스케일링된 예측값들을 저장

        print(f"📊 '{self.region}' 지역의 {self.model_type} 예측을 시작합니다...")
        print(f"   예측 기간: {target_date.strftime('%Y년 %m월 %d일')} ±15일 (총 31일)")
        
        # 3. 순차적 예측 수행
        for future_date in future_dates:
            # 3-1. 현재 시퀀스로 다음 값 예측
            next_pred_scaled = self.model.predict(current_sequence, verbose=0)[0, 0]
            future_preds_scaled.append(next_pred_scaled)

            # 3-2. 다음 예측을 위한 입력 데이터 생성 (모델별로 다른 로직)
            if self.model_type == 'temp':
                # 기온 모델: 계절성과 과거 통계를 기반으로 피처 생성
                day_of_year = future_date.dayofyear
                month = future_date.month
                
                # 해당 월의 과거 평균값 사용 (없으면 전체 평균 사용)
                hist_humidity = historical_df[historical_df.index.month == month]['평균 습도(%)'].mean()
                hist_rainfall = historical_df[historical_df.index.month == month]['강수량(mm)'].mean()

                # 피처 배열 생성
                temp_features = np.array([[
                    0,  # 예측된 기온값으로 나중에 대체될 더미값
                    historical_df['평균 습도(%)'].mean() if pd.isna(hist_humidity) else hist_humidity,
                    historical_df['강수량(mm)'].mean() if pd.isna(hist_rainfall) else hist_rainfall,
                    np.sin(2 * np.pi * day_of_year / 365.25),  # 계절성 sin
                    np.cos(2 * np.pi * day_of_year / 365.25)   # 계절성 cos
                ]])
                
                # 스케일링 후 예측값으로 첫 번째 피처 대체
                scaled_new_features = scaler_features.transform(temp_features)[0]
                scaled_new_features[0] = next_pred_scaled
                new_step = scaled_new_features.reshape(1, 1, len(feature_cols))
                
            else:  # discomfort 모델
                # 불쾌지수 모델: 더 많은 피처와 여름 가중치 고려
                day_of_year, month = future_date.dayofyear, future_date.month
                
                # 새로운 피처 딕셔너리 생성
                new_features = {
                    'season_sin': np.sin(2 * np.pi * day_of_year / 365.25),
                    'season_cos': np.cos(2 * np.pi * day_of_year / 365.25),
                    'summer_weight': 1.5 if month in [6, 7, 8] else 1.0  # 여름 가중치
                }
                
                # 기상 변수들은 과거 동월 평균값 사용
                for col in ['평균기온(℃)', '최고기온(℃)', '평균 습도(%)', '강수량(mm)']:
                    val = historical_df.loc[historical_df['month'] == month, col].mean()
                    new_features[col] = historical_df[col].mean() if pd.isna(val) else val
                
                # 피처 배열 생성 및 스케일링
                new_features_array = np.array([[new_features[col] for col in feature_cols]])
                new_step = scaler_features.transform(new_features_array).reshape(1, 1, len(feature_cols))

            # 3-3. 시퀀스 업데이트 (sliding window 방식)
            # 가장 오래된 데이터는 제거하고 새로운 예측값을 추가
            current_sequence = np.append(current_sequence[:, 1:, :], new_step, axis=1)

        # 4. 예측 결과를 원래 스케일로 역변환
        if self.model_type == 'discomfort':
            # 불쾌지수: 타겟 전용 스케일러 사용
            predictions_inv = self.scalers['target'].inverse_transform(
                np.array(future_preds_scaled).reshape(-1, 1)
            )
        else:  # temp
            # 기온: 전체 피처 스케일러에서 첫 번째 컬럼(기온)만 추출
            preds_scaled = np.array(future_preds_scaled).reshape(-1, 1)
            dummy_features = np.zeros((len(preds_scaled), len(feature_cols) - 1))
            combined_preds = np.hstack([preds_scaled, dummy_features])
            predictions_inv = self.scalers['main'].inverse_transform(combined_preds)[:, 0]
            
        return future_dates, predictions_inv.flatten()

    def plot_results(self, dates, predictions, target_date):
        """
        예측 결과를 그래프로 시각화 (2025년 이후 지원)
        
        Args:
            dates (pd.DatetimeIndex): 예측 날짜 배열
            predictions (np.array): 예측값 배열
            target_date (pd.Timestamp): 사용자가 선택한 목표 날짜
            
        그래프 구성:
        - 빨간색 선: 예측 결과 (마커 포함)
        - 파란색 수직선: 사용자 선택 날짜
        - 격자: 가독성 향상
        - 한글 폰트: 제목 및 축 레이블
        """
        target_name = self.config['target_col']
        
        # 그래프 생성 (12x6 인치 크기)
        plt.figure(figsize=(12, 6))
        
        # 예측 결과 플롯 (빨간색 선, 원형 마커)
        plt.plot(dates, predictions, 
                label=f'{target_date.year}년 예상 {target_name}', 
                color='red', marker='o', markersize=5)
        
        # 사용자 선택 날짜 표시 (파란색 수직 점선)
        plt.axvline(target_date, color='blue', linestyle='--', linewidth=2, 
                   label=f'선택 날짜: {target_date.strftime("%Y-%m-%d")}')
        
        # 그래프 제목 및 축 레이블 설정 - 수정됨 (연도 정보 포함)
        plt.title(f"'{self.region}' 지역 {target_date.strftime('%Y년 %m월 %d일')} "
                 f"주변 1개월 {target_name} 예측")
        plt.xlabel('날짜', fontsize=12)
        plt.ylabel(target_name, fontsize=12)
        
        # 격자 및 범례 표시
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        # X축 날짜 형식 설정 (YYYY-MM-DD 형태로 변경 - 긴 기간 대응)
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)  # 날짜 레이블 45도 회전 (가독성 향상)
        
        # 레이아웃 자동 조정 및 그래프 표시
        plt.tight_layout()
        plt.show()


# --- 2. 예측 실행 헬퍼 함수 ---
def run_prediction(model_type, region, target_date, df):
    """
    지정된 모델 타입에 대한 전체 예측 파이프라인 실행 (2025년 이후 지원)
    
    Args:
        model_type (str): 예측 모델 타입 ('temp' 또는 'discomfort')
        region (str): 예측할 지역명
        target_date (pd.Timestamp): 예측 목표 날짜 (2025년 이후)
        df (pd.DataFrame): 전체 기상 데이터
        
    처리 과정:
    1. WeatherPredictor 객체 생성
    2. 과거 데이터 전처리
    3. 예측 수행 (±15일)
    4. 결과 시각화
    5. 오류 처리 및 사용자 피드백
    
    장점:
    - 코드 중복 제거
    - 일관된 오류 처리
    - 사용자 친화적 메시지
    """
    # 모델 타입별 한글 이름 매핑
    model_names = {
        'temp': '기온',
        'discomfort': '불쾌지수'
    }
    model_name = model_names.get(model_type, model_type.capitalize())
    
    print(f"\n🔄 {model_name} 예측을 시작합니다...")
    
    try:
        # 1. 예측기 객체 생성 (모델과 스케일러 자동 로드)
        predictor = WeatherPredictor(model_type=model_type, region=region)
        
        # 2. 과거 데이터 전처리 및 피처 엔지니어링
        historical_data = predictor._prepare_historical_data(df)
        
        # 3. LSTM 모델을 이용한 시계열 예측 (±15일)
        pred_dates, predictions = predictor.predict(historical_data, target_date)
        
        # 4. 예측 결과 시각화
        predictor.plot_results(pred_dates, predictions, target_date)
        
        print(f"✅ {model_name} 예측 및 그래프 생성이 완료되었습니다.")
        
    except Exception as e:
        # 오류 발생 시 사용자 친화적 메시지 출력
        print(f"\n❌ 오류: {model_name} 예측 중 문제가 발생했습니다: {e}")
        print("🔧 다음 사항을 확인해주세요:")
        print("   - 모델 파일(.h5)이 올바른 경로에 있는지")
        print("   - 스케일러 파일(.pkl)이 올바른 경로에 있는지")
        print("   - 데이터 파일에 해당 지역의 데이터가 충분한지")
        print("   - 파일 권한 문제가 없는지")


# --- 3. 메인 실행부 ---
def main():
    """
    프로그램의 진입점 - 사용자 인터페이스 및 전체 워크플로우 관리 (2025년 이후 지원)
    
    실행 순서:
    1. 환경 설정 (한글 폰트, 마이너스 표시)
    2. 데이터 로드 및 전처리
    3. 사용자 입력 처리 (지역명, 예측 날짜) - 2025년 이후 허용
    4. 기온 예측 실행
    5. 불쾌지수 예측 실행
    6. 완료 메시지 출력
    
    사용자 입력 검증:
    - 지역명: 데이터에 존재하는 지역인지 확인
    - 날짜: 2025년 이후인지, 올바른 형식인지 확인 - 수정됨
    """
    # 1. 한글 폰트 및 그래프 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'    # Windows 한글 폰트
    plt.rcParams['axes.unicode_minus'] = False       # 마이너스 기호 정상 표시
    
    # 2. 데이터 로드 및 기본 전처리
    print("📂 데이터를 불러오는 중...")
    df = pd.read_csv(DATA_PATH)
    df['날짜'] = pd.to_datetime(df['날짜'])            # 날짜 컬럼을 datetime 타입으로 변환
    available_regions = df['지역'].unique()           # 사용 가능한 지역 목록 추출
    print(f"✅ 데이터 로드 완료. 사용 가능한 지역: {len(available_regions)}개")

    # 3. 사용자 입력 처리: 지역명
    print(f"\n🌍 사용 가능한 지역: {', '.join(available_regions)}")
    while True:
        user_region = input(f"예측할 지역명을 입력하세요 "
                           f"(예: {', '.join(available_regions[:3])}...): ").strip()
        
        if user_region in available_regions:
            print(f"✅ '{user_region}' 지역을 선택했습니다.")
            break
        else:
            print(f"❌ 오류: '{user_region}'은(는) 데이터가 없습니다.")
            print(f"   사용 가능한 지역: {', '.join(available_regions)}")

    # 4. 사용자 입력 처리: 예측 날짜 (2025년 이후 허용) - 수정된 부분
    print(f"\n📅 예측 지원 연도: {MIN_PREDICTION_YEAR}년 이후")
    print(f"   현재 시점: 2025년 6월 27일 기준")
    while True:
        try:
            input_date_str = input(f"예측을 보고 싶은 날짜를 입력하세요 "
                                  f"({MIN_PREDICTION_YEAR}년 이후, YYYY-MM-DD): ")
            target_date = pd.to_datetime(input_date_str)
            
            # 2025년 이후 날짜인지 확인 - 수정됨 (>= 조건으로 2025년 포함)
            if target_date.year >= MIN_PREDICTION_YEAR:
                print(f"✅ {target_date.strftime('%Y년 %m월 %d일')}을 선택했습니다.")
                break
            else:
                print(f"❌ 오류: {MIN_PREDICTION_YEAR}년 이후의 날짜만 입력할 수 있습니다.")
                print(f"   입력하신 날짜: {target_date.year}년")
                
        except ValueError:
            print("❌ 오류: 잘못된 날짜 형식입니다. YYYY-MM-DD 형식으로 입력해주세요.")
            print(f"   예시: {MIN_PREDICTION_YEAR}-10-15, 2026-01-01, 2030-12-31")
            
    # 5. 예측 순차 실행
    print(f"\n🚀 '{user_region}' 지역의 {target_date.strftime('%Y년 %m월 %d일')} 예측을 시작합니다...")
    print(f"   예측 범위: 선택 날짜 ±15일 (총 31일)")
    
    # 5-1. 기온 예측 실행
    run_prediction('temp', user_region, target_date, df)
    
    # 5-2. 불쾌지수 예측 실행
    run_prediction('discomfort', user_region, target_date, df)

    # 6. 완료 메시지
    print(f"\n🎉 모든 예측 프로세스가 완료되었습니다!")
    print(f"   - 대상 지역: {user_region}")
    print(f"   - 예측 날짜: {target_date.strftime('%Y년 %m월 %d일')}")
    print(f"   - 예측 범위: ±15일 (총 31일)")
    print(f"   - 생성된 그래프: 기온 예측, 불쾌지수 예측")


# 프로그램 실행부
if __name__ == '__main__':
    """
    스크립트가 직접 실행될 때만 main() 함수 호출
    - 모듈로 import될 때는 실행되지 않음
    - 코드의 재사용성과 테스트 용이성 향상
    """
    print("="*60)
    print("🌤️  LSTM 기반 기상 예측 시스템")
    print("   - 기온 및 불쾌지수 예측")
    print("   - 전라도 지역 대상")
    print("   - 2025년 이후 예측 지원")  # 수정된 부분
    print("="*60)
    
    main() # 메인 함수 실행
