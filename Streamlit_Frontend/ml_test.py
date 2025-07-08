import joblib
import numpy as np
import pandas as pd
import os

# 모델과 전처리기 로딩
load_dir = "mlmodel"
model = joblib.load(os.path.join(load_dir, 'total_xgb_model.pkl'))
scaler = joblib.load(os.path.join(load_dir, 'total_scaler.pkl'))
sanji_encoder = joblib.load(os.path.join(load_dir, 'total_sanji_encoder.pkl'))
season_encoder = joblib.load(os.path.join(load_dir, 'total_season_encoder.pkl'))
prdlst_encoder = joblib.load(os.path.join(load_dir, 'total_prdlst_encoder.pkl'))

# ✅ 예측 함수 정의
def predict_top3_product(sanji_nm, year, month, day):
    # 계절 추출
    season = {
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'fall', 10: 'fall', 11: 'fall'
    }[month]

    '''
    try:
        sanji_enc = sanji_encoder.transform([sanji_nm])[0]
    except ValueError:
        raise ValueError(f"'{sanji_nm}'는 학습되지 않은 지역입니다.")

    try:
        season_enc = season_encoder.transform([season])[0]
    except ValueError:
        raise ValueError(f"'{season}'는 학습되지 않은 계절입니다.")
    '''
    
    sanji_enc = sanji_encoder.transform([sanji_nm])[0]
    season_enc = season_encoder.transform([season])[0]

    # 입력 생성
    input_df = pd.DataFrame([{
        'SANJI_ENC': sanji_enc,
        'YEAR': year,
        'MONTH': month,
        'DAY': day,
        'SEASON_ENC': season_enc
    }])

    # 스케일링
    input_scaled = scaler.transform(input_df)

    # 예측 확률
    prob = model.predict_proba(input_scaled)[0]
    top3_idx = np.argsort(prob)[-3:][::-1]  # 상위 3개 인덱스

    # 결과 변환
    top3_labels = prdlst_encoder.inverse_transform(top3_idx)
    top3_scores = prob[top3_idx]

    # 출력
    print(f"입력값: 지역={sanji_nm}, 날짜={year}-{month:02}-{day:02}, 계절={season}")
    print("Top-3 예측 농작물:")
    for i, (label, score) in enumerate(zip(top3_labels, top3_scores), 1):
        print(f"{i}. {label}")

    return list(zip(top3_labels, top3_scores))