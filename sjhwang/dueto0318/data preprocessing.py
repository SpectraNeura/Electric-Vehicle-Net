import pandas as pd
import numpy as np


test_df = pd.DataFrame(pd.read_csv('./test.csv'))
train_df = pd.DataFrame(pd.read_csv('./train.csv'))


##################################################################################
####                                 모델 기반 보간 방식                         ####
##################################################################################
from sklearn.ensemble import RandomForestRegressor

# 데이터 복사본 생성
train_df_imputed = train_df.copy()
test_df_imputed = test_df.copy()

# 예시로 사용할 피처 (필요에 따라 더 많은 변수 및 전처리 적용)
features = ['주행거리(km)', '보증기간(년)', '연식(년)']

# Train 데이터: 결측치가 없는 데이터로 모델 학습
train_known = train_df_imputed[train_df_imputed['배터리용량'].notna()]
train_missing = train_df_imputed[train_df_imputed['배터리용량'].isna()]

if not train_missing.empty:
    model_imputer = RandomForestRegressor(n_estimators=100, random_state=42)
    model_imputer.fit(train_known[features], train_known['배터리용량'])
    predicted_values = model_imputer.predict(train_missing[features])
    train_df_imputed.loc[train_df_imputed['배터리용량'].isna(), '배터리용량'] = predicted_values

# Test 데이터에도 동일하게 적용 (학습된 모델 혹은 별도 모델)
test_known = test_df_imputed[test_df_imputed['배터리용량'].notna()]
test_missing = test_df_imputed[test_df_imputed['배터리용량'].isna()]
if not test_missing.empty and not test_known.empty:
    model_imputer_test = RandomForestRegressor(n_estimators=100, random_state=42)
    model_imputer_test.fit(test_known[features], test_known['배터리용량'])
    predicted_values_test = model_imputer_test.predict(test_missing[features])
    test_df_imputed.loc[test_df_imputed['배터리용량'].isna(), '배터리용량'] = predicted_values_test



##################################################################################
####                                 평균 기반 보간 방식                         ####
##################################################################################


mean_battery = train_df['배터리용량'].mean()
train_df_mean = train_df.copy()
test_df_mean = test_df.copy()

train_df_mean['배터리용량'] = train_df_mean['배터리용량'].fillna(mean_battery)
test_df_mean['배터리용량'] = test_df_mean['배터리용량'].fillna(mean_battery)

##################################################################################
####                                 KNN 기반 보간 방식                         ####
##################################################################################


from sklearn.impute import KNNImputer

# 수치형 변수만 선택 (배터리용량 포함)
numeric_cols = ['주행거리(km)', '보증기간(년)', '연식(년)', '배터리용량']

knn_imputer = KNNImputer(n_neighbors=5)

train_df_knn = train_df.copy()
test_df_knn = test_df.copy()

train_df_knn[numeric_cols] = knn_imputer.fit_transform(train_df_knn[numeric_cols])
test_df_knn[numeric_cols] = knn_imputer.transform(test_df_knn[numeric_cols])

##################################################################################
####                             선형 보간 및 다항 보간                          ####
##################################################################################

# 선형 보간
train_df_linear = train_df.copy()
train_df_linear['배터리용량'] = train_df_linear['배터리용량'].interpolate(method='linear')

test_df_linear = test_df.copy()
test_df_linear['배터리용량'] = test_df_linear['배터리용량'].interpolate(method='linear')

# 다항 보간 (order=2 예시)
train_df_poly = train_df.copy()
train_df_poly['배터리용량'] = train_df_poly['배터리용량'].interpolate(method='polynomial', order=2)

test_df_poly = test_df.copy()
test_df_poly['배터리용량'] = test_df_poly['배터리용량'].interpolate(method='polynomial', order=2)

##################################################################################
####                                    CSV 저장                               ####
##################################################################################

import os

# 저장할 디렉터리 지정 및 없으면 생성
output_dir = "data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 모델 기반 보간 결과 저장
train_df_imputed.to_csv(os.path.join(output_dir, "train_imputed_model.csv"), index=False)
test_df_imputed.to_csv(os.path.join(output_dir, "test_imputed_model.csv"), index=False)

# 평균 기반 보간 결과 저장
train_df_mean.to_csv(os.path.join(output_dir, "train_imputed_mean.csv"), index=False)
test_df_mean.to_csv(os.path.join(output_dir, "test_imputed_mean.csv"), index=False)

# KNN 기반 보간 결과 저장
train_df_knn.to_csv(os.path.join(output_dir, "train_imputed_knn.csv"), index=False)
test_df_knn.to_csv(os.path.join(output_dir, "test_imputed_knn.csv"), index=False)

# 선형 보간 결과 저장
train_df_linear.to_csv(os.path.join(output_dir, "train_imputed_linear.csv"), index=False)
test_df_linear.to_csv(os.path.join(output_dir, "test_imputed_linear.csv"), index=False)

# 다항 보간 결과 저장
train_df_poly.to_csv(os.path.join(output_dir, "train_imputed_poly.csv"), index=False)
test_df_poly.to_csv(os.path.join(output_dir, "test_imputed_poly.csv"), index=False)


