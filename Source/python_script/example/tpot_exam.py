import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tpot import TPOTRegressor

# 데이터 불러오기
file_path = './penguins.csv'
df = pd.read_csv(file_path)

# 데이터 확인
print("Data Preview:")
print(df.head())
print("\nData Information:")
print(df.info())
print("\nData Description:")
print(df.describe())

# 결측값 확인
print("\nMissing Values:")
print(df.isnull().sum())

# 데이터 전처리
df = df.dropna()
df = pd.get_dummies(df, drop_first=False)

# 인코딩 후 데이터 확인
print("\nEncoded Data Preview:")
print(df.head())

# 독립 변수(X)와 종속 변수(y) 설정
X = df.drop(columns=['body_mass_g'])
y = df['body_mass_g']

# 데이터 분할 (훈련 세트와 테스트 세트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TPOTRegressor 설정
tpot = TPOTRegressor(generations=10, population_size=50, verbosity=2, config_dict='TPOT sparse')

# 모델 학습
tpot.fit(X_train, y_train)

# 최적 모델 및 변수 확인
print("\nOptimal Pipeline:")
print(tpot.fitted_pipeline_)

# 예측 및 RMSE 계산
y_pred = tpot.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'\nRMSE: {rmse}')

# 최적 파이프라인 코드 출력
tpot.export('tpot_optimal_pipeline.py')