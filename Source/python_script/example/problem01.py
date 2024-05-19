import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import matplotlib.pyplot as plt

# 데이터 불러오기
data = pd.read_csv('./penguins.csv')

# 데이터 전처리
data = data.dropna()
data = pd.get_dummies(data, drop_first=False)

# 독립 변수(X)와 종속 변수(y) 분리
X = data.drop('body_mass_g', axis=1)
y = data['body_mass_g']

# 학습 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class PenguinWeightPredictionProblem(Problem):

    def __init__(self):
        super().__init__(n_var=X_train.shape[1], n_obj=2, n_constr=0, xl=0, xu=1, type_var=bool)

    def _evaluate(self, x, out, *args, **kwargs):
        rmse_values = []
        feature_counts = []

        for individual in x:
            selected_features = [i for i, bit in enumerate(individual) if bit]
            if len(selected_features) == 0:
                rmse = 10000  # 임의의 큰 값
                feature_count = X_train.shape[1]
            else:
                model = LinearRegression()
                model.fit(X_train.iloc[:, selected_features], y_train)
                predictions = model.predict(X_test.iloc[:, selected_features])
                rmse = np.sqrt(mean_squared_error(y_test, predictions))
                feature_count = len(selected_features)

            rmse_values.append(rmse)
            feature_counts.append(feature_count)

        out["F"] = np.column_stack([rmse_values, feature_counts])

# 문제 정의
problem = PenguinWeightPredictionProblem()

# 초기 개체 생성 함수
class CustomBinaryRandomSampling(BinaryRandomSampling):
    def _do(self, problem, n_samples, **kwargs):
        samples = super()._do(problem, n_samples, **kwargs)
        # 초기에는 feature를 적게 선택
        for i in range(n_samples):
            num_features = np.random.randint(1, problem.n_var // 2)
            selected_indices = np.random.choice(problem.n_var, num_features, replace=False)
            samples[i, :] = 0
            samples[i, selected_indices] = 1
        return samples

# 알고리즘 설정
algorithm = NSGA2(
    pop_size=100,
    sampling=CustomBinaryRandomSampling(),
    crossover=TwoPointCrossover(),
    mutation=BitflipMutation(),
    eliminate_duplicates=True
)

# 최적화 실행
res = minimize(problem,
               algorithm,
               ('n_gen', 100),
               seed=1,
               verbose=True,
               save_history=True)

# 결과 출력
print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

# 각 솔루션에서 선택된 feature의 이름을 출력
feature_names = X.columns

for i, solution in enumerate(res.X):
    selected_features = [feature_names[j] for j, selected in enumerate(solution) if selected]
    print(f"솔루션 {i+1}: 선택된 feature = {selected_features}, RMSE = {res.F[i, 0]}, Feature 개수 = {res.F[i, 1]}")

# Generation 별 데이터 추출
n_generations = len(res.history)
rmse_per_gen = []
features_per_gen = []

for gen in range(n_generations):
    opt = res.history[gen].pop
    rmse = opt.get("F")[:, 0]
    features = opt.get("F")[:, 1]
    rmse_per_gen.append(np.mean(rmse))
    features_per_gen.append(np.mean(features))

# RMSE 그래프 출력
plt.figure(figsize=(10, 5))
plt.plot(range(n_generations), rmse_per_gen, color='tab:red')
plt.xlabel('Generation')
plt.ylabel('RMSE')
plt.title('RMSE over Generations')
plt.grid(True)
plt.show()

# Feature 개수 그래프 출력
plt.figure(figsize=(10, 5))
plt.plot(range(n_generations), features_per_gen, color='tab:blue')
plt.xlabel('Generation')
plt.ylabel('Number of Features')
plt.title('Number of Features over Generations')
plt.grid(True)
plt.show()

# 최적 솔루션 선택 (RMSE가 가장 낮은 솔루션)
best_idx = np.argmin(res.F[:, 0])
best_solution = res.X[best_idx]
selected_features = [X.columns[i] for i, bit in enumerate(best_solution) if bit]

print(f"Selected Features: {selected_features}")

# 파레토 최적해 시각화
pareto_front = res.F

plt.figure(figsize=(10, 5))
plt.scatter(pareto_front[:, 0], pareto_front[:, 1], c='blue', marker='o')
plt.xlabel('RMSE')
plt.ylabel('Number of Features')
plt.title('Pareto Front')
plt.grid(True)
plt.show()