import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

# 데이터 불러오기
data = pd.read_csv('./penguins.csv')

# 데이터 전처리
data = data.dropna()
data = pd.get_dummies(data, drop_first=True)

# 최적화된 Feature 사용
selected_features = ['bill_depth_mm', 'flipper_length_mm', 'species_Gentoo', 'sex_MALE']

# 독립 변수(X)와 종속 변수(y) 분리
X = data[selected_features]
y = data['body_mass_g']

# 학습 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 정의
models = {
    'lr': LinearRegression(),
    'dt': DecisionTreeRegressor(),
    'rf': RandomForestRegressor(),
    'svr': SVR(),
    'knn': KNeighborsRegressor()
}

# 모델 학습
for model in models.values():
    model.fit(X_train, y_train)

# 앙상블 예측 함수
def ensemble_predict(weights, X):
    weights = np.array(weights)
    weights /= weights.sum()  # 가중치 합이 1이 되도록 정규화
    predictions = np.zeros(X.shape[0])
    for weight, model in zip(weights, models.values()):
        predictions += weight * model.predict(X)
    return predictions

# 유전 알고리즘을 위한 설정
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(models))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    predictions = ensemble_predict(individual, X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return (rmse,)

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutPolynomialBounded, low=0, up=1, eta=1.0, indpb=0.4)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

def main():
    np.random.seed(42)
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=100, stats=stats, halloffame=hof, verbose=True)

    return pop, hof, logbook

if __name__ == "__main__":
    pop, hof, logbook = main()
    best_weights = hof[0]
    best_weights = np.round(best_weights, 4)  # 소수점 이하 4자리까지 반올림
    best_weights /= best_weights.sum()  # 가중치 합이 1이 되도록 정규화
    print("Best Weights: ", best_weights)
    final_predictions = ensemble_predict(best_weights, X_test)
    final_rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
    print("Final RMSE: ", final_rmse)

    # 유전 알고리즘 수렴 과정 시각화
    gen = logbook.select("gen")
    min_fitness = logbook.select("min")
    avg_fitness = logbook.select("avg")

    plt.figure(figsize=(10, 5))
    plt.plot(gen, min_fitness, label="Minimum Fitness")
    plt.plot(gen, avg_fitness, label="Average Fitness")
    plt.xlabel("Generation")
    plt.ylabel("RMSE")
    plt.title("Genetic Algorithm Convergence")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 최종 앙상블 모델의 예측값과 실제값 비교 시각화
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, final_predictions, color='blue', alpha=0.5, label='Predicted')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Prediction')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Residual Plot
    residuals = y_test - final_predictions
    plt.figure(figsize=(10, 5))
    plt.scatter(final_predictions, residuals, color='blue', alpha=0.5)
    plt.hlines(0, final_predictions.min(), final_predictions.max(), colors='red', linestyles='dashed')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.grid(True)
    plt.show()
