import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

# 데이터 불러오기
data = pd.read_csv('./penguins.csv')
data.dropna(inplace=True)

# 피처와 레이블 설정
X = data[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm']]
y = data['species']

# 범주형 데이터를 숫자로 변환
y = y.astype('category').cat.codes

# 학습 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 정규화
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 선형 회귀 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 회귀식의 예측값 계산
pred_values = model.predict(X_test)

# 유전 알고리즘을 사용하여 임계값 최적화
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, -3.0, 3.0)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def classify_by_threshold(pred_value, thresholds):
    if pred_value < thresholds[0]:
        return 0
    elif thresholds[0] <= pred_value <= thresholds[1]:
        return 1
    else:
        return 2

def evaluate(individual):
    thresholds = sorted(individual)
    y_pred = np.array([classify_by_threshold(pred, thresholds) for pred in pred_values])
    return accuracy_score(y_test, y_pred),

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

def main():
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, stats=stats, halloffame=hof, verbose=True)

    return pop, logbook, hof

if __name__ == "__main__":
    pop, logbook, hof = main()
    best_thresholds = sorted(hof[0])
    print(f'Best thresholds: {best_thresholds}')
    y_pred = np.array([classify_by_threshold(pred, best_thresholds) for pred in pred_values])
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Optimized Accuracy: {accuracy:.2f}')
    print(classification_report(y_test, y_pred, target_names=data['species'].astype('category').cat.categories.tolist()))

    # 학습 과정 그래프
    gen = logbook.select("gen")
    fit_avg = logbook.select("avg")
    fit_max = logbook.select("max")

    plt.figure(figsize=(12, 6))
    plt.plot(gen, fit_avg, label='Average Fitness')
    plt.plot(gen, fit_max, label='Max Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Evolution of Fitness over Generations')
    plt.legend()
    plt.grid()
    plt.show()

    # ROC 곡선 그리기
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, pred_values)
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(12, 6))
    for i in range(3):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()
