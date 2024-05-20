import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_curve, auc
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

# 데이터 불러오기
data = pd.read_csv('./penguins.csv')

# 데이터 전처리
data = data.dropna()
data = pd.get_dummies(data, drop_first=False)

# 최적화된 Feature 사용
selected_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm']

# 독립 변수(X)와 종속 변수(y) 분리
X = data[selected_features]
y = data[['species_Adelie', 'species_Chinstrap', 'species_Gentoo']].idxmax(axis=1)  # 종을 예측할 목표 변수로 설정
y = [1 if label == 'species_Adelie' else 0 for label in y]

# 데이터 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 학습 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 로지스틱 회귀 모델 학습
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 유전 알고리즘을 위한 설정
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    threshold = individual[0]
    probs = model.predict_proba(X_test)[:, 1]
    predictions = (probs >= threshold).astype(int)
    
    accuracy = accuracy_score(y_test, predictions)
    return accuracy,

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutPolynomialBounded, low=0, up=1, eta=1.0, indpb=0.2)
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
    best_threshold = hof[0][0]
    best_threshold = np.round(best_threshold, 4)  # 소수점 이하 4자리까지 반올림
    print("Best Threshold: ", best_threshold)
    
    probs = model.predict_proba(X_test)[:, 1]
    final_predictions = (probs >= best_threshold).astype(int)
    default_predictions = (probs >= 0.5).astype(int)
    
    final_accuracy = accuracy_score(y_test, final_predictions)
    default_accuracy = accuracy_score(y_test, default_predictions)
    print("Final Accuracy: ", final_accuracy)
    print("Default Accuracy: ", default_accuracy)

    # 유전 알고리즘 수렴 과정 시각화
    gen = logbook.select("gen")
    max_fitness = logbook.select("max")
    avg_fitness = logbook.select("avg")

    plt.figure(figsize=(10, 5))
    plt.plot(gen, max_fitness, label="Maximum Fitness")
    plt.plot(gen, avg_fitness, label="Average Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Accuracy")
    plt.title("Genetic Algorithm Convergence")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 임계값 0.5일 때의 ROC 곡선
    fpr_default, tpr_default, _ = roc_curve(y_test, default_predictions)
    roc_auc_default = auc(fpr_default, tpr_default)

    # 최적의 임계값일 때의 ROC 곡선
    fpr_best, tpr_best, _ = roc_curve(y_test, final_predictions)
    roc_auc_best = auc(fpr_best, tpr_best)

    # ROC 곡선 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(fpr_default, tpr_default, color='blue', lw=2, label='ROC curve (Threshold = 0.5, AUC = %0.2f)' % roc_auc_default)
    plt.plot(fpr_best, tpr_best, color='red', lw=2, linestyle='--', label='ROC curve (Best Threshold = %0.2f, AUC = %0.2f)' % (best_threshold, roc_auc_best))
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()