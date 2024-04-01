from scipy.optimize import differential_evolution
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution
D=2
# 최적화 과정에서 각 세대의 솔루션을 저장할 글로벌 변수
population_history = []

# 콜백 함수 정의
def callback(xk, convergence):
    global population_history
    population_history.append(xk)

def fitness_function_sphere(solution):
    return np.sum(solution**2)  # 최소화 문제이므로 음수를 취하지 않음

# 변수의 범위 설정
bounds = [(-5.2, 5.2)] * D

# Differential Evolution 알고리즘 실행
result = differential_evolution(fitness_function_sphere, bounds, maxiter=100, popsize=1000, callback=callback)

# 결과 출력
print(f"최적 해: {result.x}")
print(f"최적 해의 적합도: {result.fun}")

# 진화 과정 시각화
# 주의: 콜백 함수는 인구 전체가 아닌 각 세대의 최적 해만을 저장합니다.
population_array = np.array(population_history)
plt.figure(figsize=(10, 10))
plt.subplot(2,1,1)
plt.scatter(range(len(population_array)), population_array[:, 0], color = 'black', alpha=0.5, label='x1')
plt.title('Par1')
plt.xlabel('stored population')
plt.ylabel('Value')
plt.legend()

plt.subplot(2,1,2)
plt.scatter(range(len(population_array)), population_array[:, 1], color = 'black', alpha=0.5, label='x2')
plt.title('Par2')
plt.xlabel('stored population')
plt.ylabel('Value')
plt.legend()

plt.show()
