import pygad
import numpy as np
import matplotlib.pyplot as plt

# 각 세대의 적합도를 기록하기 위한 글로벌 리스트
best_fitness = []
mean_fitness = []

def callback_generation(ga_instance):
    global best_fitness, mean_fitness
    fitness = ga_instance.last_generation_fitness  # 이것이 각 세대의 적합도 점수를 얻는 방법입니다.
    best_fitness.append(np.max(fitness))
    mean_fitness.append(np.mean(fitness))

def fitness_function_sphere(ga_instance, solution, solution_idx):
    return -np.sum(solution**2)  # 최소화 문제이므로 음수를 취함


D = 2

ga_instance = pygad.GA(num_generations=100,
                       num_parents_mating=5,
                       fitness_func=fitness_function_sphere,  # 여기서 사용할 적합도 함수를 지정
                       sol_per_pop=50,
                       num_genes=D,  # 여기서 D는 문제의 차원 수
                       gene_space=[{'low': -5.12, 'high': 5.12}] * D,  # 일반적으로 rastrigin 함수의 범위
                       mutation_percent_genes=10,
                       mutation_num_genes=1,
                       parent_selection_type="sss",
                       crossover_type="single_point",
                       mutation_type="random",
                       keep_parents=1)

ga_instance.on_generation = callback_generation

ga_instance.run()

# 결과 출력
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"최적 가격 설정: {solution}")
print(f"예상 최대 수익: {solution_fitness}")

# best만
ga_instance.plot_fitness()

# best / mean 비교 그래프
plt.figure(figsize=(12, 6))
plt.plot(best_fitness, label='Best Fitness', color='black')
plt.plot(mean_fitness, label='Mean Fitness', linestyle='--', color='grey')
plt.title('Generation vs. Fitness')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend(loc='best')
plt.show()

last_generation_solutions = ga_instance.population
last_generation_fitness = ga_instance.last_generation_fitness
normalized_fitness = (last_generation_fitness - np.min(last_generation_fitness)) / np.ptp(last_generation_fitness)

# 스캐터 플롯 생성
plt.figure(figsize=(5, 5))
plt.scatter(last_generation_solutions[:, 0], last_generation_solutions[:, 1], 
            c=normalized_fitness, cmap='Greys', alpha=0.5)
plt.title('Scatter Plot of the Last Generation Solutions')
plt.xlabel('x1')
plt.ylabel('x2')
plt.colorbar(label='Fitness')
plt.show()