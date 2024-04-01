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

def fitness_function(ga_instance, solution, solution_idx):
    unit_costs = np.array([30, 25, 20, 15, 10])
    marketing_effort = np.array([2.0, 1.75, 1.5, 1.25, 1.0])
    
    total_profit = 0
    for i in range(5):
        sales = round((1000 / np.log(solution[i] + 200) - 141) * marketing_effort[i])
        profit = solution[i] * sales - (100 + unit_costs[i] * sales)
        total_profit += profit
    
    return total_profit

ga_instance = pygad.GA(num_generations=100,
                       num_parents_mating=5,
                       fitness_func=fitness_function,
                       sol_per_pop=50,
                       num_genes=5,
                       gene_space=[{'low': 1, 'high': 1000}] * 5,
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