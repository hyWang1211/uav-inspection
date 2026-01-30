import sys
import os
# current_file_path = os.path.abspath(__file__)
# current_dir = os.path.dirname(current_file_path)
# project_root = os.path.dirname(os.path.dirname(current_dir))
# if project_root not in sys.path:
#     sys.path.append(project_root)
    
import numpy as np
import random
import math
import pandas as pd
from Model import config

class MetaheuristicSolvers:
    def __init__(self, physical_model):
        self.pm = physical_model
        self.n_nests = len(physical_model.nest_locations)
        self.n_turbines = len(physical_model.wind_turbines_df)
        self.R_lk = self.getRlk()
        
        # 预计算每个风机的可行机巢列表（加速用）
        self.feasible_nests_for_turbine = []
        for k in range(self.n_turbines):
            feasible = np.where(self.R_lk[:, k] == 1)[0]
            if len(feasible) == 0:
                print(f"Warning: Turbine {k} is unreachable!")
            self.feasible_nests_for_turbine.append(feasible)
            
        # 成本参数
        self.base_costs = config.NEST_BASE_COSTS
        self.uav_cost = config.OMEGA_2
        self.eta = config.ETA
        self.max_type = max(self.base_costs.keys())

    def getRlk(self):
        # 重新获取或计算 R_lk，避免传递复杂对象
        return self.pm.calculate_reachability_matrix(self.pm.nest_locations, self.pm.wind_turbines_df)

    def calculate_solution_cost(self, assignment_array):
        """
        通用成本计算函数
        Input: assignment_array (长度为 n_turbines 的数组，值为 nest_id)
        Output: Total Cost
        """
        # 1. 统计负载
        nest_loads = {}
        for k, nest_id in enumerate(assignment_array):
            if nest_id == -1: return float('inf') # 存在未覆盖风机，惩罚无限大
            nest_loads[nest_id] = nest_loads.get(nest_id, 0) + 1
            
        total_cost = 0
        
        # 2. 计算每个激活机巢的成本
        for nest_id, load in nest_loads.items():
            if load == 0: continue
            
            # 这里的逻辑要和 G-RSO 保持一致
            # 如果负载超过 16 (Max Type * ETA)，在 G-RSO 里是不允许的
            # 在元启发式里，如果超了，我们要么给惩罚，要么假设建了多个机巢
            # 为了对比公平，我们假设超载部分需要建新的机巢（叠加成本）
            
            remaining_load = load
            while remaining_load > 0:
                # 本次能处理多少
                current_batch = min(remaining_load, self.max_type * self.eta)
                
                # 计算这一批的成本
                n_uavs = math.ceil(current_batch / self.eta)
                nest_type = n_uavs
                if nest_type < 1: nest_type = 1
                
                cost = self.base_costs[nest_type] + nest_type * self.uav_cost
                total_cost += cost
                
                remaining_load -= current_batch
                
        return total_cost

    def generate_random_solution(self):
        """生成一个满足物理可达性的随机解"""
        assignment = np.zeros(self.n_turbines, dtype=int)
        for k in range(self.n_turbines):
            options = self.feasible_nests_for_turbine[k]
            if len(options) > 0:
                assignment[k] = np.random.choice(options)
            else:
                assignment[k] = -1
        return assignment

    # ==========================================
    # 1. Genetic Algorithm (GA)
    # ==========================================
    def run_ga(self, pop_size=50, generations=100, mutation_rate=0.1):
        print(f"--- Running GA (Pop: {pop_size}, Gen: {generations}) ---")
        
        # 初始化种群
        population = [self.generate_random_solution() for _ in range(pop_size)]
        best_solution = None
        best_cost = float('inf')
        
        history = []

        for gen in range(generations):
            # 评估适应度
            costs = [self.calculate_solution_cost(ind) for ind in population]
            
            # 记录最佳
            min_cost_idx = np.argmin(costs)
            if costs[min_cost_idx] < best_cost:
                best_cost = costs[min_cost_idx]
                best_solution = population[min_cost_idx].copy()
            
            history.append(best_cost)
            
            # 选择 (锦标赛)
            new_population = []
            for _ in range(pop_size):
                parent1 = population[np.random.randint(pop_size)]
                parent2 = population[np.random.randint(pop_size)]
                winner = parent1 if self.calculate_solution_cost(parent1) < self.calculate_solution_cost(parent2) else parent2
                new_population.append(winner.copy()) # 简化版：直接复制赢家作为下一代基底
            
            # 交叉 (单点交叉)
            for i in range(0, pop_size, 2):
                if i+1 < pop_size and random.random() < 0.8:
                    pt = random.randint(1, self.n_turbines-1)
                    # 交换
                    temp = new_population[i][pt:].copy()
                    new_population[i][pt:] = new_population[i+1][pt:]
                    new_population[i+1][pt:] = temp
            
            # 变异
            for i in range(pop_size):
                if random.random() < mutation_rate:
                    # 随机选一个风机，换一个可行的机巢
                    k = random.randint(0, self.n_turbines-1)
                    options = self.feasible_nests_for_turbine[k]
                    if len(options) > 0:
                        new_population[i][k] = np.random.choice(options)
            
            population = new_population
            
        print(f"GA Best Cost: {best_cost:.2f}")
        return best_cost, history

    # ==========================================
    # 2. Particle Swarm Optimization (PSO)
    # ==========================================
    def run_pso(self, swarm_size=50, iterations=100):
        print(f"--- Running PSO (Swarm: {swarm_size}, Iter: {iterations}) ---")
        # PSO 适配离散问题比较勉强，这里使用一种映射方法
        # 粒子位置 X 是一个 [n_turbines] 的浮点向量
        # 每一维的值范围是 [0, n_nests-1]
        
        # 初始化
        X = np.random.uniform(0, self.n_nests-1, (swarm_size, self.n_turbines))
        V = np.random.uniform(-1, 1, (swarm_size, self.n_turbines))
        
        P_best = X.copy()
        P_best_scores = np.full(swarm_size, float('inf'))
        
        G_best = X[0].copy()
        G_best_score = float('inf')
        
        history = []
        
        # 解码函数：Float -> Valid Integer Assignment
        def decode(pos_vector):
            assignment = np.zeros(self.n_turbines, dtype=int)
            for k in range(self.n_turbines):
                # 1. 转整数
                target_nest = int(round(pos_vector[k]))
                target_nest = max(0, min(self.n_nests-1, target_nest))
                
                # 2. 修复约束 (找到离 target_nest 最近的可行机巢)
                options = self.feasible_nests_for_turbine[k]
                if len(options) > 0:
                    # 在 options 里找一个数值上最接近 target_nest 的
                    # 注意：这里仅仅是数值索引接近，不代表地理位置接近，但在PSO逻辑里是可以的
                    best_option = options[np.argmin(np.abs(options - target_nest))]
                    assignment[k] = best_option
                else:
                    assignment[k] = -1
            return assignment

        w, c1, c2 = 0.7, 1.5, 1.5
        
        for it in range(iterations):
            for i in range(swarm_size):
                # 解码并计算适应度
                assignment = decode(X[i])
                score = self.calculate_solution_cost(assignment)
                
                # 更新个体最优
                if score < P_best_scores[i]:
                    P_best_scores[i] = score
                    P_best[i] = X[i].copy()
                    
                # 更新全局最优
                if score < G_best_score:
                    G_best_score = score
                    G_best = X[i].copy()
            
            history.append(G_best_score)
            
            # 更新速度和位置
            r1 = np.random.rand(swarm_size, self.n_turbines)
            r2 = np.random.rand(swarm_size, self.n_turbines)
            V = w*V + c1*r1*(P_best - X) + c2*r2*(G_best - X)
            X = X + V
            X = np.clip(X, 0, self.n_nests-1)
            
        print(f"PSO Best Cost: {G_best_score:.2f}")
        return G_best_score, history

    # ==========================================
    # 3. Ant Colony Optimization (ACO)
    # ==========================================
    def run_aco(self, n_ants=30, iterations=50, evaporation=0.1, alpha=1.0, beta=2.0):
        print(f"--- Running ACO (Ants: {n_ants}, Iter: {iterations}) ---")
        
        # 信息素矩阵 [n_turbines][n_nests]
        # 初始化为一个小常数
        pheromone = np.ones((self.n_turbines, self.n_nests)) * 0.1
        
        # 启发式信息 (1/距离)
        # 预计算 heuristic[k][l]
        heuristic = np.zeros((self.n_turbines, self.n_nests))
        nest_coords = self.pm.nest_locations[['lon', 'lat']].values
        turb_coords = self.pm.wind_turbines_df[['lon', 'lat']].values
        
        for k in range(self.n_turbines):
            options = self.feasible_nests_for_turbine[k]
            for l in options:
                dist = np.linalg.norm(nest_coords[l] - turb_coords[k])
                heuristic[k][l] = 1.0 / (dist + 1e-6) # 避免除以0
        
        best_cost = float('inf')
        history = []
        
        for it in range(iterations):
            solutions = []
            costs = []
            
            # 每只蚂蚁构建解
            for ant in range(n_ants):
                assignment = np.zeros(self.n_turbines, dtype=int)
                for k in range(self.n_turbines):
                    options = self.feasible_nests_for_turbine[k]
                    if len(options) == 0:
                        assignment[k] = -1
                        continue
                        
                    # 计算概率 P = tau^alpha * eta^beta
                    probs = []
                    for l in options:
                        p = (pheromone[k][l] ** alpha) * (heuristic[k][l] ** beta)
                        probs.append(p)
                    
                    probs = np.array(probs)
                    if probs.sum() == 0:
                        # 概率全0 (可能是因为R_lk限制)，随机选
                        chosen = np.random.choice(options)
                    else:
                        probs = probs / probs.sum()
                        chosen = np.random.choice(options, p=probs)
                    
                    assignment[k] = chosen
                
                cost = self.calculate_solution_cost(assignment)
                solutions.append(assignment)
                costs.append(cost)
                
                if cost < best_cost:
                    best_cost = cost
            
            history.append(best_cost)
            
            # 更新信息素
            # 1. 蒸发
            pheromone *= (1 - evaporation)
            
            # 2. 增强 (只取本轮最好的前几只，或者全局最好的，这里用简单的迭代最优)
            iter_best_idx = np.argmin(costs)
            iter_best_sol = solutions[iter_best_idx]
            iter_best_cost = costs[iter_best_idx]
            
            reward = 100.0 / iter_best_cost # 奖励常数
            for k, l in enumerate(iter_best_sol):
                if l != -1:
                    pheromone[k][l] += reward
                    
        print(f"ACO Best Cost: {best_cost:.2f}")
        return best_cost, history

if __name__ == "__main__":
    # 简单测试代码
    from Model import physical_model
    try:
        pm = physical_model.PhysicalModel(config.DEM_FILE, config.WIND_SPEED)
        solver = MetaheuristicSolvers(pm)
        solver.run_ga(generations=100)
        solver.run_pso(iterations=100)
        solver.run_aco(iterations=100)
    except Exception as e:
        print(e)