import numpy as np
import random
import math
import pandas as pd
from Model import config

class ExplicitMetaheuristicSolvers:
    def __init__(self, physical_model):
        self.pm = physical_model
        self.nest_locations_df = physical_model.nest_locations
        self.turbines_df = physical_model.wind_turbines_df

        self.n_nests = len(physical_model.nest_locations)
        self.n_turbines = len(physical_model.wind_turbines_df)
        self.nest_ids = list(range(self.n_nests))
        self.turbine_ids = list(range(self.n_turbines))
        
        # 预计算 R_lk 矩阵
        self.R_lk = self.pm.calculate_reachability_matrix(self.pm.nest_locations, self.pm.wind_turbines_df)
        
        # 成本参数
        self.base_costs = config.NEST_BASE_COSTS
        self.uav_cost = config.OMEGA_2
        self.eta = config.ETA
        self.max_type = max(self.base_costs.keys()) # 通常是 4
        self.global_capacity_limit = self.max_type * self.eta

        # 惩罚系数 (Big M)
        self.PENALTY_UNREACHABLE = 1e1  # 物理不可达
        self.PENALTY_NO_NEST = 1e1      # 指派给了没建机巢的点
        self.PENALTY_OVERCAPACITY = 1e1 # 超载
        self.PENALTY_UNASSIGNED = 1e1   # 风机未分配 (在PSO/GA中可能出现无效索引)

        # 预计算距离矩阵
        self.dist_matrix = self._calculate_distance_matrix()

    # ====================================================
    # 辅助：计算距离矩阵
    # ====================================================
    def _calculate_distance_matrix(self):
        dists = np.full((self.n_nests, self.n_turbines), 1e9)
        nest_coords = self.nest_locations_df[['lon', 'lat']].values
        turb_coords = self.turbines_df[['lon', 'lat']].values
        
        for l in range(self.n_nests):
            for k in range(self.n_turbines):
                if self.R_lk[l, k] == 1:
                    dx = nest_coords[l][0] - turb_coords[k][0]
                    dy = nest_coords[l][1] - turb_coords[k][1]
                    dists[l, k] = np.sqrt(dx**2 + dy**2)
        return dists

    # =====================================================
    # 辅助：根据风机数量计算机巢类型和成本
    # =====================================================
    def _get_required_type_and_cost(self, num_turbines):
        """根据风机数量计算机巢类型和成本"""
        if num_turbines == 0: return 0, 0
        n_uavs = math.ceil(num_turbines / self.eta)
        nest_type = n_uavs
        if nest_type > self.max_type: nest_type = self.max_type
        if nest_type < 1: nest_type = 1
        cost = self.base_costs[nest_type] + nest_type * self.uav_cost
        return nest_type, cost

    # =====================================================
    # 核心：适应度函数 (Cost + Penalty)[For GA/PSO/ACO]
    # =====================================================
    def calculate_fitness(self, nest_state, assignment):
        """
        Input:
            nest_state:  长度为 N 的数组, 值在 [0, 1, 2, 3, 4]. 0代表不建, 1-4代表类型.
            assignment:  长度为 K 的数组, 值在 [0, N-1]. 代表风机 k 指派给机巢 l.
        """
        total_cost = 0
        penalty = 0
        
        # 1. 计算建设成本 (对应目标函数 M1.1)
        # 只要 nest_state[l] > 0，就产生了成本
        for l in range(self.n_nests):
            tau = int(nest_state[l])
            if tau > 0:
                cost = self.base_costs[tau] + tau * self.uav_cost
                total_cost += cost
                
        # 2. 检查指派相关的约束与惩罚
        # 统计实际负载用来查 Capacity 约束
        real_loads = np.zeros(self.n_nests, dtype=int)
        
        for k in range(self.n_turbines):
            l = int(assignment[k])
            
            # 范围检查 (防止 PSO 越界)
            if l < 0 or l >= self.n_nests:
                penalty += self.PENALTY_UNASSIGNED
                continue
                
            # 约束 (4): 物理可达性 (M1.5)
            if self.R_lk[l, k] == 0:
                penalty += self.PENALTY_UNREACHABLE
            
            # 约束 (3): 建设依托 (M1.4) - 必须指派给已建机巢
            if nest_state[l] == 0:
                penalty += self.PENALTY_NO_NEST
            else:
                real_loads[l] += 1
                
        # 3. 检查容量约束 (M1.6)
        for l in range(self.n_nests):
            tau = int(nest_state[l])
            if tau > 0:
                max_capacity = tau * self.eta
                if real_loads[l] > max_capacity:
                    # 超载惩罚：每超一个都罚
                    over_count = real_loads[l] - max_capacity
                    penalty += over_count * self.PENALTY_OVERCAPACITY
                    
        return total_cost + penalty
    
    # =====================================================
    # 补全缺失的方法：纯成本计算 (无惩罚)
    # =====================================================
    # def calculate_solution_cost(self, assignment):
    #     """
    #     根据指派结果计算纯经济成本。
    #     逻辑：统计每个机巢的负载 -> 推导所需类型 -> 查表求和。
    #     """
    #     # 1. 统计每个机巢的负载
    #     nest_loads = {}
    #     for l in assignment:
    #         l = int(l)
    #         if l == -1: continue # 未指派
    #         nest_loads[l] = nest_loads.get(l, 0) + 1
        
    #     total_cost = 0
        
    #     # 2. 计算每个激活机巢的费用
    #     for l, load in nest_loads.items():
    #         if load == 0: continue
            
    #         # 根据负载推导类型 (和 G-RSO 逻辑一致)
    #         # 例如: load=5, eta=4 -> n_uavs=2 -> Type 2
    #         n_uavs = math.ceil(load / self.eta)
    #         tau = n_uavs
            
    #         # 边界限制
    #         if tau > self.max_type: tau = self.max_type
    #         if tau < 1: tau = 1
            
    #         # 费用公式: 基建 + 无人机
    #         cost = self.base_costs[tau] + tau * self.uav_cost
    #         total_cost += cost
            
    #     return total_cost

    # =====================================================
    # 新增：详细分析函数 (用于最后生成报告)
    # =====================================================
    def analyze_solution(self, nest_state, assignment):
        """
        对给定的解进行详细“体检”，返回具体的违规统计和成本结构。
        """
        analysis = {
            'pure_cost': 0,         # 纯经济成本 (无惩罚)
            'total_penalty': 0,     # 总惩罚值
            'total_nests_built': 0,    # 建设的机巢总数
            'total_uavs_deployed': 0,  # 部署的无人机总数
            'violations': {
                'unreachable': 0,   # 物理不可达数量
                'ghost_nest': 0,    # 指派给未建机巢数量
                'overcapacity': 0,  # 超载的风机数量
                'unassigned': 0     # 未指派/越界数量
            },
            'nest_counts': {1:0, 2:0, 3:0, 4:0}, # 各类型机巢建设数量
            'is_feasible': True     # 是否完全可行
        }

        # 1. 统计建设成本和机巢数量
        for l in range(self.n_nests):
            tau = int(nest_state[l])
            if tau > 0:
                analysis['pure_cost'] += self.base_costs[tau] + tau * self.uav_cost
                analysis['nest_counts'][tau] = analysis['nest_counts'].get(tau, 0) + 1
                analysis['total_nests_built'] += 1      # 机巢数 +1
                analysis['total_uavs_deployed'] += tau  # 无人机数 + 类型 (Type 4 = 4架)

        # 2. 统计指派违规
        real_loads = np.zeros(self.n_nests, dtype=int)
        
        for k in range(self.n_turbines):
            l = int(assignment[k])
            
            if l < 0 or l >= self.n_nests:
                analysis['violations']['unassigned'] += 1
                analysis['total_penalty'] += self.PENALTY_UNASSIGNED
                continue
            
            if self.R_lk[l, k] == 0:
                analysis['violations']['unreachable'] += 1
                analysis['total_penalty'] += self.PENALTY_UNREACHABLE
            
            if nest_state[l] == 0:
                analysis['violations']['ghost_nest'] += 1
                analysis['total_penalty'] += self.PENALTY_NO_NEST
            else:
                real_loads[l] += 1
        
        # 3. 统计容量违规
        for l in range(self.n_nests):
            tau = int(nest_state[l])
            if tau > 0:
                max_cap = tau * self.eta
                if real_loads[l] > max_cap:
                    over = real_loads[l] - max_cap
                    analysis['violations']['overcapacity'] += over
                    analysis['total_penalty'] += over * self.PENALTY_OVERCAPACITY
        
        # 4. 判断总体可行性
        total_viols = sum(analysis['violations'].values())
        if total_viols > 0:
            analysis['is_feasible'] = False
            
        return analysis

    # =====================================================
    #  G-RSO Algorithm (Integrated)
    # =====================================================
    def run_grso(self, history_length=200):
        """运行 G-RSO 算法并返回标准化的结果格式"""
        print(f"\n--- Running G-RSO (Locked V4) ---")
        
        # 1. 贪婪构造 (Lock-in)
        final_assignment, active_nests, nest_loads = self._grso_greedy_construction()
        
        # 2. 剪枝优化
        final_assignment, final_nests, final_loads = self._grso_pruning(final_assignment, active_nests, nest_loads)
        
        # 3. 转换结果为 DataFrame (保持与原G-RSO输出一致，方便绘图)
        # selected_nests_data = []
        total_cost = 0
        
        # 为了适配 GA/PSO 格式，同时也生成显式数组
        explicit_nest_state = np.zeros(self.n_nests, dtype=int)
        explicit_assignment = np.zeros(self.n_turbines, dtype=int)

        for l in final_nests:
            n = final_loads[l]
            nest_type, cost = self._get_required_type_and_cost(n)
            # selected_nests_data.append({
            #     'nest_id': l, 'capacity': nest_type, 'load': n
            # })
            total_cost += cost
            explicit_nest_state[l] = nest_type

        for k, l in final_assignment.items():
            if l != -1:
                explicit_assignment[k] = l
            else:
                explicit_assignment[k] = -1 # 虽然G-RSO保证分配，但防守一下

        # 生成 DataFrame (供 main.py 绘图用)
        # df_nests = pd.DataFrame(selected_nests_data)
        # assignments_list = [{'turbine_id': k, 'nest_id': l} for k, l in final_assignment.items() if l != -1]
        # df_assigns = pd.DataFrame(assignments_list)

        print(f"G-RSO Final Cost: {total_cost:.2f}")
        # 4. 生成统一的详情报告
        details = self.analyze_solution(explicit_nest_state, explicit_assignment)

        # 5. 生成对齐的 history 用于画图
        # G-RSO 是非迭代的，我们用常数填充列表
        history = [total_cost] * history_length
        
        return {
            'fitness': total_cost,
            'history': history,
            'solution': {
                'nests': explicit_nest_state, 
                'assignments': explicit_assignment
            },
            'details': details
        }

    def _grso_greedy_construction(self):
        uncovered = set(self.turbine_ids)
        final_assign = {}
        nest_loads = {}
        
        while uncovered:
            best_score = float('inf')
            best_nest = -1
            best_turbines = []
            
            candidates = [l for l in self.nest_ids if l not in nest_loads]
            if not candidates: break
            
            for l in candidates:
                reachable = np.where(self.R_lk[l] == 1)[0]
                potential = [k for k in reachable if k in uncovered]
                if not potential: continue
                
                potential.sort(key=lambda k: self.dist_matrix[l, k])
                real_take = potential[:self.global_capacity_limit]
                gain = len(real_take)
                
                _, cost = self._get_required_type_and_cost(gain)
                score = cost / gain
                if score < best_score:
                    best_score = score
                    best_nest = l
                    best_turbines = real_take
            
            if best_nest != -1:
                nest_loads[best_nest] = len(best_turbines)
                for k in best_turbines:
                    final_assign[k] = best_nest
                    uncovered.remove(k)
            else:
                break
        return final_assign, list(nest_loads.keys()), nest_loads

    def _grso_pruning(self, assignment, active_nests, loads):
        curr_assign = assignment.copy()
        curr_active = set(active_nests)
        curr_loads = loads.copy()
        
        improved = True
        while improved:
            improved = False
            sorted_nests = sorted(list(curr_active), key=lambda l: curr_loads[l])
            
            for l_remove in sorted_nests:
                my_turbines = [k for k, v in curr_assign.items() if v == l_remove]
                can_reassign = True
                moves = {}
                temp_loads = curr_loads.copy()
                del temp_loads[l_remove]
                
                _, saved = self._get_required_type_and_cost(curr_loads[l_remove])
                cost_delta = -saved
                
                for k in my_turbines:
                    candidates = []
                    for l_target in curr_active:
                        if l_target == l_remove: continue
                        if self.R_lk[l_target, k] == 1:
                            if temp_loads[l_target] < self.global_capacity_limit:
                                candidates.append(l_target)
                    if not candidates:
                        can_reassign = False
                        break
                    
                    candidates.sort(key=lambda l: self.dist_matrix[l, k])
                    best_new = candidates[0]
                    moves[k] = best_new
                    
                    old_c = self._get_required_type_and_cost(temp_loads[best_new])[1]
                    new_c = self._get_required_type_and_cost(temp_loads[best_new]+1)[1]
                    cost_delta += (new_c - old_c)
                    temp_loads[best_new] += 1
                
                if can_reassign and cost_delta < 0:
                    curr_active.remove(l_remove)
                    curr_loads = temp_loads
                    for k, new_home in moves.items():
                        curr_assign[k] = new_home
                    improved = True
                    break
        return curr_assign, list(curr_active), curr_loads


    # ==========================================
    # 1. Explicit Genetic Algorithm (GA)
    # ==========================================
    def run_ga(self, pop_size=50, generations=100, crossover_rate=0.8, mutation_rate=0.8):
        print(f"--- Running Explicit GA (Pop: {pop_size}, Gen: {generations}) ---")
        
        # 基因编码：[Nest_States (N) | Assignments (K)]
        # 前 N 位是机巢类型(0-4)，后 K 位是指派ID(0 ~ N-1)
        genome_len = self.n_nests + self.n_turbines
        
        # 初始化
        population = []
        for _ in range(pop_size):
            # 随机生成机巢状态 (偏向于生成 0，因为大部分地方不建)
            # nests = np.random.choice([0, 1, 2, 3, 4], size=self.n_nests, p=[0.8, 0.05, 0.05, 0.05, 0.05])
            nests = np.random.choice([0, 1, 2, 3, 4], size=self.n_nests)
            # 随机生成指派
            assigns = np.random.randint(0, self.n_nests, size=self.n_turbines)
            population.append(np.concatenate([nests, assigns]))
        
        best_fitness = float('inf')
        best_genome = None  # 新增：记录最佳基因
        history = []
        
        for gen in range(generations):
            fitnesses = []
            for ind in population:
                n_part = ind[:self.n_nests]
                a_part = ind[self.n_nests:]
                fit = self.calculate_fitness(n_part, a_part)
                fitnesses.append(fit)
                
            # 记录最佳
            min_idx = np.argmin(fitnesses)
            if fitnesses[min_idx] < best_fitness:
                best_fitness = fitnesses[min_idx]
                best_genome = population[min_idx].copy() # 锁定最佳解
            history.append(best_fitness)
            
            # 锦标赛选择
            new_pop = []
            for _ in range(pop_size):
                i1, i2 = np.random.randint(0, pop_size, 2)
                winner = population[i1] if fitnesses[i1] < fitnesses[i2] else population[i2]
                new_pop.append(winner.copy())
            
            # 交叉
            for i in range(0, pop_size, 2):
                if random.random() < crossover_rate: #交叉概率默认为0.8
                    pt = random.randint(1, genome_len - 1)
                    new_pop[i][pt:], new_pop[i+1][pt:] = new_pop[i+1][pt:].copy(), new_pop[i][pt:].copy()
            
            # 变异
            for i in range(pop_size):
                if random.random() < mutation_rate:
                    # 随机选一个基因位突变
                    idx = random.randint(0, genome_len - 1)
                    if idx < self.n_nests:
                        # 突变机巢状态 (0-4)
                        new_pop[i][idx] = random.choice([0, 1, 2, 3, 4])
                    else:
                        # 突变指派 (0 ~ N-1)
                        new_pop[i][idx] = random.randint(0, self.n_nests - 1)
            
            population = new_pop

        # --- 最终分析 ---
        print(f"Explicit GA Best Fitness: {best_fitness:.2f}")
        # 解码最佳解
        best_nests = best_genome[:self.n_nests]
        best_assigns = best_genome[self.n_nests:]
        details = self.analyze_solution(best_nests, best_assigns)
        
        return {
            'fitness': best_fitness, 
            'history': history, 
            'solution': {'nests': best_nests, 'assignments': best_assigns}, 
            'details': details
        }

    # ==========================================
    # 2. Explicit Particle Swarm Optimization (PSO)
    # ==========================================
    def run_pso(self, swarm_size=50, iterations=100):
        print(f"--- Running Explicit PSO (Swarm: {swarm_size}, Iter: {iterations}) ---")
        
        dim = self.n_nests + self.n_turbines
        
        # 初始化粒子 (连续值)
        # Nests: [0, 4.99], Assignments: [0, N-0.01]
        X = np.zeros((swarm_size, dim))
        X[:, :self.n_nests] = np.random.uniform(0, 5, (swarm_size, self.n_nests))
        X[:, self.n_nests:] = np.random.uniform(0, self.n_nests, (swarm_size, self.n_turbines))
        
        V = np.random.uniform(-1, 1, (swarm_size, dim))
        
        P_best = X.copy()
        P_best_scores = np.full(swarm_size, float('inf'))
        G_best = X[0].copy()
        G_best_score = float('inf')
        
        history = []
        
        w, c1, c2 = 0.7, 1.5, 1.5
        
        for it in range(iterations):
            for i in range(swarm_size):
                # 解码：四舍五入
                int_vec = np.round(X[i]).astype(int)
                
                # 边界修正
                n_part = np.clip(int_vec[:self.n_nests], 0, 4)
                a_part = np.clip(int_vec[self.n_nests:], 0, self.n_nests - 1)
                
                fit = self.calculate_fitness(n_part, a_part)
                
                if fit < P_best_scores[i]:
                    P_best_scores[i] = fit
                    P_best[i] = X[i].copy()
                    
                if fit < G_best_score:
                    G_best_score = fit
                    G_best = X[i].copy()
            
            history.append(G_best_score)
            
            # 更新
            r1 = np.random.rand(swarm_size, dim)
            r2 = np.random.rand(swarm_size, dim)
            V = w*V + c1*r1*(P_best - X) + c2*r2*(G_best - X)
            X = X + V
            
            # 边界限制
            X[:, :self.n_nests] = np.clip(X[:, :self.n_nests], 0, 4.99)
            X[:, self.n_nests:] = np.clip(X[:, self.n_nests:], 0, self.n_nests - 0.01)
        
        # --- 最终分析 ---
        print(f"Explicit PSO Best Fitness: {G_best_score:.2f}")

        # 对全局最优解进行解码
        final_int_vec = np.round(G_best).astype(int)
        best_nests = np.clip(final_int_vec[:self.n_nests], 0, 4)
        best_assigns = np.clip(final_int_vec[self.n_nests:], 0, self.n_nests - 1)
        details = self.analyze_solution(best_nests, best_assigns)

        return {
            'fitness': G_best_score, 
            'history': history, 
            'solution': {'nests': best_nests, 'assignments': best_assigns}, 
            'details': details
        }

    # ==========================================
    # 3. Explicit Ant Colony Optimization (ACO)
    # ==========================================
    def run_aco(self, n_ants=30, iterations=50, evaporation=0.1):
        """
        双层蚁群：
        Step 1: 蚂蚁走过 N 个节点，决定每个节点的 Type (0-4)
        Step 2: 蚂蚁走过 K 个节点，决定每个风机去哪个机巢 (0 ~ N-1)
        """
        print(f"--- Running Explicit ACO (Ants: {n_ants}, Iter: {iterations}) ---")
        
        # 信息素矩阵 1: 选址决策 [N_nests, 5] (Type 0-4)
        phero_nest = np.ones((self.n_nests, 5)) * 1.0
        
        # 信息素矩阵 2: 指派决策 [N_turbines, N_nests]
        phero_assign = np.ones((self.n_turbines, self.n_nests)) * 1.0
        
        best_fitness = float('inf')
        best_solution = (None, None) # 记录最佳解 (nest_state, assignment)
        history = []
        
        for it in range(iterations):
            solutions = []
            fitnesses = []
            
            for ant in range(n_ants):
                # Phase A: 决定机巢状态
                nest_state = np.zeros(self.n_nests, dtype=int)
                for l in range(self.n_nests):
                    probs = phero_nest[l] / phero_nest[l].sum()
                    nest_state[l] = np.random.choice([0, 1, 2, 3, 4], p=probs)
                
                # Phase B: 决定指派
                # 启发式：优先指派给刚才 nest_state > 0 的点
                assignment = np.zeros(self.n_turbines, dtype=int)
                active_nests = np.where(nest_state > 0)[0]
                
                for k in range(self.n_turbines):
                    # 如果有建成的机巢，稍微偏向它们，否则全盲搜
                    # 这里为了简化，直接用轮盘赌
                    probs = phero_assign[k] / phero_assign[k].sum()
                    assignment[k] = np.random.choice(range(self.n_nests), p=probs)
                
                fit = self.calculate_fitness(nest_state, assignment)
                solutions.append((nest_state, assignment))
                fitnesses.append(fit)
                
                if fit < best_fitness:
                    best_fitness = fit
                    best_solution = (nest_state.copy(), assignment.copy()) # 锁定最佳
            
            history.append(best_fitness)
            
            # 更新信息素
            phero_nest *= (1 - evaporation)
            phero_assign *= (1 - evaporation)
            
            # 只有优质解增加信息素 (避免被大量不可行解污染)
            # 选取本轮前 10% 好的
            sorted_indices = np.argsort(fitnesses)
            n_elites = max(1, int(n_ants * 0.1))
            
            for i in range(n_elites):
                idx = sorted_indices[i]
                n_sol, a_sol = solutions[idx]
                fit = fitnesses[idx]
                
                # 简单的奖励函数
                reward = 1.0 / (fit + 1e-6)
                if fit > 1e5: reward = 0 # 垃圾解不奖励
                
                for l in range(self.n_nests):
                    phero_nest[l][n_sol[l]] += reward
                for k in range(self.n_turbines):
                    phero_assign[k][a_sol[k]] += reward
        
        # --- 最终分析 ---
        print(f"ACO Best Fitness: {best_fitness:.2f}")
        best_nests, best_assigns = best_solution
        if best_nests is None: # 防御性编程
             best_nests = np.zeros(self.n_nests, dtype=int)
             best_assigns = np.zeros(self.n_turbines, dtype=int)

        details = self.analyze_solution(best_nests, best_assigns)

        return {
            'fitness': best_fitness, 
            'history': history, 
            'solution': {'nests': best_nests, 'assignments': best_assigns}, 
            'details': details
        }
    
    # ==========================================
    # 4. Reverse Greedy (Destructive Heuristic)
    # 基于文献 [X] 的反向剔除策略
    # ==========================================
    # def run_reverse_greedy(self):
    #     print(f"--- Running Reverse Greedy (Destructive) ---")
        
    #     # 1. 初始化：假设每个风机位置都建一个机巢
    #     # 注意：这里我们只能用候选点中与风机重合的那些，或者最接近的那些
    #     # 为了简化，我们假设初始激活所有候选机巢 (Active = All)
    #     # 或者更严格按照论文：只激活那些位置上有风机的候选点
        
    #     # 策略：初始激活所有候选机巢，并将每个风机指派给最近的机巢
    #     active_nests = list(range(self.n_nests))
        
    #     # 初始指派
    #     assignment = {}
    #     nest_loads = {l: 0 for l in active_nests}
        
    #     # 简单的初始分配：每个风机找最近的
    #     dists = np.full((self.n_nests, self.n_turbines), 1e9)
    #     nest_coords = self.pm.nest_locations[['lon', 'lat']].values
    #     turb_coords = self.pm.wind_turbines_df[['lon', 'lat']].values
    #     for l in range(self.n_nests):
    #         for k in range(self.n_turbines):
    #             if self.R_lk[l, k] == 1:
    #                 dists[l, k] = np.linalg.norm(nest_coords[l] - turb_coords[k])
        
    #     for k in range(self.n_turbines):
    #         # 找最近的可行机巢
    #         feasible = [l for l in active_nests if self.R_lk[l, k] == 1]
    #         if feasible:
    #             best_l = min(feasible, key=lambda l: dists[l, k])
    #             assignment[k] = best_l
    #             nest_loads[best_l] += 1
    #         else:
    #             assignment[k] = -1 # 无解情况

    #     # 2. 迭代剔除 (Pruning Loop)
    #     improved = True
    #     while improved:
    #         improved = False
            
    #         # 排序：优先尝试移除负载最小的机巢 (最可能是冗余的)
    #         # 或者是那是论文里的逻辑：按“重叠度”排序。这里用负载近似，效果一样且更快。
    #         sorted_nests = sorted(active_nests, key=lambda l: nest_loads[l])
            
    #         for l_remove in sorted_nests:
    #             # 如果这个机巢本来就是空的，直接删
    #             if nest_loads[l_remove] == 0:
    #                 active_nests.remove(l_remove)
    #                 del nest_loads[l_remove]
    #                 improved = True
    #                 break
                
    #             # 尝试移除 l_remove
    #             my_turbines = [k for k, v in assignment.items() if v == l_remove]
                
    #             # 检查这些风机能否被【其他激活的机巢】接收
    #             # 约束：物理可达 + 容量限制 (Max Type * Eta)
    #             can_reassign = True
    #             moves = {} # k -> new_l
                
    #             # 创建临时负载表
    #             temp_loads = nest_loads.copy()
    #             del temp_loads[l_remove]
                
    #             max_cap = self.max_type * self.eta # e.g., 16
                
    #             for k in my_turbines:
    #                 # 在剩余的 active_nests 里找
    #                 options = [l for l in active_nests if l != l_remove and self.R_lk[l, k] == 1]
    #                 # 按距离排序
    #                 options.sort(key=lambda l: dists[l, k])
                    
    #                 found = False
    #                 for l_target in options:
    #                     if temp_loads[l_target] < max_cap:
    #                         temp_loads[l_target] += 1
    #                         moves[k] = l_target
    #                         found = True
    #                         break
                    
    #                 if not found:
    #                     can_reassign = False
    #                     break
                
    #             # 决策逻辑：Reverse Greedy 通常只看“能不能删”，只要能删就删，为了最小化数量
    #             # 但为了公平对比成本，我们最好也检查一下 Cost 是否降低
    #             # 那篇论文的目标是 Min Number，你的目标是 Min Cost
    #             # 这里我们采用它的逻辑：Min Number (优先删)，还是你的逻辑？
    #             # 建议：采用混合逻辑 —— 只要能删且不违反约束就删，因为减少机巢数量通常也能降低成本
                
    #             if can_reassign:
    #                 # 执行移除
    #                 active_nests.remove(l_remove)
    #                 nest_loads = temp_loads
    #                 for k, new_l in moves.items():
    #                     assignment[k] = new_l
                    
    #                 print(f"  Reverse Greedy: Removed nest {l_remove}")
    #                 improved = True
    #                 break # 重新开始循环
        
    #     # 3. 计算最终成本
    #     # 构造 assignment array 格式以复用计算函数
    #     final_assignment = np.zeros(self.n_turbines, dtype=int)
    #     for k, v in assignment.items():
    #         final_assignment[k] = v
            
    #     total_cost = self.calculate_solution_cost(final_assignment)
    #     print(f"Reverse Greedy Best Cost: {total_cost:.2f}")
        
    #     # 这里的 history 没意义，因为不是迭代优化，只返回最终值，为了画图方便，返回一条直线
    #     return total_cost, [total_cost] * 200