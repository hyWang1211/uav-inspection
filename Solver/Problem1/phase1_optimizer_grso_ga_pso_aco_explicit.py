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
        # final_assignment, active_nests, nest_loads = self._grso_pruning(final_assignment, active_nests, nest_loads)
        
        # 3. [新增] 负载再平衡 (微调)
        # final_assignment, final_nests, final_loads = self._grso_post_rebalance(final_assignment, active_nests, nest_loads)

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
    
    def _grso_post_rebalance(self, assignment, active_nests, loads):
        """
        Phase 3: 负载再平衡 (尝试移动风机以降低总成本)
        逻辑：遍历所有风机，看能否换个机巢，从而触发由于机巢的“降级”省钱。
        """
        curr_assign = assignment.copy()
        curr_loads = loads.copy()
        curr_active = set(active_nests)
        
        improved = True
        while improved:
            improved = False
            # 遍历所有已分配的风机
            for k, current_nest in list(curr_assign.items()):
                if current_nest == -1: continue
                
                # 计算如果移走 k，当前机巢能否降级省钱？
                current_load = curr_loads[current_nest]
                cost_current_old = self._get_required_type_and_cost(current_load)[1]
                cost_current_new = self._get_required_type_and_cost(current_load - 1)[1]
                saving = cost_current_old - cost_current_new
                
                if saving <= 0: continue # 移走也不能省钱（比如从 3个变2个，还是Type1），跳过
                
                # 寻找潜在的新东家
                best_target = -1
                max_net_saving = 0
                
                # 在其他激活机巢中找
                candidates = [l for l in curr_active if l != current_nest]
                # 按距离排序，优先找近的
                candidates.sort(key=lambda l: self.dist_matrix[l, k])
                
                for target in candidates:
                    # 必须物理可达
                    if self.R_lk[target, k] == 0: continue
                    
                    target_load = curr_loads[target]
                    # 必须有容量
                    if target_load >= self.global_capacity_limit: continue
                    
                    # 计算目标机巢升级成本
                    cost_target_old = self._get_required_type_and_cost(target_load)[1]
                    cost_target_new = self._get_required_type_and_cost(target_load + 1)[1]
                    cost_increase = cost_target_new - cost_target_old
                    
                    # 计算净收益
                    net_saving = saving - cost_increase
                    
                    if net_saving > 0.001: # 浮点数容差
                        # 找到了一个能省钱的跳槽方案！
                        # 贪婪：只要找到收益更大的就更新，或者找到第一个就走？这里找收益最大的
                        if net_saving > max_net_saving:
                            max_net_saving = net_saving
                            best_target = target
                
                # 执行跳槽
                if best_target != -1:
                    curr_assign[k] = best_target
                    curr_loads[current_nest] -= 1
                    curr_loads[best_target] += 1
                    improved = True
                    # 如果原机巢空了，移除它
                    if curr_loads[current_nest] == 0:
                        curr_active.remove(current_nest)
                        del curr_loads[current_nest]
                    break # 结构变了，重新开始循环以防数据不一致
                    
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
    
    # # =====================================================
    # #  Reverse Greedy (Benchmark from Reference Paper)
    # #  Logic: Start with ALL nests active -> Prune
    # # =====================================================
    # def run_reverse_greedy(self, history_length=200):
    #     print(f"--- Running Reverse Greedy (Destructive Benchmark) ---")
        
    #     # 1. 初始化：满配状态 (Full Deployment)
    #     # 假设所有候选点都激活，每个风机找最近的机巢
    #     active_nests = list(self.nest_ids)
    #     current_assignment = {}
    #     nest_loads = {l: 0 for l in active_nests}
        
    #     # 初始分配：最近距离原则
    #     for k in self.turbine_ids:
    #         # 找所有物理可达的机巢
    #         reachable = [l for l in active_nests if self.R_lk[l, k] == 1]
    #         if reachable:
    #             # 选最近的
    #             best_l = min(reachable, key=lambda l: self.dist_matrix[l, k])
    #             current_assignment[k] = best_l
    #             nest_loads[best_l] += 1
    #         else:
    #             current_assignment[k] = -1 # 无解

    #     # 2. 剪枝优化 (Pruning Loop)
    #     # 这里的逻辑和 G-RSO 的 Phase 2 类似，但起点不同
    #     # 文献逻辑：按“冗余度”排序，这里用“负载”作为冗余度的近似（负载越低越容易被替代）
        
    #     improved = True
    #     while improved:
    #         improved = False
    #         # 排序：优先尝试移除负载小的
    #         sorted_nests = sorted(list(active_nests), key=lambda l: nest_loads[l])
            
    #         for l_remove in sorted_nests:
    #             # 如果是空机巢，直接删
    #             if nest_loads[l_remove] == 0:
    #                 active_nests.remove(l_remove)
    #                 del nest_loads[l_remove]
    #                 improved = True
    #                 break
                
    #             # 尝试重新安置它的风机
    #             my_turbines = [k for k, v in current_assignment.items() if v == l_remove]
    #             can_reassign = True
    #             moves = {} # 记录迁移计划 k -> new_nest
                
    #             # 建立临时负载表
    #             temp_loads = nest_loads.copy()
    #             del temp_loads[l_remove]
                
    #             # 计算成本变动
    #             # 减项：移除 l_remove 省下的钱
    #             _, saved_cost = self._get_required_type_and_cost(nest_loads[l_remove])
    #             cost_delta = -saved_cost
                
    #             for k in my_turbines:
    #                 # 在剩余的 active_nests 里找新家
    #                 candidates = []
    #                 for l_target in active_nests:
    #                     if l_target == l_remove: continue
    #                     # 必须可达
    #                     if self.R_lk[l_target, k] == 1:
    #                         # 必须有容量
    #                         if temp_loads[l_target] < self.global_capacity_limit:
    #                             candidates.append(l_target)
                    
    #                 if not candidates:
    #                     can_reassign = False
    #                     break
                    
    #                 # 贪婪选择：选距离最近的邻居 (或者选升级成本最低的，这里用距离近似)
    #                 candidates.sort(key=lambda l: self.dist_matrix[l, k])
    #                 best_new = candidates[0]
    #                 moves[k] = best_new
                    
    #                 # 算加项：邻居升级多花的钱
    #                 old_c = self._get_required_type_and_cost(temp_loads[best_new])[1]
    #                 new_c = self._get_required_type_and_cost(temp_loads[best_new]+1)[1]
    #                 cost_delta += (new_c - old_c)
                    
    #                 temp_loads[best_new] += 1
                
    #             # 决策：如果可行且总成本降低了
    #             if can_reassign and cost_delta < 0:
    #                 active_nests.remove(l_remove)
    #                 nest_loads = temp_loads
    #                 for k, new_home in moves.items():
    #                     current_assignment[k] = new_home
    #                 improved = True
    #                 break # 重新开始循环

    #     # 3. 结果标准化 (Output Formatting)
    #     # 转换为显式数组
    #     explicit_nest_state = np.zeros(self.n_nests, dtype=int)
    #     explicit_assignment = np.zeros(self.n_turbines, dtype=int)
    #     total_cost = 0

    #     for l in active_nests:
    #         n = nest_loads[l]
    #         if n > 0:
    #             nest_type, cost = self._get_required_type_and_cost(n)
    #             explicit_nest_state[l] = nest_type
    #             total_cost += cost
        
    #     for k, l in current_assignment.items():
    #         explicit_assignment[k] = l

    #     print(f"Reverse Greedy Final Cost: {total_cost:.2f}")
        
    #     # 生成详情
    #     details = self.analyze_solution(explicit_nest_state, explicit_assignment)
        
    #     return {
    #         'fitness': total_cost,
    #         'history': [total_cost] * history_length, # 方便画图的平直线
    #         'solution': {
    #             'nests': explicit_nest_state, 
    #             'assignments': explicit_assignment
    #         },
    #         'details': details
    #     }


    # =====================================================
    #  Reverse Greedy (Benchmark from Reference Paper)
    #  Logic: Sort by Intersection Score (Redundancy)
    # =====================================================
    # def run_reverse_greedy(self, history_length=200):
    #     print(f"--- Running Reverse Greedy (Intersection-based Sorting) ---")
        
    #     # 0. 预处理：获取每个机巢的物理覆盖集合 (Reachable Set)
    #     # format: {nest_id: set(turbine_ids)}
    #     nest_reachability = {}
    #     for l in self.nest_ids:
    #         # 找出 R_lk 中该行等于 1 的风机索引
    #         reachable_indices = set(np.where(self.R_lk[l] == 1)[0])
    #         nest_reachability[l] = reachable_indices

    #     # 1. 初始化：满配状态 (Full Deployment)
    #     active_nests = list(self.nest_ids)
    #     current_assignment = {}
    #     nest_loads = {l: 0 for l in active_nests}
        
    #     # 初始分配：最近距离原则
    #     for k in self.turbine_ids:
    #         reachable = [l for l in active_nests if self.R_lk[l, k] == 1]
    #         if reachable:
    #             best_l = min(reachable, key=lambda l: self.dist_matrix[l, k])
    #             current_assignment[k] = best_l
    #             nest_loads[best_l] += 1
    #         else:
    #             current_assignment[k] = -1

    #     # 2. 剪枝优化 (Pruning Loop)
    #     improved = True
    #     while improved:
    #         improved = False
            
    #         # --- [核心修改] 计算重叠度评分 (Redundancy Score) ---
    #         # Score(i) = sum( |Reach(i) ∩ Reach(j)| ) for all j in Active\{i}
    #         # 含义：我覆盖的风机，有多少也被现在的队友覆盖了？分越高越冗余。
    #         redundancy_scores = {}
            
    #         for i in active_nests:
    #             score = 0
    #             my_set = nest_reachability[i]
    #             for j in active_nests:
    #                 if i == j: continue
    #                 # 计算交集大小
    #                 intersection_size = len(my_set.intersection(nest_reachability[j]))
    #                 score += intersection_size
    #             redundancy_scores[i] = score
            
    #         # --- 排序：重叠度从大到小 (High Redundancy First) ---
    #         # 如果重叠度相同，再按负载从小到大作为次要排序键
    #         sorted_nests = sorted(
    #             active_nests, 
    #             key=lambda l: (redundancy_scores[l], -nest_loads[l]), 
    #             reverse=True
    #         )
            
    #         # 3. 尝试移除
    #         for l_remove in sorted_nests:
    #             # 空机巢直接删
    #             if nest_loads[l_remove] == 0:
    #                 active_nests.remove(l_remove)
    #                 del nest_loads[l_remove]
    #                 improved = True
    #                 break
                
    #             # 尝试重新安置
    #             my_turbines = [k for k, v in current_assignment.items() if v == l_remove]
    #             can_reassign = True
    #             moves = {} 
    #             temp_loads = nest_loads.copy()
    #             del temp_loads[l_remove]
                
    #             # 成本计算
    #             _, saved_cost = self._get_required_type_and_cost(nest_loads[l_remove])
    #             cost_delta = -saved_cost
                
    #             for k in my_turbines:
    #                 candidates = []
    #                 for l_target in active_nests:
    #                     if l_target == l_remove: continue
    #                     # 必须可达
    #                     if self.R_lk[l_target, k] == 1:
    #                         # 必须有容量
    #                         if temp_loads[l_target] < self.global_capacity_limit:
    #                             candidates.append(l_target)
                    
    #                 if not candidates:
    #                     can_reassign = False
    #                     break
                    
    #                 # 选最近的邻居
    #                 candidates.sort(key=lambda l: self.dist_matrix[l, k])
    #                 best_new = candidates[0]
    #                 moves[k] = best_new
                    
    #                 # 算升级成本
    #                 old_c = self._get_required_type_and_cost(temp_loads[best_new])[1]
    #                 new_c = self._get_required_type_and_cost(temp_loads[best_new]+1)[1]
    #                 cost_delta += (new_c - old_c)
                    
    #                 temp_loads[best_new] += 1
                
    #             # 决策
    #             if can_reassign and cost_delta < 0:
    #                 active_nests.remove(l_remove)
    #                 nest_loads = temp_loads
    #                 for k, new_home in moves.items():
    #                     current_assignment[k] = new_home
    #                 improved = True
    #                 print(f"  RG: Pruned nest {l_remove} (Score: {redundancy_scores[l_remove]})")
    #                 break 

    #     # 4. 结果格式化
    #     explicit_nest_state = np.zeros(self.n_nests, dtype=int)
    #     explicit_assignment = np.zeros(self.n_turbines, dtype=int)
    #     total_cost = 0

    #     for l in active_nests:
    #         n = nest_loads[l]
    #         if n > 0:
    #             nest_type, cost = self._get_required_type_and_cost(n)
    #             explicit_nest_state[l] = nest_type
    #             total_cost += cost
        
    #     for k, l in current_assignment.items():
    #         explicit_assignment[k] = l

    #     print(f"Reverse Greedy Final Cost: {total_cost:.2f}")
        
    #     return {
    #         'fitness': total_cost,
    #         'history': [total_cost] * history_length,
    #         'solution': {
    #             'nests': explicit_nest_state, 
    #             'assignments': explicit_assignment
    #         },
    #         'details': self.analyze_solution(explicit_nest_state, explicit_assignment)
    #     }
    # =====================================================
    #  Reverse Greedy (Benchmark from Reference Paper)
    #  Logic: Sort by Redundancy Score (Intersection), then Prune
    # =====================================================
    def run_reverse_greedy(self, history_length=200):
        print(f"--- Running Reverse Greedy (Intersection-based Pruning) ---")
        
        # 1. 初始化：满配状态 (Full Deployment)
        active_nests = list(self.nest_ids)
        current_assignment = {}
        nest_loads = {l: 0 for l in active_nests}
        
        # 初始分配：最近距离原则
        for k in self.turbine_ids:
            reachable = [l for l in active_nests if self.R_lk[l, k] == 1]
            if reachable:
                best_l = min(reachable, key=lambda l: self.dist_matrix[l, k])
                current_assignment[k] = best_l
                nest_loads[best_l] += 1
            else:
                current_assignment[k] = -1 # 未分配

        # 预处理：计算所有机巢的“可达风机集合”
        nest_reachability = {}
        for l in self.nest_ids:
            nest_reachability[l] = set(np.where(self.R_lk[l] == 1)[0])

        # 2. 迭代剪枝 (Pruning Loop)
        improved = True
        while improved:
            improved = False
            
            # --- [核心修改 1] 排序依据：重叠度 (Redundancy Score) ---
            # Score(i) = sum( |Reach(i) ∩ Reach(j)| ) for all j in Active\{i}
            # 含义：我覆盖的风机，有多少也被别人覆盖了？分数越高越冗余。
            
            redundancy_scores = {}
            for i in active_nests:
                score = 0
                my_set = nest_reachability[i]
                for j in active_nests:
                    if i == j: continue
                    intersection_size = len(my_set.intersection(nest_reachability[j]))
                    score += intersection_size
                redundancy_scores[i] = score
            
            # --- 排序：重叠度从大到小 (High Redundancy First) ---
            # (文献中没有提到负载作为次要排序键，仅用重叠度)
            sorted_nests = sorted(
                active_nests, 
                key=lambda l: redundancy_scores.get(l, 0), # get(l, 0) 防止 l 不在 dict 中
                reverse=True 
            )
            
            # 3. 尝试移除
            for l_remove in sorted_nests:
                # 如果该机巢没有风机（负载为0），直接删除
                if nest_loads.get(l_remove, 0) == 0:
                    if l_remove in active_nests:
                        active_nests.remove(l_remove)
                        if l_remove in nest_loads: del nest_loads[l_remove]
                        improved = True
                        print(f"  RG: Pruned empty nest {l_remove}")
                        break # 结构变化，重新开始循环
                    continue

                # --- [核心修改 2] 移除决策：只需判断“能否移除” ---
                # 逻辑：只要能移除，就立刻移除，不考虑成本！
                # 目标是最小化数量，而不是成本。
                
                my_turbines = [k for k, v in current_assignment.items() if v == l_remove]
                can_reassign = True
                moves = {} # 记录迁移计划 k -> new_nest
                
                # 检查这些风机是否能被其他机巢接收
                temp_loads = nest_loads.copy()
                if l_remove in temp_loads: del temp_loads[l_remove] # 模拟删除
                
                for k in my_turbines:
                    candidates = []
                    for l_target in active_nests:
                        if l_target == l_remove: continue # 跳过被删除的机巢
                        # 必须可达
                        if self.R_lk[l_target, k] == 1:
                            # 必须有容量 (使用 global_capacity_limit)
                            if temp_loads.get(l_target, 0) < self.global_capacity_limit: # get() 避免 KeyError
                                candidates.append(l_target)
                    
                    if not candidates:
                        can_reassign = False
                        break
                    
                    # 贪婪选择：选最近的
                    candidates.sort(key=lambda l: self.dist_matrix[l, k])
                    best_new = candidates[0]
                    moves[k] = best_new
                    temp_loads[best_new] += 1
                
                # 决策：只要能移除，就执行
                if can_reassign:
                    active_nests.remove(l_remove)
                    nest_loads = temp_loads
                    for k, new_home in moves.items():
                        current_assignment[k] = new_home
                    
                    improved = True
                    print(f"  RG: Pruned nest {l_remove} (Redundancy Score: {redundancy_scores.get(l_remove, 0)})")
                    break # 结构变了，重新开始循环
        
        # 4. 结果格式化
        explicit_nest_state = np.zeros(self.n_nests, dtype=int)
        explicit_assignment = np.zeros(self.n_turbines, dtype=int)
        total_cost = 0

        for l in active_nests:
            n = nest_loads.get(l, 0) # 使用 get() 避免 KeyError
            if n > 0:
                nest_type, cost = self._get_required_type_and_cost(n)
                explicit_nest_state[l] = nest_type
                total_cost += cost
        
        for k, l in current_assignment.items():
            explicit_assignment[k] = l

        print(f"Reverse Greedy Final Cost: {total_cost:.2f}")
        
        # 返回统一格式
        return {
            'fitness': total_cost,
            'history': [total_cost] * history_length, # 填充历史数据
            'solution': {
                'nests': explicit_nest_state, 
                'assignments': explicit_assignment
            },
            'details': self.analyze_solution(explicit_nest_state, explicit_assignment)
        }