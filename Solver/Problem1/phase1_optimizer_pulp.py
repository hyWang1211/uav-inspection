# phase1_optimizer.py

import pandas as pd
import numpy as np
import pulp # 优化库
from Model import config # 导入全局配置

class Phase1Optimizer:
    def __init__(self, physical_model):
        self.physical_model = physical_model
        self.nest_locations_df = physical_model.nest_locations 
        self.turbines_df = physical_model.wind_turbines_df     

        if self.nest_locations_df.empty or self.turbines_df.empty:
            raise ValueError("Nest locations or wind turbine data is missing.")

        self.n_nests = len(self.nest_locations_df)
        self.n_turbines = len(self.turbines_df)
        self.nest_ids = list(range(self.n_nests))
        self.turbine_ids = list(range(self.n_turbines))
        self.nest_capacities = config.NEST_CAPACITIES 

        # 预先计算 R_lk 矩阵
        print("\n--- Phase I: Calculating Reachability Matrix R_lk ---")
        self.R_lk = self.physical_model.calculate_reachability_matrix(
            self.nest_locations_df, 
            self.turbines_df
        )

    def formulate_optimization_problem(self):
        """
        构建 Phase I 的整数线性规划模型。
        """
        print("\n--- Phase I: Formulating Optimization Problem ---")
        prob = pulp.LpProblem("Nest_Deployment_Assignment", pulp.LpMinimize)

        # --- 决策变量 ---
        x_l_tau = pulp.LpVariable.dicts("x", 
                                       ((l, tau) for l in self.nest_ids for tau in self.nest_capacities), 
                                       cat='Binary')
        
        y_l_k = pulp.LpVariable.dicts("y", 
                                     ((l, k) for l in self.nest_ids for k in self.turbine_ids), 
                                     cat='Binary')

        # --- 目标函数 (关键修改) ---
        # 总成本 = sum( (机巢基准成本 + 无人机数量 * 单机成本) * 是否建设 )
        # config.NEST_BASE_COSTS[tau]: 不同类型机巢的基准建设成本
        # config.OMEGA_2 * tau: 该机巢内配置的无人机总成本
        # prob += (
        #     pulp.lpSum(
        #         config.OMEGA_1 * x_l_tau[(l, tau)] + config.OMEGA_2 * tau * x_l_tau[(l, tau)] 
        #         for l in self.nest_ids for tau in self.nest_capacities
        #     )
        # ), "Total_Cost"
        prob += (
            pulp.lpSum(
                (config.NEST_BASE_COSTS[tau] + config.OMEGA_2 * tau) * x_l_tau[(l, tau)] 
                for l in self.nest_ids for tau in self.nest_capacities
            )
        ), "Total_Cost"

        # --- 约束条件 ---
        
        # 1. 选址唯一性
        for l in self.nest_ids:
            prob += (
                pulp.lpSum(x_l_tau[(l, tau)] for tau in self.nest_capacities) <= 1, 
                f"Nest_Location_Unique_{l}"
            )

        # 2. 任务全覆盖
        for k in self.turbine_ids:
            prob += (
                pulp.lpSum(y_l_k[(l, k)] for l in self.nest_ids) == 1,
                f"Turbine_Coverage_{k}"
            )

        # 3. 建设依托
        for l in self.nest_ids:
            for k in self.turbine_ids:
                prob += (
                    y_l_k[(l, k)] <= pulp.lpSum(x_l_tau[(l, tau)] for tau in self.nest_capacities),
                    f"Assignment_Requires_Nest_{l}_{k}"
                )
        
        # 4. 物理可达性 (R_lk)
        count_unreachable = 0
        for l in self.nest_ids:
            for k in self.turbine_ids:
                if self.R_lk[l, k] == 0: 
                    prob += (y_l_k[(l, k)] == 0, f"Physical_Unreachable_{l}_{k}")
                    count_unreachable += 1
        print(f"Added {count_unreachable} unreachability constraints.")
        
        # 5. 机巢容量限制
        for l in self.nest_ids:
            assigned_turbines_count = pulp.lpSum(y_l_k[(l, k)] for k in self.turbine_ids)
            total_capacity_at_l = pulp.lpSum(tau * x_l_tau[(l, tau)] for tau in self.nest_capacities)
            
            prob += (
                assigned_turbines_count <= config.ETA * total_capacity_at_l,
                f"Nest_Capacity_{l}"
            )

        return prob, x_l_tau, y_l_k

    def solve(self):
        """
        求解 Phase I 优化问题 (带时间限制)。
        """
        print("\n--- Phase I: Solving Optimization Problem ---")
        prob, x_l_tau, y_l_k = self.formulate_optimization_problem()
        
        # 获取时间限制配置，默认30秒
        time_limit = getattr(config, 'SOLVER_TIME_LIMIT', 30)

        # 配置求解器
        if config.OPTIMIZER == 'PULP_CBC_CMD' or config.OPTIMIZER is None:
            # gapRel=0.01 表示只要解在最优解 1% 范围内就停止（可选）
            # timeLimit=time_limit 设置最大秒数
            print(f"Configuring solver with time limit: {time_limit} seconds...")
            solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=time_limit)
        else:
            # 如果用户指定了其他求解器（如 Gurobi），需要单独设置参数，这里默认处理 CBC
            try:
                solver = pulp.getSolver(config.OPTIMIZER)
            except:
                solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=time_limit)
            
        try:
            prob.solve(solver)
        except Exception as e:
            print(f"Error during solving: {e}")
            return None, None, None
            
        # 检查求解状态
        status_str = pulp.LpStatus[prob.status]
        print(f"Optimization Status: {status_str}")
        
        # === 核心修改逻辑 ===
        # 只要找到了可行解 (Objective 不为 None)，即使超时的状态是 'Not Solved' 或 'Undefined'，也接受结果
        if prob.objective.value() is not None:
            if status_str != 'Optimal':
                print(f"Time limit reached or suboptimal. Accepting current best solution. Cost: {pulp.value(prob.objective):.2f}")
            else:
                print(f"Optimal solution found. Cost: {pulp.value(prob.objective):.2f}")
            
            # 提取结果
            selected_nests = [] 
            assignments = []   
            
            for l in self.nest_ids:
                # 检查机巢建设
                built = False
                for tau in self.nest_capacities:
                    val = pulp.value(x_l_tau[(l, tau)])
                    # 浮点数比较安全阈值
                    if val is not None and val > 0.5:
                        selected_nests.append({'nest_id': l, 'capacity': tau})
                        built = True
                        
                # 检查风机分配
                for k in self.turbine_ids:
                    val_y = pulp.value(y_l_k[(l, k)])
                    if val_y is not None and val_y > 0.5:
                        assignments.append({'nest_id': l, 'turbine_id': k})
            
            selected_nests_df = pd.DataFrame(selected_nests)
            assignments_df = pd.DataFrame(assignments)
            
            return selected_nests_df, assignments_df, prob.objective.value()
            
        else:
            print("Optimization failed to find ANY feasible solution within the time limit.")
            print("Try increasing ETA, increasing SOLVER_TIME_LIMIT, or checking R_lk connectivity.")
            return None, None, None

    def run(self):
        if self.nest_locations_df.empty or self.turbines_df.empty or self.R_lk is None:
            print("Phase I Optimizer cannot run due to missing data.")
            return None, None, None

        selected_nests_df, assignments_df, objective_value = self.solve()
        
        if selected_nests_df is not None:
            print("\n--- Phase I Results ---")
            print(f"Selected Nests: {len(selected_nests_df)}")
            print(f"Assigned Turbines: {len(assignments_df)}")
            print(f"Total Cost (Z1): {objective_value:.2f}")
            return selected_nests_df, assignments_df, objective_value
        else:
            print("Phase I optimization failed.")
            return None, None, None