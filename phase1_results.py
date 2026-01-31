import os
from Model import config
from Model import physical_model
# from Solver.Problem1 import phase1_optimizer_heuristic as phase1_optimizer
from Solver.Problem1 import phase1_optimizer_grso_ga_pso_aco_explicit
import matplotlib.pyplot as plt
from Tools.deployment_plot import plot_results
from Tools.solution2dataframe import convert_solution_to_dataframe
# --- 全局绘图设置 ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

def run_comparison(pm, solver):
    """
    核心逻辑：运行所有算法、绘图并打印报告
    """
    nest_ids = list(range(len(pm.nest_locations)))
    
    # 1. 定义算法配置
    # 可以在这里统一调整迭代次数和种群大小
    algorithms = {
        'G-RSO': lambda: solver.run_grso(history_length=200),
        'RG':    lambda: solver.run_reverse_greedy(history_length=200), # 新增这一行
        'GA':    lambda: solver.run_ga(pop_size=100, generations=200),
        'PSO':   lambda: solver.run_pso(swarm_size=100, iterations=200),
        'ACO':   lambda: solver.run_aco(n_ants=100, iterations=200)
    }

    results = {}

    print("\n========================================")
    print("   Starting Comparative Analysis")
    print("========================================")
    
    # 2. 依次运行算法
    for name, func in algorithms.items():
        try:
            print(f"\n>>> Running {name}...")
            results[name] = func()
        except Exception as e:
            print(f"Error running {name}: {e}")

    # 3. 绘制收敛对比图
    if results:
        plt.figure()
        
        styles = {
            'G-RSO': {'color': 'red', 'ls': '-', 'lw': 2.0}, # 突出显示
            'RG':    {'color': 'blue', 'ls': '--', 'lw': 1.0}, # 文献方法用蓝色虚线
            'GA':    {'color': 'orange', 'ls': '--', 'lw': 1.5},
            'PSO':   {'color': 'green', 'ls': '-.', 'lw': 1.5},
            'ACO':   {'color': 'purple', 'ls': ':', 'lw': 1.5}
        }
        
        for name, res in results.items():
            hist = res['history']
            s = styles.get(name, {'color': 'blue', 'ls': '-'})
            plt.plot(hist, label=name, color=s['color'], linestyle=s['ls'], linewidth=s.get('lw', 1.5))
        
        # plt.yscale('log') # 对数坐标
        plt.title('Convergence Comparison')
        plt.xlabel('Iteration')
        plt.ylabel('Total Cost + Penalty')
        plt.legend()
        plt.grid(True, alpha=0.3)
        print("\nDisplaying Convergence Plot...")
        plt.show()

    # 4. 绘制各算法选址地图
    for name, res in results.items():
        print(f"\n--- Plotting {name} Map ---")
        df_n, df_a = convert_solution_to_dataframe(res['solution'], nest_ids)
        
        if not df_n.empty:
            # 传入 fitness (含惩罚) 作为显示的 Cost
            plot_results(pm, (df_n, df_a, res['fitness']))
        else:
            print(f"{name} result is infeasible (no nests built). Skipping plot.")


    # 5. 打印详细数据报告 (已修改)
    print("\n========================================")
    print("   Final Violation & Cost Report")
    print("========================================")
    
    # 调整表头宽度以适应新增列
    header = f"{'Algorithm':<8} | {'Feas.':<6} | {'Score':<10} | {'Cost':<8} | {'Nests':<6} | {'UAVs':<6} | {'Violations (Un/Gh/Ov)'}"
    print(header)
    print("-" * len(header))
    
    for name, res in results.items():
        d = res['details']
        
        # 提取数据
        is_feas = "YES" if d['is_feasible'] else "NO"
        score = res['fitness']
        pure = d['pure_cost']
        n_nests = d['total_nests_built']
        n_uavs = d['total_uavs_deployed']
        
        # 格式化违规信息
        if d['is_feasible']:
            viol_str = "None"
        else:
            v = d['violations']
            viol_str = f"{v['unreachable']} / {v['ghost_nest']} / {v['overcapacity']}"
            
        # 打印行
        print(f"{name:<8} | {is_feas:<6} | {score:<10.2f} | {pure:<8.2f} | {n_nests:<6} | {n_uavs:<6} | {viol_str}")
def main():
    """
    程序入口：初始化与启动
    """
    # 1. 检查数据
    if not os.path.exists(config.DEM_FILE) or not os.path.exists(config.OSM_FILE):
        config.generate_simulated_data(config.N_NESTS, config.N_TURBINES)
   
    # 2. 初始化物理模型
    try:
        pm = physical_model.PhysicalModel(config.DEM_FILE, config.WIND_SPEED)
        print("Physical model initialized.")
    except Exception as e:
        print(f"Init failed: {e}"); return

    # 3. 初始化求解器
    try:
        solver = phase1_optimizer_grso_ga_pso_aco_explicit.ExplicitMetaheuristicSolvers(pm)
    except Exception as e:
        print(f"Solver Init failed: {e}"); return

    # 4. 运行对比逻辑
    run_comparison(pm, solver)
if __name__ == "__main__":
    main()