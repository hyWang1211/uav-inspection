import pandas as pd
def convert_solution_to_dataframe(solution_dict, nest_ids):
    """
    [通用辅助函数] 
    将所有算法(G-RSO, GA, PSO, ACO)返回的统一数组格式转换为 DataFrame，
    以便复用 plot_results 函数进行绘图。
    """
    nests_arr = solution_dict['nests']       
    assigns_arr = solution_dict['assignments'] 

    # 1. 转换机巢数据
    selected_nests_data = []
    active_nests_set = set()
    
    for i, val in enumerate(nests_arr):
        nest_type = int(val)
        if nest_type > 0:
            selected_nests_data.append({
                'nest_id': nest_ids[i],
                'capacity': nest_type,
                'load': 0 # load仅用于绘图颜色，显式算法未直接提供，此处填0不影响核心绘图
            })
            active_nests_set.add(i)
            
    df_nests = pd.DataFrame(selected_nests_data)

    # 2. 转换指派数据
    assignments_data = []
    
    for t_idx, n_idx in enumerate(assigns_arr):
        n_idx = int(n_idx)
        # 过滤无效指派：
        # 1. 索引必须在合法范围内
        # 2. 指派的目标机巢必须是“已建设(Type>0)”的
        if 0 <= n_idx < len(nests_arr) and n_idx in active_nests_set:
            assignments_data.append({
                'turbine_id': t_idx,
                'nest_id': nest_ids[n_idx]
            })
            
    df_assigns = pd.DataFrame(assignments_data)
    
    return df_nests, df_assigns