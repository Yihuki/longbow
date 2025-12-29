"""诊断pomegranate模型参数"""

import json
import numpy as np
from pathlib import Path

from longbow.utils.model import LibraryModel as PomegranateModel

TEST_DATA_FOLDER = Path(__file__).parent / "tests" / "test_data"
MODEL_FILE = TEST_DATA_FOLDER / "models" / "mas_15+sc_10x5p.json"


def get_pomegranate_model():
    """构建pomegranate模型"""
    with open(MODEL_FILE) as f:
        model_config = json.load(f)
    return PomegranateModel.from_json_obj(model_config)


def get_pomegranate_params(pom_model):
    """提取pomegranate模型的参数"""
    model_dict = pom_model.hmm.to_dict()
    
    params = {
        'state_names': [s['name'] for s in model_dict['states']],
        'start_state_index': model_dict.get('start_index', -1),
        'end_state_index': model_dict.get('end_index', -1),
    }
    
    # 从edges提取转移概率
    edges = model_dict['edges']
    num_states = len(model_dict['states'])
    trans_matrix = np.zeros((num_states, num_states))
    
    for edge in edges:
        if len(edge) >= 3:
            src_idx, dst_idx, prob = edge[0], edge[1], edge[2]
            trans_matrix[src_idx, dst_idx] = prob
    
    params['transition_matrix'] = trans_matrix
    
    # 提取初始概率（从start状态出发的转移概率）
    start_idx = params['start_state_index']
    if start_idx >= 0:
        # 归一化
        row = trans_matrix[start_idx].copy()
        if row.sum() > 0:
            row = row / row.sum()
        params['initial_probs'] = row
        params['initial_probs_raw'] = trans_matrix[start_idx].copy()
    
    return params


def main():
    print("=" * 80)
    print("Pomegranate模型参数诊断")
    print("=" * 80)
    
    # 构建pomegranate模型
    print("\n构建Pomegranate模型...")
    pom_model = get_pomegranate_model()
    params = get_pomegranate_params(pom_model)
    
    print(f"\n状态数: {len(params['state_names'])}")
    print(f"Start状态索引: {params['start_state_index']}")
    print(f"End状态索引: {params['end_state_index']}")
    
    # 1. 分析random-start的转移（原始和归一化后）
    print("\n" + "=" * 80)
    print("1. random-start的转移概率分析")
    print("=" * 80)
    
    for i, name in enumerate(params['state_names']):
        if name == 'random-start':
            start_idx = i
            break
    
    print(f"\nrandom-start索引: {start_idx}")
    print("\n原始转移概率:")
    row_raw = params['transition_matrix'][start_idx]
    nonzero = np.where(row_raw > 1e-10)[0]
    for j in nonzero:
        print(f"  -> {params['state_names'][j]}: {row_raw[j]:.4f}")
    print(f"  原始总和: {row_raw.sum():.4f}")
    
    print("\n归一化后转移概率:")
    row_norm = params['initial_probs']
    nonzero = np.where(row_norm > 1e-10)[0]
    for j in nonzero:
        print(f"  -> {params['state_names'][j]}: {row_norm[j]:.4f}")
    print(f"  归一化总和: {row_norm.sum():.4f}")
    
    # 2. 分析A-start的转移
    print("\n" + "=" * 80)
    print("2. A-start的转移概率分析")
    print("=" * 80)
    
    for i, name in enumerate(params['state_names']):
        if name == 'A-start':
            print(f"\nA-start索引: {i}")
            row = params['transition_matrix'][i]
            nonzero = np.where(row > 1e-10)[0]
            for j in nonzero:
                print(f"  -> {params['state_names'][j]}: {row[j]:.4f}")
            break
    
    # 3. 分析5p_Adapter-start的转移
    print("\n" + "=" * 80)
    print("3. 5p_Adapter-start的转移概率分析")
    print("=" * 80)
    
    for i, name in enumerate(params['state_names']):
        if name == '5p_Adapter-start':
            print(f"\n5p_Adapter-start索引: {i}")
            row = params['transition_matrix'][i]
            nonzero = np.where(row > 1e-10)[0]
            for j in nonzero:
                print(f"  -> {params['state_names'][j]}: {row[j]:.4f}")
            break
    
    # 4. 计算有效初始概率
    print("\n" + "=" * 80)
    print("4. 有效初始概率计算（random-start归一化后 * start状态转移）")
    print("=" * 80)
    
    # 找到A:I0, A:M1, 5p_Adapter:I0, 5p_Adapter:D1, 5p_Adapter:M1
    key_states = ['A:I0', 'A:M1', '5p_Adapter:I0', '5p_Adapter:D1', '5p_Adapter:M1']
    
    print("\n有效初始概率（从random-start开始）:")
    for state in key_states:
        if state in params['state_names']:
            idx = params['state_names'].index(state)
            # 从random-start到A-start或5p_Adapter-start的概率
            # 然后从那里到具体状态的概率
            # random-start -> A-start: 0.25 (归一化后)
            # random-start -> 5p_Adapter-start: 0.25 (归一化后)
            # A-start -> A:I0: 0.05, A-start -> A:M1: 0.90
            # 5p_Adapter-start -> 5p_Adapter:I0: 0.05, 5p_Adapter:D1: 0.05, 5p_Adapter:M1: 0.90
            
            if state.startswith('A:'):
                prob = 0.25 * params['transition_matrix'][params['state_names'].index('A-start')][idx]
            else:
                prob = 0.25 * params['transition_matrix'][params['state_names'].index('5p_Adapter-start')][idx]
            
            print(f"  {state}: {prob:.4f}")
    
    # 5. 分析random:RDA的转移
    print("\n" + "=" * 80)
    print("5. random:RDA的转移概率分析")
    print("=" * 80)
    
    for i, name in enumerate(params['state_names']):
        if name == 'random:RDA':
            print(f"\nrandom:RDA索引: {i}")
            row = params['transition_matrix'][i]
            nonzero = np.where(row > 1e-10)[0]
            print(f"  转移目标数: {len(nonzero)}")
            for j in sorted(nonzero, key=lambda x: -row[x])[:20]:
                print(f"  -> {params['state_names'][j]}: {row[j]:.4f}")
            break
    
    # 6. 分析A结束状态的转移
    print("\n" + "=" * 80)
    print("6. A:I16, A:M16, A:D16的转移概率分析")
    print("=" * 80)
    
    for suffix in ['I16', 'M16', 'D16']:
        state_name = f'A:{suffix}'
        if state_name in params['state_names']:
            idx = params['state_names'].index(state_name)
            row = params['transition_matrix'][idx]
            nonzero = np.where(row > 1e-10)[0]
            print(f"\n{state_name} (索引{idx}):")
            for j in nonzero:
                print(f"  -> {params['state_names'][j]}: {row[j]:.4f}")
    
    # 7. 打印所有A相关状态
    print("\n" + "=" * 80)
    print("7. 所有A相关状态的索引")
    print("=" * 80)
    
    for i, name in enumerate(params['state_names']):
        if name.startswith('A:'):
            print(f"  {i}: {name}")
            if i > 60:
                print("  ... (更多)")
                break
    
    # 8. 打印所有5p_Adapter相关状态
    print("\n" + "=" * 80)
    print("8. 所有5p_Adapter相关状态的索引")
    print("=" * 80)
    
    for i, name in enumerate(params['state_names']):
        if name.startswith('5p_Adapter:'):
            print(f"  {i}: {name}")


if __name__ == "__main__":
    main()
