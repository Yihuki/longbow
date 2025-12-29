"""诊断dynamax模型参数 - 用uv运行"""

import json
import numpy as np
import jax.numpy as jnp
from pathlib import Path

from longbow.utils.model_utils_dynamax import ModelBuilder

TEST_DATA_FOLDER = Path(__file__).parent / "tests" / "test_data"
MODEL_FILE = TEST_DATA_FOLDER / "models" / "mas_15+sc_10x5p.json"


def main():
    print("=" * 80)
    print("Dynamax模型参数诊断")
    print("=" * 80)
    
    # 构建dynamax模型
    print("\n构建Dynamax模型...")
    with open(MODEL_FILE) as f:
        model_config = json.load(f)
    
    array_model = model_config['array']
    cdna_model = model_config['cdna']
    
    model = ModelBuilder.make_full_longbow_model(array_model, cdna_model)
    
    # 获取参数 - 通过params访问
    state_names = model.states
    initial_probs = np.array(model.params.initial.probs)
    trans_matrix = np.array(model.params.transitions.transition_matrix)
    
    print(f"\n状态数: {len(state_names)}")
    
    # 1. 初始概率分析
    print("\n" + "=" * 80)
    print("1. 初始概率分析")
    print("=" * 80)
    
    nonzero = np.where(initial_probs > 1e-10)[0]
    print(f"\n有初始概率的状态数: {len(nonzero)}")
    print("\n非零初始概率:")
    for i in sorted(nonzero, key=lambda x: -initial_probs[x]):
        print(f"  {i}: {state_names[i]}: {initial_probs[i]:.6f}")
    
    print(f"\n初始概率总和: {initial_probs.sum():.6f}")
    
    # 2. 关键状态的初始概率
    print("\n" + "=" * 80)
    print("2. 关键状态的初始概率")
    print("=" * 80)
    
    key_states = ['A:I0', 'A:M1', '5p_Adapter:I0', '5p_Adapter:D1', '5p_Adapter:M1']
    for state in key_states:
        if state in state_names:
            idx = state_names.index(state)
            print(f"  {state}: {initial_probs[idx]:.6f}")
    
    # 3. random:RDA的转移
    print("\n" + "=" * 80)
    print("3. random:RDA的转移概率分析")
    print("=" * 80)
    
    for i, name in enumerate(state_names):
        if name == 'random:RDA':
            print(f"\nrandom:RDA索引: {i}")
            row = trans_matrix[i]
            nonzero = np.where(row > 1e-10)[0]
            print(f"  转移目标数: {len(nonzero)}")
            for j in sorted(nonzero, key=lambda x: -row[x])[:20]:
                print(f"  -> {state_names[j]}: {row[j]:.6f}")
            break
    
    # 4. A:I16, A:M16, A:D16的转移
    print("\n" + "=" * 80)
    print("4. A:I16, A:M16, A:D16的转移概率分析")
    print("=" * 80)
    
    for suffix in ['I16', 'M16', 'D16']:
        state_name = f'A:{suffix}'
        if state_name in state_names:
            idx = state_names.index(state_name)
            row = trans_matrix[idx]
            nonzero = np.where(row > 1e-10)[0]
            print(f"\n{state_name} (索引{idx}):")
            for j in sorted(nonzero, key=lambda x: -row[x])[:10]:
                print(f"  -> {state_names[j]}: {row[j]:.6f}")
    
    # 5. 检查是否有random-start状态
    print("\n" + "=" * 80)
    print("5. 检查random相关状态")
    print("=" * 80)
    
    for i, name in enumerate(state_names):
        if 'random' in name.lower():
            print(f"  {i}: {name}")
    
    # 6. 所有A相关状态的索引
    print("\n" + "=" * 80)
    print("6. 所有A相关状态的索引")
    print("=" * 80)
    
    for i, name in enumerate(state_names):
        if name.startswith('A:'):
            print(f"  {i}: {name}")
    
    # 7. 所有5p_Adapter相关状态的索引
    print("\n" + "=" * 80)
    print("7. 所有5p_Adapter相关状态的索引")
    print("=" * 80)
    
    for i, name in enumerate(state_names):
        if name.startswith('5p_Adapter:'):
            print(f"  {i}: {name}")


if __name__ == "__main__":
    main()
