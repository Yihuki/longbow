#!/usr/bin/env python
"""详细对比dynamax和pomegranate的annotate状态路径"""

import os
import sys

# 添加longbow到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np

# 避免导入bam_utils（它会导入model_dynamax造成循环依赖）
# 直接从bam_utils复制必要的函数
def load_bam_file(bam_file, filter_valid=True):
    """Load reads from a BAM file.
    
    Args:
        bam_file: Path to the BAM file
        filter_valid: If True, filter reads with valid flag (flag & 0x800 == 0)
        
    Returns:
        list: List of reads
    """
    from pysam import AlignmentFile
    
    reads = []
    with AlignmentFile(bam_file, "rb", check_sq=False) as f:
        for read in f:
            if filter_valid:
                if read.flag & 0x800:
                    continue
                    
            reads.append(read)
            
    return reads

def reverse_complement(seq):
    """计算序列的反向互补"""
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return ''.join([complement.get(b, b).upper() for b in reversed(seq)])

def base_to_int(base):
    """将碱基转换为整数"""
    base = base.upper()
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    return mapping.get(base, 4)

def main():
    print("=" * 70)
    print("状态路径详细对比测试")
    print("=" * 70)
    
    # 配置
    model_name = "mas_15+sc_10x5p"
    model_file = f"tests/test_data/models/{model_name}.json"
    input_bam = "tests/test_data/mas15_test_input.bam"
    
    # 读取测试数据
    reads = load_bam_file(input_bam, filter_valid=True)
    
    test_read = reads[0]
    test_seq = list(test_read.query_sequence)
    test_seq_rc = reverse_complement(test_seq)
    
    print(f"\n测试read: {test_read.query_name}")
    print(f"序列长度: {len(test_seq)}")
    
    # 先测试dynamax
    print(f"\n构建dynamax模型...")
    from longbow.utils.model_dynamax import LibraryModel as DynamaxLibraryModel
    dynamax_model = DynamaxLibraryModel.from_json_file(model_file)
    print(f"Dynamax模型状态数: {dynamax_model.n_states}")
    
    # 转换序列为整数
    int_seq = [base_to_int(b) for b in test_seq_rc]
    int_seq_fwd = [base_to_int(b) for b in test_seq]
    
    # 运行annotate
    print(f"\n--- Dynamax Annotate结果 ---")
    
    log_prob_dyn_rc, state_path_dyn_rc = dynamax_model.annotate(int_seq)
    log_prob_dyn_fwd, state_path_dyn_fwd = dynamax_model.annotate(int_seq_fwd)
    
    print(f"Dynamax (RC) log_prob: {log_prob_dyn_rc:.4f}")
    print(f"Dynamax (Fwd) log_prob: {log_prob_dyn_fwd:.4f}")
    
    # 选择概率最高的路径
    if log_prob_dyn_rc > log_prob_dyn_fwd:
        chosen_dyn = state_path_dyn_rc
        chosen_log_prob_dyn = log_prob_dyn_rc
        dyn_is_rc = True
    else:
        chosen_dyn = state_path_dyn_fwd
        chosen_log_prob_dyn = log_prob_dyn_fwd
        dyn_is_rc = False
    
    print(f"Dynamax选择: RC={dyn_is_rc}")
    print(f"Dynamax路径长度: {len(chosen_dyn)}")
    
    # 分析状态路径
    def count_adapters(state_path):
        adapter_counts = {}
        for state in state_path:
            if ":" in state:
                adapter = state.split(":")[0]
                adapter_counts[adapter] = adapter_counts.get(adapter, 0) + 1
        return adapter_counts
    
    dyn_counts = count_adapters(chosen_dyn)
    print(f"\nDynamax各adapter状态数:")
    for adapter, count in sorted(dyn_counts.items()):
        print(f"  {adapter}: {count}")
    
    # 找出状态转换点
    def find_transitions(state_path):
        transitions = []
        prev_adapter = None
        for i, state in enumerate(state_path):
            if ":" in state:
                adapter = state.split(":")[0]
                if adapter != prev_adapter:
                    transitions.append((i, state))
                    prev_adapter = adapter
        return transitions
    
    dyn_transitions = find_transitions(chosen_dyn)
    print(f"\nDynamax状态转换点 ({len(dyn_transitions)}个):")
    for i, state in dyn_transitions[:30]:
        print(f"  [{i}]: {state}")
    if len(dyn_transitions) > 30:
        print(f"  ... (共{len(dyn_transitions)}个转换)")
    
    # 现在测试pomegranate
    print(f"\n\n构建pomegranate模型...")
    from longbow.utils.model import LibraryModel as PomeLibraryModel
    pome_model = PomeLibraryModel.from_json_file(model_file)
    print(f"Pome模型状态数: {pome_model.n_states}")
    
    # 运行annotate
    print(f"\n--- Pomegranate Annotate结果 ---")
    
    log_prob_pome_rc, state_path_pome_rc = pome_model.annotate(test_seq_rc)
    log_prob_pome_fwd, state_path_pome_fwd = pome_model.annotate(test_seq)
    
    print(f"Pome (RC) log_prob: {log_prob_pome_rc:.4f}")
    print(f"Pome (Fwd) log_prob: {log_prob_pome_fwd:.4f}")
    
    # 选择概率最高的路径
    if log_prob_pome_rc > log_prob_pome_fwd:
        chosen_pome = state_path_pome_rc
        chosen_log_prob_pome = log_prob_pome_rc
        pome_is_rc = True
    else:
        chosen_pome = state_path_pome_fwd
        chosen_log_prob_pome = log_prob_pome_fwd
        pome_is_rc = False
    
    print(f"Pome选择: RC={pome_is_rc}")
    print(f"Pome路径长度: {len(chosen_pome)}")
    
    pome_counts = count_adapters(chosen_pome)
    print(f"\nPome各adapter状态数:")
    for adapter, count in sorted(pome_counts.items()):
        print(f"  {adapter}: {count}")
    
    pome_transitions = find_transitions(chosen_pome)
    print(f"\nPome状态转换点 ({len(pome_transitions)}个):")
    for i, state in pome_transitions[:30]:
        print(f"  [{i}]: {state}")
    if len(pome_transitions) > 30:
        print(f"  ... (共{len(pome_transitions)}个转换)")
    
    # 对比分析
    print(f"\n\n{'=' * 70}")
    print("对比分析")
    print(f"{'=' * 70}")
    
    print(f"\n路径长度: Dynamax={len(chosen_dyn)}, Pome={len(chosen_pome)}")
    print(f"转换点数: Dynamax={len(dyn_transitions)}, Pome={len(pome_transitions)}")
    print(f"对数概率: Dynamax={chosen_log_prob_dyn:.4f}, Pome={chosen_log_prob_pome:.4f}")
    
    # 详细对比每个转换点
    print(f"\n转换点详细对比 (前30个):")
    max_compare = min(len(dyn_transitions), len(pome_transitions), 30)
    matches = 0
    for i in range(max_compare):
        dyn_idx, dyn_state = dyn_transitions[i]
        pome_idx, pome_state = pome_transitions[i]
        match = "✓" if dyn_state == pome_state else "✗"
        if match == "✓":
            matches += 1
        print(f"  [{i}] {match} Dyn[{dyn_idx}]: {dyn_state}")
        print(f"        Pome[{pome_idx}]: {pome_state}")
    
    print(f"\n匹配数: {matches}/{max_compare}")
    
    if len(dyn_transitions) > max_compare:
        print(f"Dynamax额外转换点:")
        for i in range(max_compare, len(dyn_transitions)):
            print(f"  [{dyn_transitions[i][0]}]: {dyn_transitions[i][1]}")
    
    if len(pome_transitions) > max_compare:
        print(f"Pome额外转换点:")
        for i in range(max_compare, len(pome_transitions)):
            print(f"  [{pome_transitions[i][0]}]: {pome_transitions[i][1]}")
    
    print(f"\n{'=' * 70}")
    print("测试完成!")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    main()
