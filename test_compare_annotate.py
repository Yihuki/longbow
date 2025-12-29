"""
简单对比测试脚本：测试annotate函数并与预期结果进行详细对比
对比SC（CIGAR格式的路径）和SG（分段信息）
"""

import json
import sys
import pysam
from pathlib import Path

# 导入dynamax版本模块
from longbow.utils.model_dynamax import LibraryModel as LibraryModelDynamax
from longbow.utils.bam_utils import reverse_complement


# 测试配置
TEST_DATA_FOLDER = Path(__file__).parent / "tests" / "test_data"

# 只测试 mas_15+sc_10x5p
INPUT_BAM = TEST_DATA_FOLDER / "mas15_test_input.bam"
EXPECTED_BAM = TEST_DATA_FOLDER / "annotate" / "mas_15+sc_10x5p.expected.bam"
MODEL_FILE = TEST_DATA_FOLDER / "models" / "mas_15+sc_10x5p.json"


def sequence_to_ints(seq):
    """将DNA序列转换为整数数组 (A=0, C=1, G=2, T=3)"""
    nuc_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}
    return [nuc_to_int.get(base.upper(), 0) for base in seq]


def state_path_to_cigar(state_path):
    """将状态路径转换为CIGAR格式
    
    例如: ['random:RI', '5p_Adapter:I0', '5p_Adapter:D1', ...] 
    -> 'random:RI,5p_Adapter:I0D1,...'
    """
    import re
    
    # 按状态分组
    segments = []
    current_state = None
    current_ops = []
    
    for state in state_path:
        # 解析状态名和操作
        match = re.match(r'(.+?):(.+)', state)
        if match:
            state_name = match.group(1)
            op = match.group(2)
            
            if state_name != current_state:
                # 保存之前的段
                if current_state is not None:
                    segments.append(f"{current_state}:{''.join(current_ops)}")
                current_state = state_name
                current_ops = [op]
            else:
                current_ops.append(op)
    
    # 保存最后一段
    if current_state is not None:
        segments.append(f"{current_state}:{''.join(current_ops)}")
    
    return ','.join(segments)


def collapse_path_to_segments(state_path):
    """将状态路径折叠为分段信息（类似SG标签格式）
    
    例如: ['5p_Adapter:I0', '5p_Adapter:I1', '5p_Adapter:D2', 'cDNA:I0', ...]
    -> '5p_Adapter:0-2,cDNA:3-100,...'
    """
    import re
    
    segments = []
    current_state = None
    start_pos = 0
    
    for i, state in enumerate(state_path):
        match = re.match(r'(.+?):.+', state)
        if match:
            state_name = match.group(1)
            
            if state_name != current_state:
                # 保存之前的段
                if current_state is not None:
                    segments.append(f"{current_state}:{start_pos}-{i-1}")
                current_state = state_name
                start_pos = i
    
    # 保存最后一段
    if current_state is not None:
        segments.append(f"{current_state}:{start_pos}-{len(state_path)-1}")
    
    return ','.join(segments)


def main():
    """主函数"""
    print("=" * 60)
    print("Longbow annotate dynamax详细对比测试")
    print("=" * 60)
    
    # 加载模型配置
    if not MODEL_FILE.exists():
        print(f"❌ 错误: 模型文件不存在: {MODEL_FILE}")
        return 1
    
    with open(MODEL_FILE) as f:
        model_config = json.load(f)
    
    print(f"加载模型: {model_config['name']}")
    
    # 构建dynamax模型
    print("构建dynamax模型...")
    dynamax_model = LibraryModelDynamax.from_json_obj(model_config)
    print(f"模型构建完成，状态数: {dynamax_model.n_states}")
    print(f"模型是否已构建: {dynamax_model.is_built}")
    
    # 读取输入bam文件
    print(f"\n读取输入bam文件: {INPUT_BAM}")
    pysam.set_verbosity(0)
    with pysam.AlignmentFile(INPUT_BAM, "rb", check_sq=False, require_index=False) as bam_file:
        reads = list(bam_file)
    
    print(f"读取到 {len(reads)} 个read")
    
    # 读取预期输出bam文件
    print(f"读取预期输出bam文件: {EXPECTED_BAM}")
    with pysam.AlignmentFile(EXPECTED_BAM, "rb", check_sq=False, require_index=False) as bam_file:
        expected_reads = {r.query_name: r for r in bam_file}
    
    print(f"预期输出包含 {len(expected_reads)} 个read")
    
    # 测试第一条序列
    read = reads[0]
    print(f"\n{'='*60}")
    print(f"测试第一条序列: {read.query_name}")
    print(f"{'='*60}")
    
    # 获取预期输出中的对应read
    if read.query_name not in expected_reads:
        print(f"❌ 错误: 在预期输出中未找到read: {read.query_name}")
        return 1
    
    expected_read = expected_reads[read.query_name]
    
    print(f"\n--- 序列信息 ---")
    print(f"输入序列长度: {len(read.query_sequence)}")
    print(f"预期序列长度: {len(expected_read.query_sequence)}")
    
    # 检查序列是否匹配
    if read.query_sequence == expected_read.query_sequence:
        print(f"序列匹配: 正向序列")
        is_rc = False
    elif reverse_complement(read.query_sequence) == expected_read.query_sequence:
        print(f"序列匹配: 反向互补序列")
        is_rc = True
    else:
        print(f"序列不匹配")
        is_rc = expected_read.get_tag('RC') if expected_read.has_tag('RC') else False
    
    # 运行annotate
    seq_ints = sequence_to_ints(read.query_sequence)
    rc_seq_ints = sequence_to_ints(reverse_complement(read.query_sequence))
    
    print("\n--- Dynamax annotate结果 ---")
    
    print("\n正向序列:")
    logp, state_path = dynamax_model.annotate_with_log_prob(seq_ints)
    print(f"  对数概率: {logp:.4f}")
    print(f"  状态路径长度: {len(state_path)}")
    
    print("\n反向互补序列:")
    rc_logp, rc_state_path = dynamax_model.annotate_with_log_prob(rc_seq_ints)
    print(f"  对数概率: {rc_logp:.4f}")
    print(f"  状态路径长度: {len(rc_state_path)}")
    
    # 选择更好的结果
    if rc_logp > logp:
        print(f"\n→ 选择反向互补结果 (RC=True)")
        best_logp = rc_logp
        best_path = rc_state_path
        best_is_rc = True
    else:
        print(f"\n→ 选择正向结果 (RC=False)")
        best_logp = logp
        best_path = state_path
        best_is_rc = False
    
    # 查看预期输出中的标签
    print(f"\n--- 预期输出标签 ---")
    
#    # SC标签 (CIGAR格式的路径)
#    if expected_read.has_tag('SC'):
#        expected_sc = expected_read.get_tag('SC')
#        print(f"\n预期SC标签 (CIGAR路径):")
#        print(f"  值: {expected_sc[:200]}..." if len(expected_sc) > 200 else f"  值: {expected_sc}")
#        expected_sc_parts = expected_sc.split(',')
#        print(f"  段数: {len(expected_sc_parts)}")
#        print(f"  前5段: {expected_sc_parts[:5]}")
#        print(f"  后5段: {expected_sc_parts[-5:]}")
#    else:
#        expected_sc = None
#        print(f"⚠ 预期输出中没有SC标签")
    
    # SG标签 (分段信息)
    if expected_read.has_tag('SG'):
        expected_sg = expected_read.get_tag('SG')
        print(f"\n预期SG标签 (分段信息):")
        print(f"  值: {expected_sg[:200]}..." if len(expected_sg) > 200 else f"  值: {expected_sg}")
        expected_sg_parts = expected_sg.split(',')
        print(f"  段数: {len(expected_sg_parts)}")
        print(f"  前5段: {expected_sg_parts[:5]}")
        print(f"  后5段: {expected_sg_parts[-5:]}")
    else:
        expected_sg = None
        print(f"⚠ 预期输出中没有SG标签")
    
    # 计算dynamax的SC和SG
    dynamax_sc = state_path_to_cigar(best_path)
    dynamax_sg = collapse_path_to_segments(best_path)
    
#    print(f"\n--- Dynamax结果转换 ---")
#    print(f"\nDynamax SC标签 (CIGAR路径):")
#    print(f"  值: {dynamax_sc[:200]}..." if len(dynamax_sc) > 200 else f"  值: {dynamax_sc}")
#    dynamax_sc_parts = dynamax_sc.split(',')
#    print(f"  段数: {len(dynamax_sc_parts)}")
#    print(f"  前5段: {dynamax_sc_parts[:5]}")
#    print(f"  后5段: {dynamax_sc_parts[-5:]}")
    
    print(f"\nDynamax SG标签 (分段信息):")
    print(f"  值: {dynamax_sg[:200]}..." if len(dynamax_sg) > 200 else f"  值: {dynamax_sg}")
    dynamax_sg_parts = dynamax_sg.split(',')
    print(f"  段数: {len(dynamax_sg_parts)}")
    print(f"  前5段: {dynamax_sg_parts[:5]}")
    print(f"  后5段: {dynamax_sg_parts[-5:]}")
    
#    # 比较SC标签
#    print(f"\n--- SC标签对比 ---")
#    if expected_sc and dynamax_sc:
#        if expected_sc == dynamax_sc:
#            print(f"✓ SC标签完全匹配!")
#        else:
#            print(f"⚠ SC标签不匹配")
#            print(f"  预期段数: {len(expected_sc_parts)}")
#            print(f"  Dynamax段数: {len(dynamax_sc_parts)}")
#            
#            # 统计不匹配
#            expected_set = set(expected_sc_parts)
#            dynamax_set = set(dynamax_sc_parts)
#            
#            only_expected = expected_set - dynamax_set
#            only_dynamax = dynamax_set - expected_set
#            
#            if only_expected:
#                print(f"  预期独有: {list(only_expected)[:5]}...")
#            if only_dynamax:
#                print(f"  Dynamax独有: {list(only_dynamax)[:5]}...")
    
#    # 比较SG标签
#    print(f"\n--- SG标签对比 ---")
#    if expected_sg and dynamax_sg:
#        if expected_sg == dynamax_sg:
#            print(f"✓ SG标签完全匹配!")
#        else:
#            print(f"⚠ SG标签不匹配")
#            print(f"  预期段数: {len(expected_sg_parts)}")
#            print(f"  Dynamax段数: {len(dynamax_sg_parts)}")
#            
#            # 统计不匹配
#            expected_set = set(expected_sg_parts)
#            dynamax_set = set(dynamax_sg_parts)
#            
#            only_expected = expected_set - dynamax_set
#            only_dynamax = dynamax_set - expected_set
#            
#            if only_expected:
#                print(f"  预期独有: {list(only_expected)[:5]}...")
#            if only_dynamax:
#                print(f"  Dynamax独有: {list(only_dynamax)[:5]}...")
#    
#    print(f"\n" + "=" * 60)
#    print("测试完成!")
#    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
