"""Independent CPU-only analysis. Inputs are immutable; output must be new."""
import argparse
import csv
import json
from pathlib import Path
import time
import torch
import cfpi_common as c
import selector_dual_domain_analysis as h
from selector_dual_domain_io import Study
from selector_dual_domain_probes import training_probes
from selector_dual_domain_cases import case_analysis


def progress(stage, **detail):
    print(json.dumps(dict(stage=stage,time=time.time(),**detail)),flush=True)


def markdown(stats, probes, cases):
    lines=['# P1 双域机制分析', '',
        '离线复核：无训练、无新增闭环。开发集 rare58/common64；不是盲测，不与正式噪声快照混算。', '',
        '| 方法 | rare NR | rare R | common NR | common R |','|---|---:|---:|---:|---:|']
    for arm in ('P0','P1','L_Pair','L_Pair_NLL','Continue','CL_NLL'):
        scores=[stats['summaries'][k]['arms'][arm]['mean']['score'] for k in
                ('development_NR','development_R','common_NR','common_R')]
        lines.append('| '+arm+' | '+' | '.join(f'{v:.6f}' for v in scores)+' |')
    lines+=['','## 机制与边界','',
        '- 已证实的恒等式：NLL = Pair + SetMass；P1 = 2 Pair + SetMass，CL_NLL = Pair + SetMass（省略共同 GRPO/KL）。这是分析工具，不是创新声明。',
        '- 双域均值表现存在差异，但不等于不可兼得；固定权重、优化强度和有限反馈支持仍是待区分假说。',
        '- Pair 与 NLL 在同一状态的 logit 梯度不是天然相反；参数共享、跨状态干扰必须单独验证。',
        '- 未比较候选没有真实回报标签；降低其概率不等同于已经证明其不安全。',
        '- 逐状态前向、梯度方向及后验案例只能提示机制，不能替代动作干预闭环。',
        '', '## 标签构成','', '| 模式/来源 | 类型 | 状态 | logs | gap mass |','|---|---|---:|---:|---:|']
    for mode,groups in probes['labels'].items():
        for group,v in groups.items():
            lines.append(f'| {mode} | {group} | {v["states"]} | {v["logs"]} | {v["gap_mass"]:.5f} |')
    lines+=['','## 分歧证据','',f'按预定规则选取 {len(stats["cases"])} 个场景×模式，检查 {len(cases["cases"])} 个 seed 案例。',
        '包括全部逐 seed 安全分量分歧及每域每模式正负各三个独立 log 的最大分差场景；不按新结果删例。',
        'decision_cases.json 保留首次共同状态动作分歧、各访问状态的三模型交叉分数及真实结果。首违例时刻缺失标为 NOT_RECORDED。',
        '', '## 下一轮最小验证协议（仅建议，未启动）','',
        '1. 优先排除固定权重解释：在相同 shared 父模型、数据和 500 步预算上比较 NLL + λ Pair，λ=0、0.5、1，三个 seed 全部报告。端点仅在协议一致时复用。',
        '2. 增加监督总强度匹配对照：在训练缓存上固定每 seed 的归一系数，使 λ=0.5 的反馈梯度 RMS 匹配 P1；零范数时报不可识别，不用评测集调系数。',
        '3. 只有固定调权不足，且有限反馈造成的全候选选择偏差得到重复证据，才设计支持范围感知的更新；不自动恢复已失败的 P3。',
        '4. 若差异主要受个别场景/噪声影响，先冻结全部三个 CL_NLL seed，另行规划与 P1 同协议原生完整评测。',
        '5. 成功目标为 rare/common PDM 同步改善；逐项报告 NR/R、NC/DAC/TTC、进度及新增失败，不以加权总分隐藏碰撞。',
        '', '## 论文定位','',
        '不得把纯 selector、Pair+NLL、闭环重排序或一般安全偏好学习本身作为新颖性。比较 RAD-2、CRAFT、DriveDPO、CLOVER 的具体监督来源、候选支持和交互方式。',
        '- RAD-2: https://arxiv.org/abs/2604.15308',
        '- CRAFT: https://arxiv.org/abs/2605.04470',
        '- DriveDPO: https://arxiv.org/abs/2509.17940',
        '- CLOVER: https://arxiv.org/html/2605.15120v2',
        '', '终点：REVIEW_REQUIRED。没有授权自动训练或扩展完整评测。']
    return '\n'.join(lines)+'\n'


def run(args):
    output=h.output_location(args.output,[args.attribution_run,args.snapshot_run])
    torch.set_num_threads(args.threads)
    progress('verify_inputs')
    study=Study(args.attribution_run,args.snapshot_run)
    for path in Path(__file__).parent.glob('*dual_domain*.py'):
        study.ev.use(path)
    output.mkdir(parents=True,exist_ok=True)
    stats=study.statistics()
    c.atomic_json(output/'statistics.json',stats)
    progress('statistics_complete',cases=len(stats['cases']))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260914)
        probes=training_probes(study,progress)
        c.atomic_json(output/'training_probes.json',probes)
        cases=case_analysis(study,stats['cases'],progress)
    c.atomic_json(output/'decision_cases.json',cases)
    with (output/'paired_scenes.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['cohort','mode','policy','scene_id',*h.d.METRICS])
        writer.writeheader()
        for (cohort,mode,policy),metrics in sorted(study.table.items()):
            for scene,values in sorted(metrics.items()):
                writer.writerow(dict(cohort=cohort,mode=mode,policy=policy,scene_id=scene,**values))
    with (output/'review.md').open('x') as stream:
        stream.write(markdown(stats,probes,cases))
    immutable=study.ev.finish()
    c.atomic_json(output/'input_manifest.json',immutable)
    c.atomic_json(output/'report.json',dict(status='PASS',method=h.METHOD,decision='REVIEW_REQUIRED',
        completed_conditions=80,optimizer_steps=0,new_rollouts=0,model_parameters_updated=False,
        outputs={p.name:h.d.artifact(p) for p in sorted(output.iterdir()) if p.is_file()},
        command=vars(args),automatic_training=False,automatic_full_evaluation=False))
    progress('complete',output=str(output))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attribution-run',required=True)
    parser.add_argument('--snapshot-run',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    h.require(1<=args.threads<=16,'CPU threads must be 1..16')
    run(args)


if __name__=='__main__':
    main()
