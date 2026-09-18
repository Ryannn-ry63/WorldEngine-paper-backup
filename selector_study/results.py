"""Export audited formal results without relabeling their training sources."""
import csv
import statistics
from pathlib import Path
from .common import read, artifact, verify, write

BLOCKS = ('open_navtest', 'open_rare', 'full_NR', 'full_R')
METRICS = ('no_at_fault_collisions', 'drivable_area_compliance', 'ego_progress',
           'time_to_collision_within_bound', 'comfort', 'score')
SHORT = ('NC', 'DAC', 'EP', 'TTC', 'Comfort', 'PDM')


def table_rows(data):
    rows = []
    for arm in ('scalar_v3', 'P1', 'CL_NLL', 'MID'):
        count = 1 if arm == 'scalar_v3' else 3
        numeric = []
        for seed in range(count):
            policy = arm if count == 1 else f'{arm}_seed{seed}'
            ade = data['ade_fde'][policy]
            values = [ade['ade_4s'], ade['fde_4s']]
            for block in BLOCKS:
                m = data['blocks'][block]['arms'][arm]['seed_means'][seed]
                values.extend(m[k] for k in METRICS)
            values.append(data['blocks']['full_R']['arms'][arm]['seed_means'][seed]['success'])
            numeric.append(values)
            name = 'V3_rarelog_seed0' if count == 1 else f'V3rarelog_to_mixed_{arm}_optseed{seed}'
            note = 'init=V3 rare_tuned seed0; ' + ('no continuation; fixed reference' if count == 1 else
                f'continuation=mixed dense + closed feedback; method={arm}; optimization_seed={seed}; acquisition_seed=0')
            rows.append([name, note + '; eval_noise=formal_navtest_seed0; historically exposed; ADE/FDE=4s,m', *values])
        if count > 1:
            for label, function in [('mean', statistics.mean), ('std', statistics.stdev)]:
                values = [function(column) for column in zip(*numeric)]
                rows.append([f'V3rarelog_to_mixed_{arm}_3optseed_{label}',
                    f'init=V3 rare_tuned seed0; continuation=mixed dense + closed feedback; {label} across optimization seeds 0,1,2; acquisition_seed=0; sample SD uses ddof=1; not independent pipelines; eval_noise=formal_navtest_seed0; ADE/FDE=4s,m', *values])
    return rows


def export(workspace, paths):
    source = Path(paths['review_dir'])
    output = workspace / 'reports/20260916_existing_results'
    data = read(source / 'comparison.json')
    validation = read(source / 'verification.json')
    if data['status'] != 'PASS' or validation['status'] != 'PASS':
        raise RuntimeError('Missing prior acceptance audit')
    rechecked = []
    for item in validation['source_artifacts']:
        if item['path'].endswith(('/summary.json', '/inputs.json', '/paired_scenes.csv')):
            verify(item)
            rechecked.append(item)
    output.mkdir(parents=True, exist_ok=True)
    header = ['Model Name', 'Note', 'ADE', 'FDE'] + [f'{b}:{k}' for b in BLOCKS for k in SHORT] + ['CL-R:Success Rate']
    rows = table_rows(data)
    if len(header) != 29 or any(len(row) != 29 for row in rows):
        raise RuntimeError('Wrong table width')
    for percent in (False, True):
        name = 'formal_29cols' + ('_percent' if percent else '') + '.tsv'
        with (output / name).open('w', newline='') as stream:
            writer = csv.writer(stream, delimiter='\t')
            writer.writerow(header)
            for row in rows:
                writer.writerow(row[:2] + [f'{value * (100 if percent and index >= 4 else 1):.9f}'
                                          for index, value in enumerate(row[2:], start=2)])
    lines = ['# 2026-09-16：现有正式结果复核', '',
        'P1 保留为有效但尚未定稿的基线；MID 不升为主方法。本次导出不包含新增训练或评测。', '',
        'PDM 为百分制。三个优化 seed 共用 V3 rarelog seed0 起点和 acquisition seed0。', '',
        '| 评测 | V3 | P1 | CL_NLL | MID |', '|---|---:|---:|---:|---:|']
    for block in (*BLOCKS, 'common_NR', 'common_R'):
        arms = data['blocks'][block]['arms']
        lines.append('| ' + block + ' | ' + ' | '.join(f"{arms[a]['mean']['score'] * 100:.3f}"
                     for a in ('scalar_v3', 'P1', 'CL_NLL', 'MID')) + ' |')
    lines += ['', '## 结论与边界', '',
        '- P1 完整 NR/R PDM 比固定 V3 高 1.444/1.663 分；success 下降 0.926/0.579 个百分点，DAC 下降 1.852/1.736 个百分点。',
        '- MID 没有形成稳定优于 P1 的折中。development58 的收益不能证明独立泛化。',
        '- P1/MID/CL_NLL 都是 rarelog 初始化后在 mixed dense + 闭环反馈上续训，不能填成三个独立数据训练组。',
        '- 三 seed 是条件于固定初始化和采集的优化重复。common64 和 remainder230 均有历史暴露。',
        '- 先前 log-cluster 区间为条件于固定采集和噪声的探索性区间；跨零不代表等效或不退化。', '',
        '## 表格约定', '',
        '29 列包含模型、Note、ADE/FDE（4秒，米）、四个六指标块和 R Success=NC&DAC。navtest 为 12,146 个实际 token，rare 和完整闭环为 288 场景。common64 是辅助评测。', '',
        '## 核验范围', '',
        '先前验收检查了 215 个产物哈希及 144,518 条原始指标记录（含复用引用）。本次重新核验其 summary、inputs 和 paired CSV 身份，从保存的 comparison 导出逐 seed、均值、样本标准差；不重跑模拟器或声称重新核验全部 checkpoint。', '',
        '## 下一步', '',
        '从未接受闭环训练的 V3 开始 D1–D4，再比较同反馈目标、通过动作干预分析 DAC 失败。独立验证后冻结方法，补三类训练数据主表。']
    (output / 'review.md').write_text('\n'.join(lines) + '\n')
    inventory = [dict(regime=r, V3='historical; compatibility/compute audit required',
                     matched_dense_continuation='NOT_RUN', matched_feedback_continuation='NOT_RUN',
                     paper_main_table_complete=False) for r in ('common', 'rarelog', 'rarerollout')]
    write(output / 'inventory.json', dict(training_regimes=inventory, current_cross_source_methods=['P1', 'CL_NLL', 'MID']))
    write(output / 'provenance.json', dict(comparison=artifact(source / 'comparison.json'),
        previous_verification=artifact(source / 'verification.json'), rechecked_sources=rechecked,
        outputs=[artifact(output / n) for n in ('formal_29cols.tsv', 'formal_29cols_percent.tsv', 'review.md', 'inventory.json')]))
    return dict(status='PASS', output=str(output), rows=len(rows), columns=29)
