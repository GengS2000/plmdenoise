"""Reproducible figures and Chinese interpretation of all-feature experiments."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import predict_phenotype as old

HERE = Path(__file__).resolve().parent
GROUPS = old.SITES + ['all6_common', 'full4_common']
LABELS = ['Saliva', 'Tongue', 'Upper interprox.', 'Upper lingual', 'Lower interprox.',
          'Lower lingual', 'All 6 sites (common)', 'Full 4 sites (common)']
FOLDERS = ['results_full_microbiome', 'results_full_microbiome_two_part']


def main():
    data = [pd.read_csv(HERE / f / 'contrasts_all.tsv', sep='\t') for f in FOLDERS]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for row, (d, title) in enumerate(zip(data, ['All CLR features', 'CLR + occurrence (sensitivity)'])):
        for col, contrast in enumerate(['M2-M1', 'M2-M0']):
            ax = axes[row, col]
            s = d[d.contrast == contrast].set_index(['site', 'window'])
            a = np.array([[s.loc[(site, w), 'delta_auc'] for w in old.WINDOWS[:4]] for site in GROUPS])
            im = ax.imshow(a, cmap='RdBu', vmin=-.1, vmax=.1, aspect='auto')
            for i in range(8):
                for j in range(4):
                    ax.text(j, i, f'{a[i,j]:+.3f}', ha='center', va='center', fontsize=9,
                            color='white' if abs(a[i,j]) > .065 else 'black')
            ax.set_xticks(range(4), old.WINDOWS[:4])
            ax.set_yticks(range(8), [f'{label} (n={int(s.loc[(site,"W0"),"n"])})'
                                   for site, label in zip(GROUPS, LABELS)])
            ax.set_xlabel('Prediction window (last day: 0, 2, 5, 9)')
            ax.set_ylabel('Sampling site / joint cohort')
            ax.set_title(f'{title}: {contrast}', fontsize=11)
    fig.colorbar(im, ax=axes, shrink=.6, label='Mean paired difference in macro OVR AUC')
    fig.suptitle('All features, no PCA | 10 x nested stratified 5-fold CV', fontsize=14)
    fig.supxlabel('Source: paired full-feature runs, 2026-10-30 | Visits: day -14 through day 9 | M0 clinical; M1 + raw; M2 + denoised', fontsize=8)
    fig.savefig(HERE / FOLDERS[0] / 'full_feature_deltas.png', dpi=180)
    plt.close(fig)
    d = data[0]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey=True, constrained_layout=True)
    for site, label, ax in zip(GROUPS, LABELS, axes.flat):
        s = d[(d.site == site) & (d.contrast == 'M2-M1')].set_index('window').loc[old.WINDOWS[:4]]
        for model, color in [('M0', '#555555'), ('M1', '#d27a2c'), ('M2', '#176b91')]:
            ax.plot(range(4), s[f'{model}_auc'], marker='o', label=model, color=color)
        ax.set_title(f'{label} (n={s.n.iloc[0]})', fontsize=10)
        ax.set_xticks(range(4), old.WINDOWS[:4])
        ax.set_ylim(.65, .9)
        ax.set_xlabel('Prediction window')
        ax.set_ylabel('Mean repeat macro OVR AUC')
        ax.grid(axis='y', alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, ['M0 clinical', 'M1 clinical + raw', 'M2 clinical + denoised'],
               loc='outside upper center', ncol=3)
    fig.supxlabel('Source: results_full_microbiome, 2026-10-30 | All CLR features | 10 x nested stratified 5-fold CV | Visits through day 9', fontsize=8)
    fig.savefig(HERE / FOLDERS[0] / 'full_feature_curves.png', dpi=180)
    plt.close(fig)

    def get(i, site, win, contrast='M2-M1'):
        t = data[i]
        return t[(t.site == site) & (t.window == win) & (t.contrast == contrast)].iloc[0]
    def describe(i, site, win):
        a, b = get(i, site, win), get(i, site, win, 'M2-M0')
        return (f'M0={a.M0_auc:.3f}，M1={a.M1_auc:.3f}，M2={a.M2_auc:.3f}；'
                f'M2−M1={a.delta_auc:+.3f}（条件区间 {a.ci_low_conditional:+.3f} 至 '
                f'{a.ci_high_conditional:+.3f}），{int(a.positive_repeats)}/10次划分为正，'
                f'{int(a.seed_positive)}/3个seed平均增益为正；M2−M0={b.delta_auc:+.3f}'
                f'（条件区间 {b.ci_low_conditional:+.3f} 至 {b.ci_high_conditional:+.3f}）。')
    parts = ['# 全特征预测：去噪是否改善临床＋微生物组学的day14分型？',
             '', '日期：2026-10-30。原PCA分析及其输出未修改。', '',
             '## 结论', '',
             '取消PCA后，下颌邻面出现跨窗口的去噪增益线索；但多部位联合模型并未稳定超过临床基线。'
             '本轮支持部位特异的探索性信号，不支持“去噪普遍提高预测”或“已经证明临床增量”。', '',
             '## 本轮固定方法', '',
             '- W0/W1/W2/W3分别截止day0/2/5/9。完全排除day14和day21微生物及临床特征。',
             '- 使用现有文件内全部保留菌×窗口访视，未做PCA、按标签筛菌或事后删去负结果。'
             '“全部”指现有数据的全部菌特征，不包括已在上游过滤的菌、绝对菌量或未测定功能。',
             '- 主分析沿用raw_count+0.5的CLR与去噪total的中心化。所有访视展平；缺访视只用本人窗口内均值补齐。',
             '- 双组线性岭分类：对三类one-hot目标进行岭拟合，argmax分类；AUC用类别得分逐类计算后平均，不将得分冒充概率。'
             '对偶核计算与使用全部列的原始空间岭回归等价，没有截断秩或选取前几个方向。',
             '- 每个训练折内标准化临床变量；微生物逐列中心化，不对每种低丰度菌作单位方差放大。'
             '每个部位/通道按训练集平均平方行范数归一化，多部位核取均值，避免菌数多的部位自动占优。',
             '- 外层10次重复分层5折，内层分层3折。所有模型/seed共用相同受试者划分。'
             '内层按平均macro AUC选择alpha∈{0.01,0.1,1,10}与微生物权重∈{0,0.1,1,10}。'
             'M0只允许权重0；M1/M2均允许退回临床模型，网格完全相同。',
             '- 每次重复获得每人一次折外预测，主指标为10次完整折外macro OVR AUC的均值，'
             '而非将折、重复或seed当成独立样本。保存准确率、balanced accuracy、混淆矩阵、全部折外得分及选择参数。',
             '- 全六部位交集n=23；前四部位（唾液、舌背、上颌邻面、上颌舌侧）交集n=35。'
             '联合模型与单部位模型因队列不同，不能直接解释AUC高低为部位优势。',
             '- 运行前配置指定全六部位W3的M2−M1为主要比较；这是本轮分析约定，不是独立注册或外部验证。'
             '下颌邻面等部位的强调来自本轮筛查，必须视为探索性发现。', '',
             '## 全丰度特征结果', '',
             '**主要比较：六部位联合W3，n=23。** ' + describe(0, 'all6_common', 'W3'), '',
             '**下颌邻面，n=26：本轮最值得独立复核的线索。**', '']
    for w in old.WINDOWS[:4]:
        parts.append(f'- {w}：' + describe(0, 'lower_jaw_interproximal', w))
    parts += ['', '**其他结果必须一起报告。**']
    for i, name in enumerate(['丰度主分析', '丰度＋有无敏感性分析']):
        t = data[i][data[i].contrast == 'M2-M1']
        b = data[i][data[i].contrast == 'M2-M0']
        single = t[t.site.isin(old.SITES)]
        parts.append(f'- {name}：六个单部位×四窗口中，M2−M1为正{int((single.delta_auc>0).sum())}/24格，'
                     f'简单平均Δ={single.delta_auc.mean():+.3f}；包括联合模型共32格，'
                     f'M2−M0为正{int((b.delta_auc>0).sum())}/32格。格子高度相关，不能用此计数做显著性检验。')
    parts += ['', '## 补充丰度之外的有无信息', '',
              '导出说明明确total为去噪log positive abundance，另有absence_probability。'
              '原PCA方案及本轮丰度主分析没有显式纳入后者。敏感性分析增加独立通道：'
              '原始为1(raw_count>0)，去噪为1−absence_probability；三个seed均值也在概率尺度平均。'
              '每个部位的CLR与有无通道等权，缩放仍只用训练受试者。'
              '该分析在看到部分主分析结果后，基于文件定义补充，不能当成预注册确认性检验。', '',
              '六部位W3：' + describe(1, 'all6_common', 'W3'), '',
              '下颌邻面W2：' + describe(1, 'lower_jaw_interproximal', 'W2'), '',
              '## 不确定性与解读边界', '',
              '- 区间为2000次按标签分层、受试者配对bootstrap，对固定折外得分计算10次重复AUC差的均值。'
              '这只反映给定拟合结果下受试者抽样变化，没有重新拟合训练、内层调参或去噪器；'
              '折外预测也共享训练集，因此不是完整泛化误差的置信保证。',
              '- 未做32格及敏感性分析的多重比较校正，不把条件区间不跨零称为确认性的统计显著。'
              '三个去噪seed共用受试者与标签，方向一致只证明优化随机性下的稳定程度。',
              '- 去噪为既有全队列预计算结果，仍有转导式信息共享；下游缩放与调参已嵌套，不等于去噪器实现了完全折外训练。'
              '因此尚不能证明对新受试者的归纳式泛化增益。',
              '- 本轮临床基线为线性岭模型，与旧随机森林OOB结果不能直接比较。'
              '即使超过本轮M0，也不代表超过更强的临床模型；确认研究应保留强临床基线。',
              '- 本轮同时改变了特征表示、分类器和验证流程，不能把与旧结果的差别全部归因于取消PCA。'
              '若要单独检验PCA影响，需要在完全相同的模型及嵌套划分中另作PCA消融。',
              '- 下颌邻面M2−M1为正，并不自动证明临床增量：必须同时考察M2−M0。'
              '本轮该部位后者的条件区间仍覆盖零。',
              '- 全丰度岭模型保留全部列，但只拟合线性可加作用；不代表检验过全部非线性相互作用。'
              '没有正增益不能证明没有生物学信息。', '',
              '## 下一步应如何形成可发表证据', '',
              '优先把下颌邻面W2作为独立验证的候选，而不是继续在当前队列上筛选模型直到出现显著。'
              '冻结菌特征表示、窗口、模型及调参规则；用新增或独立受试者验证M2−M1和M2−M0。'
              '若只能使用现有队列，需要首先实现去噪器训练折拟合、测试受试者仅用预测截止日前数据推断，'
              '再进行包含全部拟合步骤的不确定性评估，并同时比较强临床基线。', '',
              '## 文件与复现', '',
              '- `predict_full_microbiome.py`：完整分析；`summarize_full_microbiome.py`：图与本报告。',
              '- `results_full_microbiome/`：丰度主分析；`results_full_microbiome_two_part/`：双通道敏感性分析。',
              '- 每目录的`config.json`记录固定网格；`metrics_all.tsv`记录各重复指标；'
              '`oof_all.tsv`保存全部受试者折外得分；`selection_all.tsv`保存内层选择；'
              '`contrasts_all.tsv`保存配对增量、条件区间和seed一致性。',
              '- 主分析图：`results_full_microbiome/full_feature_deltas.png`与`full_feature_curves.png`。'
              '图中数据来源为本轮两个结果目录，研究窗口截止day9。', '',
              '- `verify_full_microbiome.py`验证通过：对偶/原始空间岭解等价、AUC计算、留出样本不影响训练缩放、'
              '快速bootstrap与显式重采样一致；两套分析共121920条折外预测逐项核验，各自1920条重复指标与96条增量记录完整，'
              '同一受试者在各模型中使用同一折，两种表示的临床基线完全一致。', '',
              '```powershell',
              'python predict_full_microbiome.py --repeats 10',
              'python predict_full_microbiome.py --repeats 10 --representation two_part --out results_full_microbiome_two_part',
              'python summarize_full_microbiome.py', '```', '',
              '方法参考：[scikit-learn嵌套交叉验证说明](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)。', '']
    (HERE / '全微生物特征预测_结果解读.md').write_text('\n'.join(parts), encoding='utf-8')


if __name__ == '__main__':
    main()
