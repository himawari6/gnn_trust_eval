import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# 全局设置为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

# 创建数据
models = ['LogReg', 'SVM', 'RF', 'XGBoost', 'MLP', 'RuleAgg-MLP', 'Our HGNN']
f1_scores = [0.6220, 0.7266, 0.7148, 0.7066, 0.6823, 0.8445, 0.9725]

# 学术风格的图表
fig, ax = plt.subplots(figsize=(4,4))

# 使用更学术的颜色方案
colors = ['#4C72B0'] * (len(models) - 2) +  ["#478D86"] + ['#C44E52']  # 蓝色系 + 红色突出

bars = ax.bar(models, f1_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=0.8)

# 学术风格的标签
for i, (model, score) in enumerate(zip(models, f1_scores)):
    ax.text(i, score + 0.008, f'{score:.3f}', 
            ha='center', va='bottom', fontsize=11)

# 学术风格的设置
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)

ax.set_xlabel('Methods', fontsize=12)
ax.set_ylabel('Macro F1-Score', fontsize=12)
ax.set_title('F1-Score Performance Comparison', fontsize=14, fontweight='bold', pad='20')

plt.xticks(rotation=45, ha='right')
plt.yticks()

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

plt.tight_layout()
# plt.savefig('f1_score_academic.png', dpi=300, bbox_inches='tight')
plt.savefig(f'result/evaluate/figures/f1_score_comparison_with_baselines_bar_chart_at_{current_time}.pdf', bbox_inches='tight')
plt.show()