import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# 设置全局字体为 Times New Roman
plt.rcParams["font.family"] = "Times New Roman"

# 指标名称
metrics = ["Accuracy", "Precision", "Recall", "F1-score"]

# HGNN 和 MLP 的实验结果
hgnn = [0.9796, 0.9684, 0.9610, 0.9646]
mlp  = [0.8265, 0.9137, 0.6143, 0.6823]

x = np.arange(len(metrics))  # x轴位置
width = 0.4  # 柱子宽度

fig, ax = plt.subplots(figsize=(3.5, 2.8))  # 尺寸接近 IEEE 双栏图

# 画柱子
rects1 = ax.bar(x - width * 0.5, mlp, width, label='MLP (w/o aggregation)', color='#e0e0e0', edgecolor='black')
rects2 = ax.bar(x + width * 0.5, hgnn, width, label='HGNN', color='#202020', edgecolor='black')

# 添加数值标注 (保留3位小数)
for rects in [rects1, rects2]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 2),  # 微小偏移
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

# 设置坐标和标题
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel("Score", fontsize=9)
ax.set_ylim(0, 1)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=9)
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15),
          frameon=False, ncol=2)

# 紧凑排版，适合论文
plt.tight_layout()

# 保存为IEEE兼容的矢量格式
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f"result/train/figures/ablation_bar_chart_at_{current_time}.pdf", 
            format='pdf', bbox_inches='tight', pad_inches=0.05)

plt.show()