import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime

# 设置符合IEEE的格式参数
# IEEE通常推荐使用Times New Roman或类似serif字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学字体设置

# 设置字体大小 - IEEE通常要求8-12pt
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.titlesize'] = 10

# 设置线宽 - IEEE通常要求0.5pt或更粗
plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['axes.linewidth'] = 0.8  # 坐标轴线宽
plt.rcParams['grid.linewidth'] = 0.5  # 网格线宽

loss_values = [
    74.0025, 22.5860, 39.0989, 19.2014, 17.1706, 20.2966, 17.7100, 18.9605, 14.0774, 20.0992,
    11.1276, 15.9764, 27.3065, 21.9277, 16.9577, 15.3654, 13.4055, 13.6514, 13.4933, 13.7466,
    25.5264, 11.4359, 5.0171, 7.7046, 7.6193, 6.4156, 20.8001, 7.9091, 10.3754, 8.4018,
    7.2713, 8.1227, 6.4612, 4.9996, 10.3656, 5.5412, 6.2389, 5.2192, 5.1100, 5.1768,
    4.2774, 2.9446, 2.9922, 4.3502, 1.4487, 2.2548, 2.2557, 2.1692, 1.6046, 2.8477,
    1.6515, 1.3828, 2.3907, 1.7284, 1.1885, 1.4548, 1.2420, 1.0440, 1.5316, 1.7547,
    1.0016, 1.4143, 0.9599, 1.0381, 1.2431, 0.5789, 0.7485, 1.1320, 0.6952, 1.3705,
    1.5822, 0.3629, 0.2726, 0.5303, 0.5752, 0.5343, 0.5508, 0.6444, 0.4996, 0.4370,
    0.3727, 0.0918, 0.4028, 0.5697, 0.4277, 0.2651, 0.0903, 0.1957, 0.3849, 0.2413,
    0.0895, 0.0902, 0.2854, 0.0890, 0.2504, 0.0886, 0.2519, 0.0883, 0.0882, 0.0882
]

# 创建图表 - IEEE通常要求图形宽度为3.5英寸(单栏)或7英寸(双栏)
# 这里使用单栏宽度
fig, ax = plt.subplots(figsize=(3.5, 2.5))  # 宽度3.5英寸，高度适当

# 绘制训练和验证损失曲线
ax.plot(range(1, 101), loss_values, 
        linewidth=1.0, color='black', marker='o', markersize=2, label='Training Loss')

# 添加标题和标签
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('Epochs', fontsize=10)
ax.set_ylabel('Loss', fontsize=10)

# 设置网格
ax.grid(True, linestyle=':', alpha=0.7)

# 添加图例 - 位置调整以避免遮挡重要数据
ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')

# 设置坐标轴范围
ax.set_xlim(0, 100)
ax.set_ylim(0, max(loss_values) * 1.1)

# 优化布局
plt.tight_layout(pad=0.5)

# 保存为IEEE兼容的矢量格式
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f"result/train/figures/loss_curve_at_{current_time}.pdf", format='pdf', bbox_inches='tight', pad_inches=0.02)
