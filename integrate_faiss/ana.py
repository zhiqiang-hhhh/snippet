import pandas as pd
import matplotlib.pyplot as plt
from math import ceil

# ===============================
# 1. 数据准备
# ===============================
data = [
    # Rows, Method, Params, Train Time(s), Build Time(s), Total Time(s),
    # Search Time(ms), Recall@100, Mem(MB), Compress
    [10000, "HNSW-Flat", "M=32", 0.00, 0.22, 0.22, 0.01, 0.788, 4.9, 1.0],
    [10000, "HNSW-SQ8", "M=32, QT_8bit", 0.00, 1.69, 1.70, 0.06, 0.787, 1.2, 4.0],
    [10000, "HNSW-SQ4", "M=32, QT_4bit", 0.00, 2.57, 2.57, 0.07, 0.703, 0.6, 8.0],
    [50000, "HNSW-Flat", "M=32", 0.00, 7.82, 7.82, 0.02, 0.502, 24.4, 1.0],
    [50000, "HNSW-SQ8", "M=32, QT_8bit", 0.01, 86.39, 86.40, 0.08, 0.505, 6.1, 4.0],
    [50000, "HNSW-SQ4", "M=32, QT_4bit", 0.01, 77.14, 77.15, 0.11, 0.456, 3.1, 8.0],
    [100000, "HNSW-Flat", "M=32", 0.00, 22.28, 22.28, 0.04, 0.378, 48.8, 1.0],
    [100000, "HNSW-SQ8", "M=32, QT_8bit", 0.01, 226.14, 226.15, 0.08, 0.381, 12.2, 4.0],
    [100000, "HNSW-SQ4", "M=32, QT_4bit", 0.01, 207.30, 207.32, 0.11, 0.355, 6.1, 8.0]
]

columns = [
    "Rows", "Method", "Params",
    "Train Time(s)", "Build Time(s)", "Total Time(s)",
    "Search Time(ms)", "Recall@100", "Mem(MB)", "Compress"
]

df = pd.DataFrame(data, columns=columns)

# ===============================
# 2. 保存 CSV 文件
# ===============================
csv_path = "hnsw_results.csv"
df.to_csv(csv_path, index=False)
print(f"CSV 已生成: {csv_path}")

############################################################
# 3 & 4. 分组柱状图绘制函数（更适合小数据量，可读性更好）
############################################################

def plot_grouped_bars(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    title: str,
    ylabel: str,
    out_png: str,
    value_fmt: str = "{:.2f}",
    palette=None,
    legend_loc: str = "upper left",
    legend_anchor=None,
    legend_ncol: int | None = None,
    legend_inside: bool = True,
):
    """绘制分组柱状图并在柱顶添加数值标签。

    Parameters
    ----------
    data : DataFrame
        输入数据。
    x_col : str
        X 轴分类列（例如 Rows）。
    y_col : str
        数值列。
    group_col : str
        分组列（例如 Method）。
    title : str
        图标题。
    ylabel : str
        Y 轴标签。
    out_png : str
        输出图片文件名。
    value_fmt : str
        数值标签格式。
    palette : list[str] | None
        自定义颜色列表。
    """

    # 基础数据准备
    categories = sorted(data[x_col].unique())
    groups = list(data[group_col].unique())
    n_cat = len(categories)
    n_grp = len(groups)

    # 动态图宽：每个类别 0.7 英寸 + 边距
    fig_width = max(6, min(18, 0.7 * n_cat * n_grp + 2.5))
    fig, ax = plt.subplots(figsize=(fig_width, 5.2), dpi=140)

    # 调色板
    if palette is None:
        base_colors = plt.get_cmap("tab10").colors
        if n_grp <= len(base_colors):
            colors = base_colors[:n_grp]
        else:
            # 重复使用，保证有颜色
            times = ceil(n_grp / len(base_colors))
            colors = (base_colors * times)[:n_grp]
    else:
        colors = palette

    total_group_width = 0.82  # 每个类别内部总宽度
    bar_width = total_group_width / n_grp

    x_index = range(n_cat)

    for gi, g in enumerate(groups):
        subset = data[data[group_col] == g]
        # 按类别顺序对齐
        y_vals = [subset[subset[x_col] == c][y_col].values[0] for c in categories]
        # 计算每个分组的偏移
        offsets = [x + (gi - n_grp / 2) * bar_width + bar_width / 2 for x in x_index]
        bars = ax.bar(offsets, y_vals, width=bar_width * 0.92, color=colors[gi], label=g, edgecolor="#333", linewidth=0.5)
        # 添加数值标签
        for b, v in zip(bars, y_vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                value_fmt.format(v),
                ha="center",
                va="bottom",
                fontsize=9,
                rotation=0,
                color="#222",
            )

    ax.set_xticks(list(x_index))
    ax.set_xticklabels([str(c) for c in categories], fontsize=10)
    ax.set_xlabel(x_col, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, pad=10, weight="semibold")

    # 辅助网格仅 y 轴
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)

    # 去掉顶部和右边框
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # 图例放上方横排
    # 图例位置控制：默认放在图内左上角；可配置
    if legend_inside:
        ax.legend(
            loc=legend_loc,
            frameon=True,
            fontsize=9,
            fancybox=True,
            framealpha=0.85,
            borderpad=0.4,
            labelspacing=0.4,
        )
    else:
        ax.legend(
            loc=legend_loc,
            bbox_to_anchor=legend_anchor if legend_anchor else (0.5, 1.08),
            ncol=legend_ncol if legend_ncol else min(4, n_grp),
            frameon=False,
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(out_png)
    print(f"图片已生成(柱状图): {out_png}")
    plt.close(fig)


# Build Time 分组柱状图
plot_grouped_bars(
    df,
    x_col="Rows",
    y_col="Build Time(s)",
    group_col="Method",
    title="Build Time vs Rows (Grouped Bars)",
    ylabel="Build Time (s)",
    out_png="build_time_vs_rows.png",
    value_fmt="{:.2f}",
    legend_loc="upper left",
    legend_inside=True,
)

# Memory Usage 分组柱状图
plot_grouped_bars(
    df,
    x_col="Rows",
    y_col="Mem(MB)",
    group_col="Method",
    title="Memory Usage vs Rows (Grouped Bars)",
    ylabel="Memory (MB)",
    out_png="memory_usage_vs_rows.png",
    value_fmt="{:.1f}",
    legend_loc="upper left",
    legend_inside=True,
)

# 额外：Recall@100 分组柱状图（可帮助直观看质量差异）
plot_grouped_bars(
    df,
    x_col="Rows",
    y_col="Recall@100",
    group_col="Method",
    title="Recall@100 vs Rows (Grouped Bars)",
    ylabel="Recall@100",
    out_png="recall_vs_rows.png",
    value_fmt="{:.3f}",
    legend_loc="upper left",
    legend_inside=True,
)
print("全部柱状图绘制完成。")
