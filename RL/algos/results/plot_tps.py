import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# tokens/sec per algorithm's actual pick, per batch size -- speed_tok_s values
# copied verbatim from RL/algos/results/live_validate.jsonl, matched against
# each algorithm's pick in RL/algos/results/picks.json

INK = "#1c2024"
INK_MUTED = "#626b76"
INK_FAINT = "#97a0aa"
GRID = "#e4e0d8"

# speed_tok_s lookup: (batch_size, steps, topk, draft) -> tok/s, from live_validate.jsonl
SPEED = {
    (1, 0, 0, 0): 97.254, (1, 3, 4, 8): 156.197, (1, 5, 4, 16): 150.956, (1, 5, 4, 8): 139.058,
    (4, 0, 0, 0): 95.042, (4, 3, 4, 8): 142.638, (4, 3, 1, 4): 129.574, (4, 5, 4, 16): 134.803,
    (8, 0, 0, 0): 93.515, (8, 3, 4, 8): 134.214, (8, 3, 2, 4): 125.921, (8, 1, 1, 2): 116.676,
    (16, 0, 0, 0): 90.793, (16, 3, 4, 8): 113.417, (16, 3, 2, 4): 117.784, (16, 1, 4, 2): 107.542, (16, 5, 2, 8): 102.643,
}

REFERENCE = (3, 4, 8)
picks = {
    "reference": {1: (3, 4, 8), 4: (3, 4, 8), 8: (3, 4, 8), 16: (3, 4, 8)},
    "lookup_table": {1: (3, 4, 8), 4: (3, 4, 8), 8: (3, 4, 8), 16: (3, 2, 4)},
    "doubly_robust": {1: (3, 4, 8), 4: (3, 4, 8), 8: (3, 4, 8), 16: (3, 2, 4)},
    "linucb": {1: (3, 4, 8), 4: (3, 4, 8), 8: (3, 2, 4), 16: (3, 2, 4)},
    "gbt": {1: (5, 4, 16), 4: (5, 4, 16), 8: (3, 4, 8), 16: (5, 2, 8)},
    "thompson_sampling": {1: (5, 4, 16), 4: (3, 1, 4), 8: (1, 1, 2), 16: (1, 4, 2)},
    "bcq": {1: (3, 4, 8), 4: (0, 0, 0), 8: (0, 0, 0), 16: (0, 0, 0)},
    "cql": {1: (5, 4, 8), 4: (0, 0, 0), 8: (0, 0, 0), 16: (0, 0, 0)},
}
order = ["reference", "lookup_table", "doubly_robust", "linucb", "gbt", "thompson_sampling", "bcq", "cql"]
colors = {
    "reference": INK, "lookup_table": "#2f6f9f", "doubly_robust": "#8a5ac9",
    "linucb": "#3f9142", "gbt": "#c98a1f", "thompson_sampling": "#c2588f",
    "bcq": "#8a6a4a", "cql": "#5a636e",
}
batch_sizes = [1, 4, 8, 16]

fig, axes = plt.subplots(1, 4, figsize=(16.5, 5.6), facecolor="white", sharey=True)
fig.subplots_adjust(wspace=0.08, left=0.045, right=0.99, top=0.82, bottom=0.24)

for ax, bs in zip(axes, batch_sizes):
    ax.set_facecolor("white")
    for y in [90, 100, 110, 120, 130, 140, 150, 160]:
        ax.axhline(y, color=GRID, linewidth=0.8, zorder=0)

    baseline_speed = SPEED[(bs, 0, 0, 0)]
    ax.axhline(baseline_speed, color=INK_FAINT, linewidth=1.3, linestyle=(0, (4, 2)), zorder=1)

    xs = list(range(len(order)))
    heights = [SPEED[(bs, *picks[name][bs])] for name in order]
    bar_colors = [colors[name] for name in order]
    bars = ax.bar(xs, heights, color=bar_colors, width=0.62, zorder=3,
                   edgecolor="white", linewidth=0.6)

    for x, name, h in zip(xs, order, heights):
        cfg = picks[name][bs]
        label = f"{cfg[0]},{cfg[1]},{cfg[2]}"
        weight = "bold" if cfg == REFERENCE else "normal"
        color = INK_MUTED if cfg == REFERENCE else INK
        ax.text(x, h + 2.2, label, ha="center", va="bottom", fontsize=7.3,
                 family="monospace", color=color, fontweight=weight, rotation=0)

    ax.set_xticks(xs)
    ax.set_xticklabels(order, rotation=55, ha="right", fontsize=8.3, color=INK_MUTED)
    ax.set_title(f"batch size {bs}", fontsize=12, fontweight="bold", color=INK, pad=10)
    ax.set_ylim(80, 172)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(INK_FAINT)
    ax.tick_params(colors=INK_MUTED, labelsize=8.5)

axes[0].set_ylabel("throughput (tok/s)", color=INK_MUTED, fontsize=10.5)
axes[0].text(-0.9, 84, "dashed line = non-speculative baseline",
             fontsize=7.6, color=INK_FAINT, rotation=0, ha="left")

fig.suptitle("Live throughput per algorithm's pick -- labels are the (steps, topk, draft) combo chosen",
             fontsize=14, fontweight="bold", color=INK, x=0.02, ha="left", y=0.975)

out_path = "tps_overview.png"
fig.savefig(out_path, dpi=170, facecolor="white", bbox_inches="tight", pad_inches=0.25)
print(f"saved {out_path}")
