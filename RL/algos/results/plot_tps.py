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
BASELINE = (0, 0, 0)
picks = {
    "baseline": {1: (0, 0, 0), 4: (0, 0, 0), 8: (0, 0, 0), 16: (0, 0, 0)},
    "reference": {1: (3, 4, 8), 4: (3, 4, 8), 8: (3, 4, 8), 16: (3, 4, 8)},
    "lookup_table": {1: (3, 4, 8), 4: (3, 4, 8), 8: (3, 4, 8), 16: (3, 2, 4)},
    "doubly_robust": {1: (3, 4, 8), 4: (3, 4, 8), 8: (3, 4, 8), 16: (3, 2, 4)},
    "linucb": {1: (3, 4, 8), 4: (3, 4, 8), 8: (3, 2, 4), 16: (3, 2, 4)},
    "gbt": {1: (5, 4, 16), 4: (5, 4, 16), 8: (3, 4, 8), 16: (5, 2, 8)},
    "thompson_sampling": {1: (5, 4, 16), 4: (3, 1, 4), 8: (1, 1, 2), 16: (1, 4, 2)},
    "bcq": {1: (3, 4, 8), 4: (0, 0, 0), 8: (0, 0, 0), 16: (0, 0, 0)},
    "cql": {1: (5, 4, 8), 4: (0, 0, 0), 8: (0, 0, 0), 16: (0, 0, 0)},
    "ddpg": None,  # never trained -- no config was ever picked, drawn as a void column
}
order = ["baseline", "reference", "lookup_table", "doubly_robust", "linucb",
         "gbt", "thompson_sampling", "bcq", "cql", "ddpg"]
colors = {
    "baseline": "#9aa3ad", "reference": INK, "lookup_table": "#2f6f9f", "doubly_robust": "#8a5ac9",
    "linucb": "#3f9142", "gbt": "#c98a1f", "thompson_sampling": "#c2588f",
    "bcq": "#8a6a4a", "cql": "#5a636e",
}
batch_sizes = [1, 4, 8, 16]
YLO, YHI = 80, 172

fig, axes = plt.subplots(1, 4, figsize=(19.5, 5.9), facecolor="white", sharey=True)
fig.subplots_adjust(wspace=0.08, left=0.04, right=0.99, top=0.82, bottom=0.26)

for ax, bs in zip(axes, batch_sizes):
    ax.set_facecolor("white")
    for y in [90, 100, 110, 120, 130, 140, 150, 160]:
        ax.axhline(y, color=GRID, linewidth=0.8, zorder=0)

    xs = list(range(len(order)))
    for x, name in zip(xs, order):
        if name == "ddpg":
            ax.bar(x, YHI - YLO, bottom=YLO, width=0.62, facecolor="none",
                    edgecolor=INK_FAINT, linewidth=1.1, linestyle=(0, (3, 2)),
                    hatch="////", zorder=3)
            ax.text(x, (YLO + YHI) / 2, "never\ntrained", ha="center", va="center",
                     fontsize=7.6, color=INK_FAINT, rotation=90, fontweight="bold")
            continue
        cfg = picks[name][bs]
        h = SPEED[(bs, *cfg)]
        ax.bar(x, h, color=colors[name], width=0.62, zorder=3, edgecolor="white", linewidth=0.6)
        label = f"{cfg[0]},{cfg[1]},{cfg[2]}"
        weight = "bold" if cfg == REFERENCE else "normal"
        tcolor = INK_MUTED if cfg == REFERENCE else INK
        ax.text(x, h + 2.2, label, ha="center", va="bottom", fontsize=7.3,
                 family="monospace", color=tcolor, fontweight=weight)

    ax.set_xticks(xs)
    ax.set_xticklabels(order, rotation=55, ha="right", fontsize=8.3, color=INK_MUTED)
    ax.set_title(f"batch size {bs}", fontsize=12, fontweight="bold", color=INK, pad=10)
    ax.set_ylim(YLO, YHI)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(INK_FAINT)
    ax.tick_params(colors=INK_MUTED, labelsize=8.5)

axes[0].set_ylabel("throughput (tok/s)", color=INK_MUTED, fontsize=10.5)

fig.suptitle("Live throughput per algorithm's pick -- labels are the (steps, topk, draft) combo chosen",
             fontsize=14, fontweight="bold", color=INK, x=0.02, ha="left", y=0.975)

out_path = "tps_overview.png"
fig.savefig(out_path, dpi=170, facecolor="white", bbox_inches="tight", pad_inches=0.25)
print(f"saved {out_path}")
