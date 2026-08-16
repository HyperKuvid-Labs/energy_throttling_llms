import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# static png for the readme -- source numbers copied verbatim from the
# live-validated tables in README.md, not recomputed here

INK = "#1c2024"
INK_MUTED = "#626b76"
INK_FAINT = "#97a0aa"
AMBER = "#c97a1f"
TEAL = "#178363"
CORAL = "#c23f3f"
GRID = "#e4e0d8"
BAND = "#178363"

batch_sizes = [1, 4, 8, 16]
reference = [0.5323, 0.4975, 0.4653, 0.2945]
algos = [
    ("lookup_table", [0.5323, 0.4975, 0.4653, 0.3855], "o", "-"),
    ("doubly_robust", [0.5323, 0.4975, 0.4653, 0.3855], "^", "--"),
    ("linucb", [0.5323, 0.4975, 0.4113, 0.3855], "s", "-"),
    ("gbt", [0.5173, 0.4290, 0.4653, 0.2555], "D", "-"),
    ("thompson_sampling", [0.5173, 0.4177, 0.3413, 0.2971], "P", "-"),
    ("bcq", [0.5323, 0.0000, 0.0000, 0.0000], "v", "-"),
    ("cql", [0.4483, 0.0000, 0.0000, 0.0000], "X", "-"),
]
palette = ["#2f6f9f", "#8a5ac9", "#3f9142", "#c98a1f", "#c2588f", "#8a6a4a", "#5a636e"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6), facecolor="white")
fig.subplots_adjust(wspace=0.32, left=0.06, right=0.98, top=0.86, bottom=0.14)

# ---- left: reward trajectory per algorithm across batch sizes ----
ax1.set_facecolor("white")
for y in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    ax1.axhline(y, color=GRID, linewidth=0.8, zorder=0)

ax1.plot(batch_sizes, reference, color=INK, linewidth=2.6, linestyle=(0, (5, 2)),
          marker="o", markersize=6, label="reference (3,4,8)", zorder=5)

for (name, vals, marker, ls), color in zip(algos, palette):
    ax1.plot(batch_sizes, vals, color=color, linewidth=1.9, linestyle=ls,
              marker=marker, markersize=6, alpha=0.9, label=name, zorder=4)

ax1.set_xticks(batch_sizes)
ax1.set_xticklabels([str(b) for b in batch_sizes])
ax1.set_xlabel("batch size", color=INK_MUTED, fontsize=10)
ax1.set_ylabel("live reward", color=INK_MUTED, fontsize=10)
ax1.set_title("Reward per algorithm, across batch sizes", loc="left",
               fontsize=13, fontweight="bold", color=INK, pad=14)
ax1.set_ylim(-0.02, 0.58)
for spine in ["top", "right"]:
    ax1.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax1.spines[spine].set_color(INK_FAINT)
ax1.tick_params(colors=INK_MUTED, labelsize=9)
ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False,
           fontsize=8.3, labelcolor=INK_MUTED, handlelength=1.6, columnspacing=1.2)
ax1.text(0.0, 0.585, "cql and bcq collapse to the non-speculative baseline (reward 0) past bs=1",
          fontsize=7.6, color=INK_FAINT, transform=ax1.transData, ha="left")

# ---- right: energy band gauge, bs=16, mlp bandit trials ----
ax2.set_facecolor("white")
lo, hi = 0.90, 1.03
band_lo, band_hi = 0.95, 0.98
ax2.axhspan(band_lo, band_hi, xmin=0, xmax=1, color=BAND, alpha=0.14, zorder=0)
ax2.axhline(band_lo, color=TEAL, linewidth=1, alpha=0.5, linestyle=":")
ax2.axhline(band_hi, color=TEAL, linewidth=1, alpha=0.5, linestyle=":")

ref_util = [0.997, 1.005, 1.007]
policy_util = [0.966, 0.974, 0.979]
xs_ref = [0.32] * 3
xs_policy = [0.68] * 3
ax2.scatter(xs_ref, ref_util, s=110, color=CORAL, edgecolor="white", linewidth=1.4,
            zorder=5, label="reference (3,4,8)")
ax2.scatter(xs_policy, policy_util, s=110, color=TEAL, edgecolor="white", linewidth=1.4,
            zorder=5, label="policy (3,2,4)")

ax2.set_xlim(0, 1)
ax2.set_ylim(lo, hi)
ax2.set_xticks([0.32, 0.68])
ax2.set_xticklabels(["reference\n(3,4,8)", "policy\n(3,2,4)"], fontsize=9.5, color=INK)
ax2.set_ylabel("energy utilization  (avg power / 80W cap)", color=INK_MUTED, fontsize=9.5)
ax2.set_title("Why bs=16 changes: 3 live trials, MLP bandit", loc="left",
               fontsize=13, fontweight="bold", color=INK, pad=14)
ax2.text(0.5, band_hi + 0.001, "95-98% target band", ha="center", va="bottom",
          fontsize=8.3, color=TEAL, fontweight="bold")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
for spine in ["top", "right"]:
    ax2.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax2.spines[spine].set_color(INK_FAINT)
ax2.tick_params(colors=INK_MUTED, labelsize=9)

fig.suptitle("EAGLE3 speculative decoding -- live-validated results (RTX 4060 Laptop, 8GB)",
             fontsize=14.5, fontweight="bold", color=INK, x=0.02, ha="left", y=0.985)

out_path = "results_overview.png"
fig.savefig(out_path, dpi=170, facecolor="white", bbox_inches="tight", pad_inches=0.25)
print(f"saved {out_path}")
