# %%
import matplotlib.pyplot as plt

# --- Data ---
total = 120
counts = {"Base": 38, "Hybrid": 76, "Thinking": 97}

acc_base = counts["Base"] / total * 100
acc_hybrid = counts["Hybrid"] / total * 100
acc_thinking = counts["Thinking"] / total * 100

gap_bt = acc_thinking - acc_base          # Base → Thinking gap (in %)
closed = acc_hybrid - acc_base            # Closed by Hybrid (in %)
closure_pct = (closed / gap_bt * 100) if gap_bt else 0.0

# --- Figure ---
fig, ax = plt.subplots(figsize=(14, 4), dpi=300)

# Colors (warm sophisticated palette)
neutral = "#264653"         # deep teal-gray base
hybrid_color = "#1A6A60"    # darker sage green
thinking_color = "#8B4B6F"  # deep mauve
bw = 0.6

# Bars
x_base, x_hybrid, x_think = 0, 1, 2
ax.bar(x_base, acc_base, width=bw, color=neutral)

# Hybrid = base portion (neutral) + closed portion (teal)
ax.bar(x_hybrid, acc_base, width=bw, color=neutral)
ax.bar(x_hybrid, closed, bottom=acc_base, width=bw, color=hybrid_color)

# Thinking = base portion (neutral) + full gap (orange)
ax.bar(x_think, acc_base, width=bw, color=neutral)
ax.bar(x_think, gap_bt, bottom=acc_base, width=bw, color=thinking_color)

# Figure style and background
fig.patch.set_facecolor('#FFFFFF')
ax.set_facecolor('#FFFFFF')

# Axes / styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.weight'] = 'regular'

# Title and labels with larger fonts
ax.set_title("Hybrid Model Evaluation - GSM8K", pad=25, fontsize=16, fontweight='bold', color='#333333')
ax.set_xlabel("", labelpad=15, fontsize=14, color='#333333', fontweight='regular')  # removed Model label
ax.set_ylabel("Accuracy (%)", labelpad=15, fontsize=14, color='#333333', fontweight='bold')
ax.set_ylim(0, 90)

# Tick labels
ax.set_xticks([x_base, x_hybrid, x_think])
ax.set_xticklabels(["Base", "Hybrid", "Thinking"], fontsize=13, color='#333333', fontweight='regular')
ax.tick_params(axis='both', colors='#333333', labelsize=12)

# Add border around the main figure and grid
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('#DDDDDD')
    spine.set_linewidth(1.0)
ax.grid(axis="y", linestyle='-', linewidth=0.3, alpha=0.2, color='#444444')

# Top-of-bar labels
top_labels = [
    (x_base, acc_base, "Base"),
    (x_hybrid, acc_hybrid, "Hybrid"),
    (x_think, acc_thinking, "Thinking"),
]
for xi, acc, name in top_labels:
    ax.text(xi, acc + 1.2, f"{acc:.1f}% ({counts[name]}/{total})",
            ha="center", va="bottom", fontsize=13.5, color='#333333')

# Inside colored segments
ax.text(x_hybrid, acc_base + closed / 2,
        f"+{closed:.1f}%\n({closure_pct:.0f}% of Gap)",
        ha="center", va="center", color="white", fontsize=13.5, fontweight='regular')
ax.text(x_think, acc_base + gap_bt / 2,
        f"{gap_bt:.1f}% Gap",
        ha="center", va="center", color="white", fontsize=13.5, fontweight='regular')

# Remove all padding/margins
plt.tight_layout(pad=0)
fig.savefig("model_comparison_gap.png", bbox_inches='tight', pad_inches=0)
fig.savefig("model_comparison_gap.pdf", bbox_inches='tight', pad_inches=0)
print("Saved: model_comparison_gap.png and model_comparison_gap.pdf")

# %%
