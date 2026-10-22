"""
plot_method.py — H2O+ pipeline diagram for the paper.
Two-stage layout: offline pretrain (top) → online H2O+ loop (bottom).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "method_diagram")

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.set_xlim(0, 11)
ax.set_ylim(0, 6)
ax.axis("off")

def box(x, y, w, h, text, color="#e8eef7", edge="#2c3e50",
        fontsize=8.5, weight="normal"):
    ax.add_patch(plt.Rectangle((x, y), w, h, linewidth=1.2,
                               edgecolor=edge, facecolor=color, zorder=2))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight, zorder=3)

def arrow(x1, y1, x2, y2, color="#444", lw=1.2, style="->"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=4)

# ── Stage 1: Offline pretrain (top row) ──────────────────────────────
ax.text(0.1, 5.6, "Stage 1: Offline pretrain (60K gradient steps)",
        fontsize=10, weight="bold", color="#5a3a86")

box(0.2, 4.4, 1.7, 0.9,
    "$\\mathcal{D}_{\\rm off}$\n675K transitions\n(SUMO-collected)",
    color="#f4ecd8", edge="#8c6d31", weight="bold")
box(2.5, 4.4, 2.2, 0.9,
    "Ensemble Q\n($E{=}5$, RE-SAC,\nLCB target $\\beta{=}-2$)",
    color="#e5f1e4", edge="#2a6f2a")
box(5.3, 4.4, 1.7, 0.9,
    "SAC policy\n$\\pi_\\phi$",
    color="#f6e1e7", edge="#8c2c4b")
box(7.7, 4.4, 3.0, 0.9,
    "Pretrained checkpoint\n$(\\theta_{1..E}, \\phi)$\n(initialises Stage 2)",
    color="#fff1cc", edge="#8c6d31", weight="bold")

arrow(1.9, 4.85, 2.5, 4.85)
arrow(4.7, 4.85, 5.3, 4.85)
arrow(7.0, 4.85, 7.7, 4.85)

# Divider
ax.plot([0.1, 10.9], [4.0, 4.0], "--", color="#999", lw=0.8, zorder=1)

# ── Stage 2: Online H2O+ loop (main panel) ──────────────────────────
ax.text(0.1, 3.7, "Stage 2: Online H2O+ (200 epochs, repeat per epoch)",
        fontsize=10, weight="bold", color="#5a3a86")

# Left: simulator
box(0.2, 1.8, 2.0, 1.5,
    "$\\mathcal{M}_{\\rm sim}$\nLSTM sim-core\n(40$\\times$ faster)\n+ snapshot reset\n($p_{\\rm reset}{=}0.5$)",
    color="#dceefa", edge="#1f4e79", fontsize=8)

# Centre: collect → buffer
box(2.7, 2.5, 1.7, 0.8,
    "Rollout\n100 events/epoch",
    color="#e8eef7", edge="#2c3e50")
box(2.7, 1.4, 1.7, 0.8,
    "$\\mathcal{D}_{\\rm on}$\nonline buffer",
    color="#f4ecd8", edge="#8c6d31")

# Discriminator
box(4.9, 2.5, 2.2, 0.8,
    "Discriminator $f_\\psi$\n(Trans / Dyn / Contrastive)",
    color="#fff1cc", edge="#8c6d31")
box(4.9, 1.4, 2.2, 0.8,
    "$w_{\\rm IS}{=}\\sigma/(1{-}\\sigma)$\nclip$[0.1,5.0]$",
    color="#fde2e2", edge="#a04040")

# SAC update
box(7.5, 2.5, 2.0, 0.8,
    "Hybrid Bellman\nLCB target",
    color="#e5f1e4", edge="#2a6f2a")
box(7.5, 1.4, 2.0, 0.8,
    "Policy update\n100 grad steps",
    color="#f6e1e7", edge="#8c2c4b")

# Eval branch (right)
box(9.8, 1.8, 1.0, 1.5,
    "$\\mathcal{M}_{\\rm real}$\nSUMO\neval\n(18000 s)",
    color="#d8e8d8", edge="#2a6f2a", fontsize=8, weight="bold")

# Arrows: rollout flow
arrow(2.2, 2.85, 2.7, 2.85)            # sim → rollout
arrow(3.55, 2.5, 3.55, 2.2)            # rollout → online buffer
arrow(4.4, 1.8, 4.9, 1.8)              # buffer → IS weight (via disc)
arrow(4.4, 2.85, 4.9, 2.85)            # rollout → disc training
arrow(7.1, 2.85, 7.5, 2.85)            # disc → bellman
arrow(7.1, 1.8, 7.5, 1.8)              # IS weight → bellman target
arrow(8.5, 2.5, 8.5, 2.2)              # bellman → policy
arrow(7.5, 1.8, 2.2, 0.5, color="#5a3a86", style="->", lw=1.5)
ax.text(4.85, 0.55, "policy update → next-epoch rollout",
        ha="center", va="center", fontsize=8, color="#5a3a86", style="italic")

# Eval (dashed) — periodic SUMO eval, post-training
arrow(9.5, 2.0, 9.8, 2.0, color="#2a6f2a", lw=1.4)
ax.text(9.65, 2.5, "after\ntraining", ha="center", va="bottom",
        fontsize=7, color="#2a6f2a")

# ── Legend ──
patches = [
    mpatches.Patch(color="#f4ecd8", label="Buffers"),
    mpatches.Patch(color="#dceefa", label="Sim-core (low fidelity)"),
    mpatches.Patch(color="#d8e8d8", label="SUMO (high fidelity)"),
    mpatches.Patch(color="#fff1cc", label="Discriminator"),
    mpatches.Patch(color="#e5f1e4", label="Q ensemble + Bellman"),
    mpatches.Patch(color="#f6e1e7", label="Policy $\\pi_\\phi$"),
]
ax.legend(handles=patches, loc="lower right", fontsize=7, ncol=2,
          framealpha=0.9, bbox_to_anchor=(1.0, -0.02))

plt.tight_layout()
plt.savefig(OUT + ".pdf", bbox_inches="tight")
plt.savefig(OUT + ".png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUT}.pdf / .png")
