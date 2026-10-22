from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(exist_ok=True)


def add_box(ax, x, y, w, h, title, body, face, edge):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.009",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.15,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.018,
        y + h - 0.035,
        title,
        ha="left",
        va="top",
        fontsize=8.2,
        fontweight="bold",
        color="#172033",
    )
    ax.text(
        x + 0.018,
        y + h - 0.085,
        body,
        ha="left",
        va="top",
        fontsize=6.2,
        color="#27364A",
        linespacing=1.23,
    )


def add_arrow(ax, x1, y1, x2, y2):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=1.25,
        color="#526071",
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arrow)


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(7.45, 4.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.965,
        "Registered v6 cross-fidelity evidence chain",
        ha="center",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        0.5,
        0.915,
        "One transit contract links immutable target data, online learning, and held-out evaluation",
        ha="center",
        va="top",
        fontsize=7.1,
        color="#536173",
    )

    x_positions = [0.025, 0.272, 0.519, 0.766]
    width = 0.209
    top_y = 0.505
    top_h = 0.335

    add_box(
        ax,
        x_positions[0],
        top_y,
        width,
        top_h,
        "1  Target evidence",
        "SUMO microsimulation\n12 directional services\n389 scheduled trips\n\nImmutable target replay\n2,831,150 transitions\n210 complete episodes\n30 training cells",
        "#EAF3FA",
        "#2F6F9F",
    )
    add_box(
        ax,
        x_positions[1],
        top_y,
        width,
        top_h,
        "2  Online-model gate",
        "Causal event simulator\nSame timetable, capacity,\njourneys, state, action,\nand reward\n\n30 zero-hold cell pairs\nThree demand levels\nAll predefined\nconfidence gates pass",
        "#ECF7F1",
        "#2D7D59",
    )
    add_box(
        ax,
        x_positions[2],
        top_y,
        width,
        top_h,
        "3  Registered learners",
        "Local H2O+ family\nIQL: 50,000 warm-start\nupdates\n5,000-event warmup\n95,000 post-warmup\nevent target\n100,000-event hard cap\n\nServer baselines\nRLPD, WSRL, mixed/online SAC\nOffline: CQL, IQL, BC",
        "#FFF5E8",
        "#A65B13",
    )
    add_box(
        ax,
        x_positions[3],
        top_y,
        width,
        top_h,
        "4  Frozen evaluation",
        "Final checkpoints only\n5 seeds x 30 held-out\nSUMO cells, paired blocks\n\nPrimary outcome\nPassenger seconds per\ndeparted passenger\n\nHierarchical bootstrap\nSign-flip tests + Holm",
        "#F2F3F6",
        "#606A78",
    )

    add_arrow(ax, 0.234, 0.675, 0.272, 0.675)
    add_arrow(ax, 0.481, 0.675, 0.519, 0.675)
    add_arrow(ax, 0.728, 0.675, 0.766, 0.675)

    ax.text(
        0.025,
        0.445,
        "Protocol invariants",
        fontsize=8.2,
        fontweight="bold",
        color="#172033",
    )
    invariants = [
        ("Chronology", "Simultaneous requests are serialized; zero-time tokens do not duplicate cost."),
        ("Passenger welfare", "Wait, ride, hold, denied boarding, transfers, and unfinished journeys share one ledger."),
        ("Action semantics", "A hurdle policy assigns explicit probability to exact zero hold and density above zero."),
        ("Provenance", "Data, simulator, plans, checkpoints, and evaluations are hash-locked and independently audited."),
    ]
    row_y = [0.365, 0.292, 0.219, 0.146]
    colors = ["#EAF3FA", "#ECF7F1", "#FFF5E8", "#F2F3F6"]
    edges = ["#2F6F9F", "#2D7D59", "#A65B13", "#606A78"]
    for (title, body), y, face, edge in zip(invariants, row_y, colors, edges):
        ax.add_patch(Rectangle((0.025, y), 0.95, 0.057, facecolor=face, edgecolor=edge, linewidth=0.8))
        ax.text(0.041, y + 0.029, title, ha="left", va="center", fontsize=6.9, fontweight="bold", color="#172033")
        ax.text(0.215, y + 0.029, body, ha="left", va="center", fontsize=6.35, color="#27364A")

    ax.add_patch(Rectangle((0.025, 0.045), 0.95, 0.065, facecolor="#FFF1F1", edgecolor="#A33A3A", linewidth=1.0))
    ax.text(0.041, 0.0775, "Claim boundary", ha="left", va="center", fontsize=7.2, fontweight="bold", color="#7A2626")
    ax.text(
        0.215,
        0.0775,
        "Simulator-to-simulator evidence only; policy outcomes require passed\nexecution and evaluation gates.",
        ha="left",
        va="center",
        fontsize=6.2,
        color="#5A2A2A",
        linespacing=1.15,
    )

    fig.tight_layout(pad=0.25)
    fig.savefig(OUT_DIR / "framework_overview.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "framework_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
