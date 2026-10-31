#!/usr/bin/env python3
"""Draw the bus-holding workflow as an editable vector PDF."""
from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/tits-bus-method-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "dejavusans",
})
fig, ax = plt.subplots(figsize=(7.10, 2.60), dpi=180)
fig.subplots_adjust(0, 0, 1, 1)
ax.set_xlim(0, 7.10)
ax.set_ylim(0, 2.66)
ax.axis("off")
ink = "#1d2939"
line = "#46566b"
blue = "#edf3f7"
warm = "#fff2de"

def box(x, y, w, h, text, face=blue, edge=line, lw=.8,
        fontsize=8.15, weight="normal", linestyle="-"):
    p = FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.012,rounding_size=0.035",
                      linewidth=lw,edgecolor=edge,facecolor=face,linestyle=linestyle)
    ax.add_patch(p)
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",color=ink,
            fontsize=fontsize,fontweight=weight,linespacing=1.20)
    return p

def arrow(p, q, color=line, lw=.85, style="-|>", rad=0):
    a=FancyArrowPatch(p,q,arrowstyle=style,mutation_scale=7.3,
                     linewidth=lw,color=color,
                     connectionstyle=f"arc3,rad={rad}",shrinkA=0,shrinkB=0)
    ax.add_patch(a)
    return a

def route(points, color=line, lw=.85, arrow_at_end=True):
    for a,b in zip(points[:-2],points[1:-1]):
        ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=lw)
    if arrow_at_end: arrow(points[-2],points[-1],color=color,lw=lw)
    else: ax.plot([points[-2][0],points[-1][0]],[points[-2][1],points[-1][1]],color=color,lw=lw)

ax.text(.09,2.53,"TRAINING",fontsize=8.7,fontweight="bold",color=ink)
ax.text(.83,2.53,"Offline data and interaction with the event-based simulator",fontsize=8.0,color=line)

# Training row.
box(.09,2.04,1.37,.39,"Offline SUMO data")
box(1.83,2.04,1.41,.39,"IQL initialization\n50,000 updates")
box(3.65,2.04,1.78,.39,"Hybrid learning\n25,000 simulator events")
box(5.89,2.04,1.12,.39,"Fixed policy")
arrow((1.475,2.235),(1.815,2.235))
arrow((3.255,2.235),(3.635,2.235))
arrow((5.445,2.235),(5.875,2.235))

# The hybrid critic samples half of each minibatch from each source.
route([(.775,2.025),(.775,1.79),(3.42,1.79),(3.42,2.12),(3.635,2.12)])
ax.text(1.99,1.835,"50% logged critic replay",fontsize=7.85,color=line,ha="center")
box(3.65,1.325,1.78,.37,"Event-based simulator")
arrow((4.54,1.71),(4.54,2.025))
ax.text(4.65,1.85,"50% critic replay",fontsize=7.7,color=line,ha="left")
ax.text(.1,1.43,"No online interaction with evaluation SUMO.",fontsize=7.8,color=line)

ax.plot([.09,7.01],[1.16,1.16],color="#c7ced5",lw=.65)

# The same learned actor supplies both fixed evaluation controllers.
arrow((6.45,2.025),(6.45,.855))
ax.text(6.53,1.53,"Weights fixed",fontsize=7.65,color=line,ha="left",rotation=90,va="center")
ax.text(.09,1.04,"SIMULATION EVALUATION",fontsize=8.7,fontweight="bold",color=ink)

box(5.89,.435,1.12,.405,"Shared hold\ncalibration")
box(3.18,.66,2.13,.37,"Matched reference: at least 20 s\n" + r"$0.50 \leq t/T < 0.75$",
    face="#f4f5f6",linestyle="--",fontsize=8.0)
box(3.18,.165,2.13,.37,"Proposed: exactly 60 s\n" + r"$0.15 \leq t/T < 0.75$",
    face=warm,edge="#8b5e24",lw=1.0,fontsize=8.0)
box(.19,.365,2.22,.53,"Paired SUMO evaluation\nNo parameter updates",face="#f4f5f6",fontsize=8.3)
route([(5.875,.638),(5.57,.638),(5.57,.845),(5.325,.845)])
route([(5.57,.638),(5.57,.35),(5.325,.35)])
route([(3.165,.845),(2.72,.845),(2.72,.72),(2.425,.72)])
route([(3.165,.35),(2.72,.35),(2.72,.54),(2.425,.54)])
ax.text(3.55,.045,"Both rules require low load, continuing demand, a preceding bus record, and no active following bus.",
        fontsize=7.05,color=line,ha="center")

for suffix in ("pdf","png"):
    fig.savefig(OUT/f"controller_overview.{suffix}",dpi=240,facecolor="white")
plt.close(fig)
print(OUT/"controller_overview.pdf")
print(OUT/"controller_overview.png")

