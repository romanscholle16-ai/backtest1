"""One chart per trade + gallery + MFE/MAE diagnostics.  Usage: python scripts/plot_trades.py"""
import sys, html
sys.path.insert(0, ".")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from strategy import engine
from strategy.config import Params

P = Params()
M15 = pd.Timedelta(minutes=15)
INK, MUTED, GRID, BG = "#1f2328", "#6b7280", "#e5e7eb", "#ffffff"
C_STOP, C_TP, C_FVG, C_LIQ, C_ENTRY = "#c0392b", "#1f6feb", "#8e6bbf", "#d98e04", "#1f2328"
OUT = Path("reports/charts"); OUT.mkdir(parents=True, exist_ok=True)

d15 = pd.read_parquet("data/ETHUSDT_15m.parquet")
res = engine.run_backtest(d15, P)
tr, prep = res.trades.reset_index(drop=True), res.prep


def candles(ax, df):
    w = 0.6 * (M15 / pd.Timedelta(days=1))
    x = mdates.date2num(df.index.to_pydatetime())
    for xi, o, h, l, c in zip(x, df.open, df.high, df.low, df.close):
        up = c >= o
        ax.vlines(xi + w / 2, l, h, color=INK, lw=0.7, zorder=3)
        ax.add_patch(Rectangle((xi, min(o, c)), w, max(abs(c - o), 1e-9), zorder=4,
                               facecolor=BG if up else INK, edgecolor=INK, lw=0.7))
    return w


GUT = []  # (y, label, color) for the right-hand label gutter


def hline(ax, y, x0, x1, color, label, ls="-", lw=1.1, side="right"):
    ax.hlines(y, x0, x1, color=color, ls=ls, lw=lw, zorder=2)
    if side == "gutter":
        GUT.append((y, label, color))
        return
    ax.annotate(label, (x1 if side == "right" else x0, y), xytext=(4 if side == "right" else -4, 0),
                textcoords="offset points", ha="left" if side == "right" else "right", va="center",
                fontsize=7.5, color=color)


diag = []
for i, t in tr.iterrows():
    s = prep.setups.loc[t.setup_id]
    g = prep.triggers[(prep.triggers.setup_id == t.setup_id) & (prep.triggers.time == t.entry_time)].iloc[0]
    long = t.direction == "long"
    legs = [l.split("@") for l in t.legs.split(";")]
    t_exit = pd.Timestamp(legs[-1][2])
    risk = abs(t.entry - t.stop)
    # --- diagnostics: excursion between entry and final exit
    seg = d15[(d15.index >= t.entry_time) & (d15.index <= t_exit)]
    fav = ((seg.high.max() - t.entry) if long else (t.entry - seg.low.min())) / risk
    adv = ((t.entry - seg.low.min()) if long else (seg.high.max() - t.entry)) / risk
    first_bar_stop = len(seg) <= 2 and legs[0][0] == "stop"
    tgt_R = abs(t.target - t.entry) / risk
    diag.append(dict(trade=i, mfe_R=fav, mae_R=adv, bars_held=len(seg), stopped_within_2_bars=first_bar_stop,
                     target_R=tgt_R, max_pct_of_target=fav / tgt_R))
    # --- window
    w0 = min(s.sweep_time, g.fvg_known_at) - pd.Timedelta(hours=10)
    w1 = min(t_exit + pd.Timedelta(hours=6), t.entry_time + pd.Timedelta(days=3))
    df = d15[(d15.index >= w0) & (d15.index <= w1)]
    GUT.clear()
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=110)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    candles(ax, df)
    x0, x1 = mdates.date2num(df.index[0].to_pydatetime()), mdates.date2num(w1.to_pydatetime())
    xe = mdates.date2num(t.entry_time.to_pydatetime())
    xx = mdates.date2num(t_exit.to_pydatetime()) + 0.01
    xsw = mdates.date2num(s.sweep_time.to_pydatetime())
    XR = xx + max(0.25, 0.12 * (xx - x0))   # right end of trade lines; gutter labels sit just beyond
    # setup phases (45m bars)
    for key, lab in (("sweep_time", "sweep"), ("retrace_time", "retrace"), ("disp_time", "displacement")):
        a = mdates.date2num(s[key].to_pydatetime()); b = a + 0.75 / 24
        ax.axvspan(a, b, color=C_LIQ, alpha=0.10 if key != "disp_time" else 0.18, lw=0, zorder=1)
        ax.annotate(lab, (a, 0), xycoords=("data", "axes fraction"), xytext=(2, 4), textcoords="offset points",
                    fontsize=7, color=MUTED, rotation=90, va="bottom")
    ax.axvline(mdates.date2num(s.armed_at.to_pydatetime()), color=MUTED, ls=":", lw=0.9)
    # liquidity level + sweep extreme
    hline(ax, s.level_price, x0, xsw + 0.75 / 24, C_LIQ, f"liquidity {s.level_price:.2f} ({s.level_kind})", ls="--", side="left")
    # FVG box
    ax.add_patch(Rectangle((mdates.date2num(g.fvg_known_at.to_pydatetime()), g.fvg_bottom),
                           xe - mdates.date2num(g.fvg_known_at.to_pydatetime()), g.fvg_top - g.fvg_bottom,
                           facecolor=C_FVG, alpha=0.22, edgecolor=C_FVG, lw=0.8, zorder=2))
    ax.annotate("FVG", (mdates.date2num(g.fvg_known_at.to_pydatetime()), g.fvg_top), xytext=(2, 3),
                textcoords="offset points", fontsize=7.5, color=C_FVG)
    # trade lines
    hline(ax, t.entry, xe, XR, C_ENTRY, f"entry {t.entry:.2f}", lw=1.2, side="gutter")
    hline(ax, t.stop, xe, XR, C_STOP, f"stop {t.stop:.2f} ({t.risk_pct_of_price:.2f}% = 1R)", ls="--", side="gutter")
    hline(ax, t.target, xe, XR, C_TP, f"24h {'high' if long else 'low'} {t.target:.2f} ({tgt_R:.1f}R)", ls="--", side="gutter")
    if pd.notna(t.tp1): hline(ax, t.tp1, xe, XR, C_TP, f"TP1 {t.tp1:.2f}", ls=":", side="gutter")
    if pd.notna(t.tp2): hline(ax, t.tp2, xe, XR, C_TP, f"TP2 {t.tp2:.2f}", ls=":", side="gutter")
    # entry / exit markers
    ax.scatter([xe], [t.entry], marker="^" if long else "v", s=70, color=C_ENTRY, zorder=6)
    for reason, price, tm, frac in legs:
        ax.scatter([mdates.date2num(pd.Timestamp(tm).to_pydatetime()) + 0.005], [float(price)], marker="x" if reason in ("stop", "be", "trail") else "o",
                   s=55, color=C_STOP if reason in ("stop", "be", "trail") else C_TP, zorder=7, linewidths=1.8)
    # axes
    ax.set_xlim(x0, XR + 0.22 * (XR - x0))
    ys = pd.concat([df.high, df.low]); pad = (ys.max() - ys.min()) * 0.06
    lo_y, hi_y = min(ys.min(), t.stop) - pad, max(ys.max(), t.stop, t.target) + pad
    ax.set_ylim(lo_y, hi_y)
    # de-overlap gutter labels (min spacing 3.2% of the y-range), keep original order by price
    gap, placed = (hi_y - lo_y) * 0.032, []
    for y, lab, col in sorted(GUT):
        yy = y if not placed else max(y, placed[-1] + gap)
        placed.append(yy)
        ax.annotate(lab, (XR, y), xytext=(XR + 0.01, yy), textcoords="data", fontsize=7.5, color=col, va="center",
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.5, shrinkA=0, shrinkB=0) if abs(yy - y) > 1e-9 else None)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M")); ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
    ax.grid(color=GRID, lw=0.6); ax.tick_params(labelsize=8, colors=MUTED)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(GRID)
    res_txt = "WIN" if t.net_r > 0 else "LOSS"
    ax.set_title(f"#{i}  {t.direction.upper()}  entered {t.entry_time:%a %Y-%m-%d %H:%M} UTC   |   {res_txt} {t.net_r:+.2f}R net "
                 f"({t.gross_r:+.2f} gross, fees {t.fee_r:.2f})   exits: {t.exits}\n"
                 f"went {fav:.1f}R in favour ({fav / tgt_R * 100:.0f}% of the way to target), {adv:.1f}R against; held {len(seg)} bars",
                 loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / f"trade_{i:03d}.png"); plt.close(fig)

dg = pd.DataFrame(diag)
full = pd.concat([tr.drop(columns=["legs"]), dg.drop(columns="trade")], axis=1)
full.to_csv("reports/trades_diagnostics.csv", index=False)

# gallery
rows = []
for i, t in full.iterrows():
    cls = "w" if t.net_r > 0 else "l"
    rows.append(f'<a class="card {cls}" href="charts/trade_{i:03d}.png"><img loading="lazy" src="charts/trade_{i:03d}.png">'
                f'<div><b>#{i} {t.direction}</b> {t.entry_time:%Y-%m-%d %H:%M}<br>{t.net_r:+.2f}R · {html.escape(t.exits)}<br>'
                f'MFE {t.mfe_R:.1f}R · stop {t.risk_pct_of_price:.2f}%</div></a>')
Path("reports/index.html").write_text(f"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Trade charts</title><style>body{{font:14px system-ui;margin:16px;background:#fff;color:#1f2328}}
.g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:12px}}
.card{{display:block;border:1px solid #e5e7eb;border-radius:6px;padding:6px;text-decoration:none;color:inherit}}
.card.l{{border-left:4px solid #c0392b}}.card.w{{border-left:4px solid #1f6feb}}img{{width:100%}}div{{font-size:12px;margin-top:4px}}</style>
<h1>{len(full)} trades</h1><p>Red edge = loss, blue edge = win. Click for full-size chart.</p><div class=g>{''.join(rows)}</div>""")
print("charts:", len(full))
