"""
Gamry EIS Plotter  –  v5.4 (Bugfix)
---------------------------------
Fix for UnboundLocalError in find_best_rect.
Corrected the variable scope for the Right-to-Left scan logic.
"""

import os, sys, time
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import savgol_filter
from matplotlib.lines import Line2D

# ── Aesthetics ───────────────────────────────────────────────────────────────
MARKERS = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '<']
COLOURS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#555555',
    '#bcbd22', '#17becf',
]

# ── Nice-number tick helpers ─────────────────────────────────────────────────

def nice_spacing(data_range: float, n_ticks: int) -> float:
    """Return a 'nice' tick interval (1/2/5 × 10^n) for the given range."""
    if data_range <= 0:
        return 1.0
    raw  = data_range / n_ticks
    exp  = np.floor(np.log10(raw))
    base = raw / 10 ** exp
    if   base < 1.5: nice = 1
    elif base < 3.5: nice = 2
    elif base < 7.5: nice = 5
    else:            nice = 10
    return nice * 10 ** exp


def set_linear_ticks(ax, axis_max: float, n_ticks: int, which: str = 'both'):
    sp    = nice_spacing(axis_max, n_ticks)
    ticks = np.arange(0, axis_max + sp * 0.01, sp)
    fmt   = _EngFmt()
    if which in ('both', 'x'):
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(fmt)
    if which in ('both', 'y'):
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(fmt)


def nice_axis_max(data_max: float, n_ticks: int) -> float:
    sp = nice_spacing(data_max, n_ticks)
    return np.ceil(data_max / sp) * sp


class _EngFmt(ticker.Formatter):
    def __call__(self, x, pos=None):
        if x == 0: return '0'
        exp   = int(np.floor(np.log10(abs(x))))
        coeff = x / 10 ** exp
        c = int(round(coeff)) if abs(coeff - round(coeff)) < 0.05 else round(coeff, 1)
        if exp == 0: return f'{c}'
        if exp == 1: return f'{c}×10'
        return f'{c}×10$^{{{exp}}}$'


# ── Smart placement ──────────────────────────────────────────────────────────

def find_best_rect(norm_px, norm_py,
                   req_w=0.42, req_h=0.42,
                   min_w=0.12, min_h=0.12,
                   step=0.03, N=60,
                   margin=0.04,
                   left_tick_margin=0.13,
                   bottom_tick_margin=0.08,
                   top_tick_margin=0.0,
                   right_tick_margin=0.0,
                   exclude_rects=None,
                   scan_order='LR'): # 'LR' (Left->Right) or 'RL' (Right->Left)
    if exclude_rects is None: exclude_rects = []
    base_grid = np.zeros((N, N), dtype=np.float32)

    lc = int(np.ceil(left_tick_margin   * N))
    br = int(np.ceil(bottom_tick_margin * N))
    base_grid[:, :lc] += 999
    base_grid[:br, :] += 999

    if top_tick_margin > 0:
        tr = int(np.ceil(top_tick_margin * N))
        base_grid[N-tr:, :] += 999
    if right_tick_margin > 0:
        rc = int(np.ceil(right_tick_margin * N))
        base_grid[:, N-rc:] += 999

    mc = max(1, int(np.ceil(margin * N)))
    for px, py in zip(norm_px, norm_py):
        xi = int(np.clip(px * N, 0, N - 1))
        yi = int(np.clip(py * N, 0, N - 1))
        base_grid[max(0, yi - mc):min(N, yi + mc + 1),
                  max(0, xi - mc):min(N, xi + mc + 1)] += 1

    for (ex0, ey0, ew, eh) in exclude_rects:
        c0 = int(np.floor((ex0 - margin) * N))
        c1 = int(np.ceil( (ex0 + ew + margin) * N))
        r0 = int(np.floor((ey0 - margin) * N))
        r1 = int(np.ceil( (ey0 + eh + margin) * N))
        base_grid[max(0, r0):min(N, r1), max(0, c0):min(N, c1)] += 999

    arr_px = np.array(norm_px, dtype=float)
    arr_py = np.array(norm_py, dtype=float)

    for w_try in np.arange(req_w, min_w - 1e-9, -step):
        for h_try in np.arange(req_h, min_h - 1e-9, -step):
            req_wc = max(1, int(np.ceil(w_try * N)))
            req_hc = max(1, int(np.ceil(h_try * N)))

            # FIX: Determine column scan order HERE, after req_wc is known
            if scan_order == 'RL':
                col_range = range(N - req_wc, -1, -1)
            else:
                col_range = range(N - req_wc + 1)

            best_score = -1
            best_pos   = None

            for row in range(N - req_hc + 1):
                for col in col_range:
                    if base_grid[row:row + req_hc, col:col + req_wc].sum() > 0:
                        continue
                    x0 = col / N;  y0 = row / N
                    w  = req_wc / N; h  = req_hc / N
                    if x0 + w > 1.0 or y0 + h > 1.0:
                        continue
                    box_cx = x0 + w / 2
                    box_cy = y0 + h / 2
                    dists  = np.hypot(arr_px - box_cx, arr_py - box_cy)
                    score  = dists.min()
                    if score > best_score:
                        best_score = score
                        best_pos   = (x0, y0, w, h)

            if best_pos is not None:
                return best_pos
    return None


# ── Gamry .dta parser ────────────────────────────────────────────────────────

def parse_dta(filepath: str) -> dict:
    with open(filepath, 'r', encoding='latin-1', errors='replace') as fh:
        lines = fh.readlines()
    table_start = header_line = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith('ZCURVE') or (s.startswith('CURVE') and 'OCVCURVE' not in s):
            for j in range(i + 1, len(lines)):
                cand = lines[j].strip()
                if cand and not cand.startswith('#'):
                    header_line = j
                    table_start = j + 2
                    break
            break
    if table_start is None:
        raise ValueError(f"No ZCURVE/CURVE table found in {filepath}")
    col_names = [c.strip().upper() for c in lines[header_line].split('\t')]
    while col_names and col_names[0] == '': col_names.pop(0)

    def col(*options):
        for n in options:
            if n in col_names: return col_names.index(n)
        return None

    idx_freq  = col('FREQ', 'FREQUENCY')
    idx_zreal = col('ZREAL', 'ZRE', "Z'")
    idx_zimag = col('ZIMAG', 'ZIM', 'Z"', "Z''")
    idx_zmod  = col('ZMOD', 'ZM', '|Z|')
    idx_phase = col('ZPHZ', 'PHASE', 'ZIMPH', 'THETA')

    if None in (idx_freq, idx_zreal, idx_zimag):
        raise ValueError(f"Required columns missing.\nFound: {col_names}")

    rows = []
    for line in lines[table_start:]:
        s = line.strip()
        if not s or s.startswith('#') or s.upper().startswith('END'): break
        parts = s.split('\t')
        try:
            rows.append([float(p.replace(',', '.')) for p in parts])
        except ValueError: continue

    arr = np.array(rows)
    freq  = arr[:, idx_freq]
    zreal = arr[:, idx_zreal]
    zimag = arr[:, idx_zimag]
    zmod  = arr[:, idx_zmod]  if idx_zmod  is not None else np.hypot(zreal, zimag)
    phase = arr[:, idx_phase] if idx_phase is not None else np.degrees(np.arctan2(zimag, zreal))

    order = np.argsort(freq)[::-1]
    return {
        'filename': os.path.splitext(os.path.basename(filepath))[0],
        'Freq' : freq [order],
        'Zreal': zreal[order],
        'Zimag': zimag[order],
        'Zmod' : zmod [order],
        'Phase': phase[order],
    }


def smooth(y: np.ndarray, window: int = 9, poly: int = 3) -> np.ndarray:
    n = len(y)
    if n < 5: return y.copy()
    w = window if window % 2 == 1 else window + 1
    w = min(w, n if n % 2 == 1 else n - 1)
    w = max(w, poly + 2 if (poly + 2) % 2 == 1 else poly + 3)
    return savgol_filter(y, window_length=w, polyorder=poly)


# ── GUI helpers ──────────────────────────────────────────────────────────────

def select_files():
    root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
    files = filedialog.askopenfilenames(
        title='Select Gamry .dta files',
        filetypes=[('Gamry DTA', '*.dta *.DTA'), ('All files', '*.*')],
    )
    root.destroy()
    return list(files)

def ask_name(default: str) -> str:
    root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
    keep = messagebox.askyesno('Dataset label', f'Keep the default name?\n\n  "{default}"', parent=root)
    if keep: root.destroy(); return default
    name = simpledialog.askstring('Dataset label', 'Enter a custom label:', initialvalue=default, parent=root)
    root.destroy()
    return (name or default).strip() or default


# ── Interactive Elements Helper ──────────────────────────────────────────────

# ── Interactive Elements Helper (Blit-Optimized for Instant Response) ───────

class DraggableAxes:
    # (Keep DraggableAxes exactly as it was in the previous step, or use standard)
    # I'll include it here for completeness, but the lag is mostly in Legend.
    def __init__(self, axins, fig):
        self.axins = axins
        self.fig = fig
        self.press = None
        self.cidpress = fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.axins: return
        self.press = (event.x, event.y), self.axins.get_position()

    def on_motion(self, event):
        if self.press is None: return
        (xpress, ypress), box = self.press
        dx = event.x - xpress
        dy = event.y - ypress
        inv = self.fig.transFigure.inverted()
        dx_fig = inv.transform((xpress + dx, ypress))[0] - inv.transform((xpress, ypress))[0]
        dy_fig = inv.transform((xpress, ypress + dy))[1] - inv.transform((xpress, ypress))[1]
        new_box = [box.x0 + dx_fig, box.y0 + dy_fig, box.width, box.height]
        self.axins.set_position(new_box)
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        self.press = None
        self.fig.canvas.draw_idle()

class DraggableLegend:
    """
    Blit-Optimized Legend Dragger.
    Takes a snapshot of the background (blitting) to avoid redrawing the complex graph.
    Result: Smooth, lag-free dragging.
    """
    def __init__(self, legend, fig):
        self.legend = legend
        self.fig = fig
        self.press = None
        self.background = None # Will store the snapshot

        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.button != 1: return
        
        renderer = self.fig.canvas.get_renderer()
        bbox = self.legend.get_window_extent(renderer=renderer)
        
        if (bbox.x0 <= event.x <= bbox.x1) and (bbox.y0 <= event.y <= bbox.y1):
            # Save the click offset so it doesn't snap to corner
            offset_x = event.x - bbox.x0
            offset_y = event.y - bbox.y0
            self.press = (event.x, event.y, offset_x, offset_y)
            
            # CRITICAL: Take a snapshot of the current canvas
            # We use fig.bbox to save the whole window
            self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def on_motion(self, event):
        if self.press is None: return
        
        # If we failed to get a background, fall back to slow redraw (safety)
        if self.background is None:
            self.fig.canvas.draw_idle()
            return

        _, _, offset_x, offset_y = self.press
        
        new_x_px = event.x - offset_x
        new_y_px = event.y - offset_y
        
        inv = self.fig.transFigure.inverted()
        fig_x, fig_y = inv.transform((new_x_px, new_y_px))
        
        # Move the legend in memory
        self.legend.set_bbox_to_anchor((fig_x, fig_y))
        
        # BLITTING STEPS:
        # 1. Restore the old snapshot (erases the legend from previous position)
        self.fig.canvas.restore_region(self.background)
        
        # 2. Draw the legend in the new position
        self.fig.draw_artist(self.legend)
        
        # 3. Update the screen instantly
        self.fig.canvas.blit(self.fig.bbox)

    def on_release(self, event):
        if self.press:
            self.press = None
            # One final full draw to ensure everything is perfect
            self.fig.canvas.draw()
            self.background = None # Clear the snapshot

def setup_interactive(fig, axins=None, legends=None, force_custom=False):
    if legends is None: legends = []
    if not hasattr(fig, '_drag_refs'):
        fig._drag_refs = []

    for leg in legends:
        if leg:
            leg.set_zorder(100)
            if force_custom:
                # Use the Blit-Optimized handler
                handler = DraggableLegend(leg, fig)
                fig._drag_refs.append(handler)
            else:
                # Standard for Nyquist
                leg.set_draggable(True, use_blit=True, update='loc')
    
    if axins is not None:
        handler = DraggableAxes(axins, fig)
        fig._drag_refs.append(handler)


# ── Nyquist plot ─────────────────────────────────────────────────────────────

def plot_nyquist(datasets):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    all_x, all_y = [], []
    for i, ds in enumerate(datasets):
        c, m = COLOURS[i % len(COLOURS)], MARKERS[i % len(MARKERS)]
        x =  ds['Zreal']; y = -ds['Zimag']
        ax.plot(smooth(x), smooth(y), '-', color=c, lw=1.8, zorder=2)
        ax.plot(x, y, ls='None', marker=m, color=c, ms=5, label=ds['label'], zorder=3)
        all_x.extend(x); all_y.extend(y)

    data_max = max(max(all_x), max(all_y))
    N_MAIN   = 4
    ax_lim   = nice_axis_max(data_max, N_MAIN)
    ax.set_xlim(0, ax_lim); ax.set_ylim(0, ax_lim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Z$_{real}$ (Ω)', fontsize=12); ax.set_ylabel('–Z$_{imag}$ (Ω)', fontsize=12)
    ax.set_title('Nyquist Plot', fontsize=14, pad=10); ax.grid(False)
    set_linear_ticks(ax, ax_lim, N_MAIN, 'both')

    # Inset
    N_INS  = 3
    si     = int(np.argmin([ds['Zmod'].max() for ds in datasets]))
    ins_x  = max(0.0, datasets[si]['Zreal'].max())
    ins_y  = max(0.0, (-datasets[si]['Zimag']).max())
    ins_data_max = max(ins_x, ins_y)
    ins_lim      = nice_axis_max(ins_data_max, N_INS)

    px = [x / ax_lim for x in all_x]
    py = [y / ax_lim for y in all_y]

    inset_rect = find_best_rect(px, py, N=60, req_w=0.42, req_h=0.42, min_w=0.15, min_h=0.15, step=0.03,
                                margin=0.04, left_tick_margin=0.13, bottom_tick_margin=0.08,
                                top_tick_margin=0.10, right_tick_margin=0.10)
    if inset_rect is None: inset_rect = (0.60, 0.60, 0.35, 0.35)

    axins = ax.inset_axes(list(inset_rect))
    axins.set_facecolor('white'); axins.grid(False)
    for i, ds in enumerate(datasets):
        c, m = COLOURS[i % len(COLOURS)], MARKERS[i % len(MARKERS)]
        x =  ds['Zreal']; y = -ds['Zimag']
        axins.plot(smooth(x), smooth(y), '-', color=c, lw=1.8, zorder=2)
        axins.plot(x, y, ls='None', marker=m, color=c, ms=4, zorder=3)
    axins.set_xlim(0, ins_lim); axins.set_ylim(0, ins_lim)
    axins.set_aspect('equal', adjustable='datalim')
    axins.tick_params(labelsize=7); axins.set_xlabel(''); axins.set_ylabel('')
    set_linear_ticks(axins, ins_lim, N_INS, 'both')
    ax.indicate_inset_zoom(axins, edgecolor='0.40', lw=0.9)

    # Legend
    n_ds = len(datasets)
    leg_h_est = min(0.12 + n_ds * 0.055, 0.45)
    leg_w_est = 0.24

    leg_rect = find_best_rect(px, py, N=60, req_w=leg_w_est, req_h=leg_h_est, min_w=0.12, min_h=0.10, step=0.02,
                              margin=0.06, left_tick_margin=0.14, bottom_tick_margin=0.10,
                              top_tick_margin=0.08, right_tick_margin=0.08, exclude_rects=[inset_rect],
                              scan_order='LR') # Nyquist: Standard Left->Right scan
    
    leg1 = None
    if leg_rect:
        lx, ly, lw, lh = leg_rect
        leg1 = ax.legend(loc='lower left', bbox_to_anchor=(lx, ly), bbox_transform=ax.transAxes,
                         fontsize=9, framealpha=0.95, edgecolor='0.55', borderpad=0.7)
    else:
        leg1 = ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), bbox_transform=ax.transAxes,
                         fontsize=9, framealpha=0.95, edgecolor='0.55', borderpad=0.7)
        fig.subplots_adjust(right=0.80)

    fig.tight_layout()
    
    setup_interactive(fig, axins=axins, legends=[leg1])
    
    return fig


# ── Bode plot ─────────────────────────────────────────────────────────────────

def plot_bode(datasets):
    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax1.set_facecolor('white'); fig.patch.set_facecolor('white')
    ax2 = ax1.twinx(); ax2.set_facecolor('none')

    proxies = []
    for i, ds in enumerate(datasets):
        c, m = COLOURS[i % len(COLOURS)], MARKERS[i % len(MARKERS)]
        freq, zmod, phase = ds['Freq'], ds['Zmod'], ds['Phase']
        ax1.plot(freq, 10 ** smooth(np.log10(zmod)), '-', color=c, lw=1.8, zorder=2)
        ax1.plot(freq, zmod, ls='None', marker=m, color=c, ms=5, zorder=3)
        ax2.plot(freq, smooth(phase), '--', color=c, lw=1.8, zorder=2)
        ax2.plot(freq, phase, ls='None', marker=m, color=c, ms=5, markerfacecolor='white', zorder=3)
        proxies.append(Line2D([0], [0], color=c, marker=m, lw=1.8, label=ds['label']))

    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_xlabel('Frequency (Hz)', fontsize=12)
    ax1.set_ylabel('|Z| (Ω)', fontsize=12); ax2.set_ylabel('Phase (°)', fontsize=12)
    ax1.set_title('Bode Plot', fontsize=14, pad=10)
    ax1.grid(False); ax2.grid(False)

    style_solid  = Line2D([0], [0], color='#333', lw=1.8, ls='-',  label='|Z|  (solid)')
    style_dashed = Line2D([0], [0], color='#333', lw=1.8, ls='--', label='Phase (dashed)')

    # Normalization
    all_freq  = np.concatenate([d['Freq']  for d in datasets])
    all_zmod  = np.concatenate([d['Zmod']  for d in datasets])
    all_phase = np.concatenate([d['Phase'] for d in datasets])
    lf_min = np.log10(all_freq.min()); lf_max = np.log10(all_freq.max())
    lz_min = np.log10(all_zmod.min()); lz_max = np.log10(all_zmod.max())
    ph_min = all_phase.min(); ph_max = all_phase.max()

    # Combined points for Legend 1 (Datasets) search.
    bode_px, bode_py = [], []
    for d in datasets:
        lf = (np.log10(d['Freq'])  - lf_min) / max(lf_max - lf_min, 1e-9)
        lz = (np.log10(d['Zmod'])  - lz_min) / max(lz_max - lz_min, 1e-9)
        ph = (d['Phase'] - ph_min) / max(ph_max - ph_min, 1e-9)
        bode_px.extend(list(lf) * 2)
        bode_py.extend(list(lz) + list(ph))

    n_ds  = len(datasets)
    l1_h  = min(0.10 + n_ds * 0.055, 0.45)
    l1_w  = 0.22

    leg1_rect = find_best_rect(
        bode_px, bode_py, N=60,
        req_w=l1_w, req_h=l1_h, min_w=0.10, min_h=0.06, step=0.02,
        margin=0.06, left_tick_margin=0.0, bottom_tick_margin=0.10,
        top_tick_margin=0.05, right_tick_margin=0.05,
        scan_order='RL' 
    )
    
    leg2_rect = find_best_rect(
        bode_px, bode_py, N=60,
        req_w=0.24, req_h=0.14, min_w=0.10, min_h=0.08, step=0.02,
        margin=0.06, left_tick_margin=0.0, bottom_tick_margin=0.0,
        top_tick_margin=0.0, right_tick_margin=0.0,
        exclude_rects=[leg1_rect] if leg1_rect else [],
        scan_order='RL'
    )

    # Draw Legend 1
    l1 = None
    if leg1_rect:
        lx, ly, lw, lh = leg1_rect
        l1 = ax1.legend(handles=proxies, loc='lower left', bbox_to_anchor=(lx, ly), bbox_transform=ax1.transAxes,
                        fontsize=9, framealpha=0.95, edgecolor='0.55', title='Datasets', title_fontsize=9)
    else:
        l1 = ax1.legend(handles=proxies, loc='upper left', bbox_to_anchor=(1.12, 1.0), bbox_transform=ax1.transAxes,
                        fontsize=9, framealpha=0.95, edgecolor='0.55', title='Datasets', title_fontsize=9)
    ax1.add_artist(l1)

    # Draw Legend 2
    if leg2_rect:
        lx2, ly2, lw2, lh2 = leg2_rect
        l2 = ax1.legend(handles=[style_solid, style_dashed], loc='lower left', bbox_to_anchor=(lx2, ly2), 
                        bbox_transform=ax1.transAxes, fontsize=9, framealpha=0.95, edgecolor='0.55',
                        title='Line style', title_fontsize=9)
    else:
        l2 = ax1.legend(handles=[style_solid, style_dashed], loc='upper left', bbox_to_anchor=(1.12, 0.50), 
                        bbox_transform=ax1.transAxes, fontsize=9, framealpha=0.95, edgecolor='0.55',
                        title='Line style', title_fontsize=9)


    if leg1_rect is None or leg2_rect is None:
        fig.subplots_adjust(right=0.78)

    fig.tight_layout()
    
    setup_interactive(fig, legends=[l1, l2], force_custom=True)
    
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    files = select_files()
    if not files:
        print('No files selected – exiting.')
        sys.exit(0)

    datasets = []
    for fp in files:
        try:
            ds = parse_dta(fp)
        except Exception as exc:
            print(f'[SKIP] {fp}\n  ↳ {exc}')
            continue
        ds['label'] = ask_name(ds['filename'])
        datasets.append(ds)

    if not datasets:
        print('No valid datasets – exiting.')
        sys.exit(1)

    print(f'\nLoaded {len(datasets)} dataset(s):')
    for ds in datasets:
        print(f"  • {ds['label']}  ({len(ds['Freq'])} pts, {ds['Freq'].min():.3g} – {ds['Freq'].max():.3g} Hz)")

    plot_nyquist(datasets)
    plot_bode(datasets)
    plt.show()


if __name__ == '__main__':
    main()