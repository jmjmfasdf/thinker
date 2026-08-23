"""
새 figure 생성 스크립트
- Fig 1: Run-by-run rho strip plot (모델 × ROI)
- Fig 2: 개선된 RSA heatmap (실제 데이터 범위 컬러스케일 + 유의미 마커)
- Fig 3: Fisher combined chi2 bar chart (런 간 일관성)
- Fig 4: tree_reps 런별 유의미성 상세 (rho_obs + significance flag)
- Fig 5: Encoding incremental improvement (hippocampus)
"""

import csv
import math
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats

# ── paths ────────────────────────────────────────────────────────────────────
BASE = (
    '/scratch/jeongmin/workcache/'
    '553d4f3685a0f4b4bc2c39a783636be6673efff0ff2524a5c7a7106ce5fe3335/'
    'research_script/outputs/07_encoding_rsa/sub001_game2'
)
PERM_CSV     = f'{BASE}/rsa/rsa_permutation_manifest.csv'
ENCODING_CSV = f'{BASE}/encoding/encoding_manifest.csv'
OUT_DIR      = '/scratch/user/figures'
os.makedirs(OUT_DIR, exist_ok=True)

# ── load data ────────────────────────────────────────────────────────────────
def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

perm_rows = load_csv(PERM_CSV)
enc_rows  = load_csv(ENCODING_CSV)

for r in perm_rows:
    r['rho_obs']         = float(r['rho_obs'])
    r['p_one_sided']     = float(r['p_one_sided'])
    r['q_fdr_one_sided'] = float(r['q_fdr_one_sided'])

for r in enc_rows:
    r['mean_r']   = float(r['mean_r'])
    r['median_r'] = float(r['median_r'])

# ── constants ────────────────────────────────────────────────────────────────
RUNS = [
    'sub001_ses01_block03_game2',
    'sub001_ses01_block06_game2',
    'sub001_ses02_block02_game2',
    'sub001_ses02_block04_game2',
    'sub001_ses02_block08_game2',
    'sub001_ses03_block02_game2',
    'sub001_ses03_block04_game2',
    'sub001_ses03_block08_game2',
    'sub001_ses04_block02_game2',
    'sub001_ses04_block04_game2',
    'sub001_ses04_block08_game2',
]
RUN_SHORT = [r.replace('sub001_', '').replace('_game2', '') for r in RUNS]

PRIMARY_MODELS  = ['tree_reps_primary', 'im_vectors_primary', 'im_vp_vectors_primary', 'ram']
MODEL_LABELS    = {
    'tree_reps_primary':     'tree_reps\n(primary)',
    'tree_reps_s2only':      'tree_reps\n(s2only)',
    'im_vectors_primary':    'im_vectors\n(primary)',
    'im_vectors_s2only':     'im_vectors\n(s2only)',
    'im_vp_vectors_primary': 'im_vp\n(primary)',
    'im_vp_vectors_s2only':  'im_vp\n(s2only)',
    'ram':                   'RAM',
}
MODEL_COLORS = {
    'tree_reps_primary':     '#2166AC',
    'tree_reps_s2only':      '#92C5DE',
    'im_vectors_primary':    '#D6604D',
    'im_vectors_s2only':     '#F4A582',
    'im_vp_vectors_primary': '#4DAC26',
    'im_vp_vectors_s2only':  '#B8E186',
    'ram':                   '#7B3294',
}

FOCUS_ROIS   = ['hippocampus', 'right_hippocampus', 'pfc', 'coupling_hipp_pfc']
ROI_LABELS   = {
    'hippocampus':        'Hipp\n(bilateral)',
    'left_hippocampus':   'Hipp\n(left)',
    'right_hippocampus':  'Hipp\n(right)',
    'pfc':                'PFC',
    'coupling_hipp_pfc':  'Hipp-PFC\ncoupling',
}

# ── helper ───────────────────────────────────────────────────────────────────
def get_rhos(rows, model, roi):
    d = {r['run_label']: r for r in rows if r['model'] == model and r['roi'] == roi}
    rhos = [d[run]['rho_obs']         if run in d else float('nan') for run in RUNS]
    ps   = [d[run]['p_one_sided']     if run in d else float('nan') for run in RUNS]
    qs   = [d[run]['q_fdr_one_sided'] if run in d else float('nan') for run in RUNS]
    return np.array(rhos), np.array(ps), np.array(qs)


def fisher_stat(ps):
    ps = np.array([max(p, 1e-4) for p in ps if not math.isnan(p)])
    if len(ps) == 0:
        return float('nan')
    return -2.0 * np.sum(np.log(ps))


# ════════════════════════════════════════════════════════════════════════════
# Figure 1: Strip plot — rho per run, per primary model, for focus ROIs
# ════════════════════════════════════════════════════════════════════════════
print("Generating Fig 1: strip plot...")

fig, axes = plt.subplots(1, len(FOCUS_ROIS), figsize=(16, 5), sharey=False)
fig.suptitle('sub001_game2 — RSA rho per run (strip plot)', fontsize=13, fontweight='bold')

JITTER = 0.12
np.random.seed(42)
x_pos = np.arange(len(PRIMARY_MODELS))

for ax, roi in zip(axes, FOCUS_ROIS):
    for xi, model in enumerate(PRIMARY_MODELS):
        rhos, ps, qs = get_rhos(perm_rows, model, roi)
        valid = ~np.isnan(rhos)
        jitter = (np.random.rand(valid.sum()) - 0.5) * JITTER

        colors = ['#D62728' if q < 0.05 else '#AAAAAA' for q in qs[valid]]

        ax.scatter(xi + jitter, rhos[valid],
                   c=colors, s=45, zorder=3, edgecolors='none', alpha=0.85)

        mn = np.nanmean(rhos)
        ax.hlines(mn, xi - 0.28, xi + 0.28, colors=MODEL_COLORS[model],
                  linewidth=2.5, zorder=4)

    ax.set_title(ROI_LABELS[roi], fontsize=10, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([MODEL_LABELS[m] for m in PRIMARY_MODELS], fontsize=8)
    ax.set_ylabel('Spearman rho', fontsize=8)
    ax.axhline(0, color='black', linewidth=0.6, linestyle='--')
    ax.set_xlim(-0.6, len(PRIMARY_MODELS) - 0.4)
    ax.tick_params(axis='y', labelsize=8)

red_patch  = mpatches.Patch(color='#D62728', label='FDR q < 0.05')
grey_patch = mpatches.Patch(color='#AAAAAA', label='n.s.')
fig.legend(handles=[red_patch, grey_patch], loc='lower right', fontsize=9,
           title='Block perm', bbox_to_anchor=(1.0, 0.05))
plt.tight_layout(rect=[0, 0, 0.97, 1])
plt.savefig(f'{OUT_DIR}/new_fig1_strip_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("  saved new_fig1_strip_plot.png")

# ════════════════════════════════════════════════════════════════════════════
# Figure 2: 개선된 RSA heatmap
# ════════════════════════════════════════════════════════════════════════════
print("Generating Fig 2: improved heatmap...")

HEATMAP_MODELS = ['tree_reps_primary', 'tree_reps_s2only',
                  'im_vectors_primary', 'im_vp_vectors_primary', 'ram']
HEATMAP_ROIS   = ['hippocampus', 'right_hippocampus', 'pfc', 'coupling_hipp_pfc']

mean_rho_mat = np.zeros((len(HEATMAP_MODELS), len(HEATMAP_ROIS)))
sig_count    = np.zeros_like(mean_rho_mat, dtype=int)

for i, model in enumerate(HEATMAP_MODELS):
    for j, roi in enumerate(HEATMAP_ROIS):
        rhos, ps, qs = get_rhos(perm_rows, model, roi)
        mean_rho_mat[i, j] = np.nanmean(rhos)
        sig_count[i, j]    = int(np.sum(qs < 0.05))

fig, ax = plt.subplots(figsize=(9, 5))
im = ax.imshow(mean_rho_mat, aspect='auto', cmap='Reds', vmin=0, vmax=0.10)
plt.colorbar(im, ax=ax, label='Mean Spearman rho (across runs)')

ax.set_xticks(range(len(HEATMAP_ROIS)))
ax.set_xticklabels([ROI_LABELS[r].replace('\n', ' ') for r in HEATMAP_ROIS], fontsize=10)
ax.set_yticks(range(len(HEATMAP_MODELS)))
ax.set_yticklabels([MODEL_LABELS[m].replace('\n', ' ') for m in HEATMAP_MODELS], fontsize=10)

for i in range(len(HEATMAP_MODELS)):
    for j in range(len(HEATMAP_ROIS)):
        val = mean_rho_mat[i, j]
        sig = sig_count[i, j]
        star = '' if sig == 0 else ('  ★' if sig >= 2 else '  ·')
        txt = f'{val:.3f}{star}\n({sig}/11 sig)' if sig > 0 else f'{val:.3f}'
        ax.text(j, i, txt, ha='center', va='center', fontsize=9,
                color='white' if val > 0.065 else 'black')

ax.set_title(
    'sub001_game2 — Mean RSA rho  |  Color: [0, 0.10]\n'
    'Numbers: mean rho  |  ★ = FDR q<0.05 in >=2 runs  |  (N/11 sig runs)',
    fontsize=10
)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/new_fig2_heatmap_scaled.png', dpi=150, bbox_inches='tight')
plt.close()
print("  saved new_fig2_heatmap_scaled.png")

# ════════════════════════════════════════════════════════════════════════════
# Figure 3: Fisher combined chi2 stat
# ════════════════════════════════════════════════════════════════════════════
print("Generating Fig 3: Fisher combined chi2...")

CHI2_THRESH = stats.chi2.ppf(0.95, df=2 * len(RUNS))  # ~33.92

fig, axes = plt.subplots(1, len(FOCUS_ROIS), figsize=(16, 4.5), sharey=True)
fig.suptitle(
    f'sub001_game2 — Fisher combined chi2 (block-perm p-values, k={len(RUNS)} runs)\n'
    f'Threshold chi2(df={2*len(RUNS)}) = {CHI2_THRESH:.1f}  ->  combined p < 0.05',
    fontsize=11, fontweight='bold'
)

for ax, roi in zip(axes, FOCUS_ROIS):
    fstats = []
    for model in PRIMARY_MODELS:
        rhos, ps, qs = get_rhos(perm_rows, model, roi)
        fstats.append(fisher_stat(ps))

    colors_bar = ['#D62728' if fs > CHI2_THRESH else '#AAAAAA' for fs in fstats]
    ax.bar(range(len(PRIMARY_MODELS)), fstats, color=colors_bar,
           edgecolor='white', linewidth=0.5)

    ax.axhline(CHI2_THRESH, color='black', linewidth=1.5, linestyle='--', zorder=5)
    ax.text(len(PRIMARY_MODELS) - 0.5, CHI2_THRESH + 0.5,
            f'{CHI2_THRESH:.1f}', ha='right', fontsize=8)

    ax.set_title(ROI_LABELS[roi], fontsize=10, fontweight='bold')
    ax.set_xticks(range(len(PRIMARY_MODELS)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in PRIMARY_MODELS], fontsize=8)
    ax.set_ylabel('Fisher chi2 stat', fontsize=8)

    for xi, fs in enumerate(fstats):
        ax.text(xi, fs + 0.5, f'{fs:.0f}', ha='center', va='bottom', fontsize=8)

red_patch  = mpatches.Patch(color='#D62728', label='Combined p < 0.05')
grey_patch = mpatches.Patch(color='#AAAAAA', label='n.s. combined')
fig.legend(handles=[red_patch, grey_patch], loc='lower right', fontsize=9,
           bbox_to_anchor=(1.0, 0.05))
plt.tight_layout(rect=[0, 0, 0.97, 1])
plt.savefig(f'{OUT_DIR}/new_fig3_fisher_combined.png', dpi=150, bbox_inches='tight')
plt.close()
print("  saved new_fig3_fisher_combined.png")

# ════════════════════════════════════════════════════════════════════════════
# Figure 4: tree_reps 런별 rho — 어느 런이 유의미한가?
# ════════════════════════════════════════════════════════════════════════════
print("Generating Fig 4: tree_reps per-run detail...")

DETAIL_MODELS = ['tree_reps_primary', 'tree_reps_s2only']
DETAIL_ROIS   = ['hippocampus', 'right_hippocampus', 'pfc']

fig, axes = plt.subplots(len(DETAIL_MODELS), len(DETAIL_ROIS),
                         figsize=(14, 7), sharey='row')
fig.suptitle('sub001_game2 — tree_reps: rho per run (bar = FDR sig, orange = nominal p<0.05)',
             fontsize=12, fontweight='bold')

for row_i, model in enumerate(DETAIL_MODELS):
    for col_j, roi in enumerate(DETAIL_ROIS):
        ax = axes[row_i][col_j]
        rhos, ps, qs = get_rhos(perm_rows, model, roi)

        bar_colors = []
        for q, p in zip(qs, ps):
            if q < 0.05:
                bar_colors.append('#D62728')
            elif p < 0.05:
                bar_colors.append('#FF7F0E')
            else:
                bar_colors.append('#CCCCCC')

        x = np.arange(len(RUNS))
        ax.bar(x, rhos, color=bar_colors, edgecolor='white', linewidth=0.3)
        mean_r = np.nanmean(rhos)
        ax.axhline(mean_r, color='steelblue', linewidth=1.5, linestyle='--')
        ax.axhline(0, color='black', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(RUN_SHORT, rotation=45, ha='right', fontsize=7)
        ax.tick_params(axis='y', labelsize=8)
        ax.set_ylabel('rho_obs', fontsize=8)

        mlabel = MODEL_LABELS[model].replace('\n', ' ')
        rlabel = ROI_LABELS[roi].replace('\n', ' ')
        ax.set_title(f'{mlabel} | {rlabel}\nmean={mean_r:.3f}', fontsize=9)

        for xi, (rho, q, p) in enumerate(zip(rhos, qs, ps)):
            lbl = '*' if q < 0.05 else ('+' if p < 0.05 else '')
            if lbl:
                ax.text(xi, rho + 0.001, lbl, ha='center', va='bottom',
                        fontsize=11, color='black', fontweight='bold')

patches = [
    mpatches.Patch(color='#D62728', label='FDR q < 0.05 (*)'),
    mpatches.Patch(color='#FF7F0E', label='nominal p < 0.05 (+)'),
    mpatches.Patch(color='#CCCCCC', label='n.s.'),
]
fig.legend(handles=patches, loc='lower right', fontsize=9, bbox_to_anchor=(1.0, 0.01))
plt.tight_layout(rect=[0, 0.03, 0.97, 1])
plt.savefig(f'{OUT_DIR}/new_fig4_tree_reps_per_run.png', dpi=150, bbox_inches='tight')
plt.close()
print("  saved new_fig4_tree_reps_per_run.png")

# ════════════════════════════════════════════════════════════════════════════
# Figure 5: Encoding — RAM vs RAM+Thinker
# ════════════════════════════════════════════════════════════════════════════
print("Generating Fig 5: encoding comparison...")

ENC_MODELS_ORDER = [
    'thinker_tree_reps_primary',
    'thinker_im_vectors_primary',
    'thinker_im_vp_vectors_primary',
    'ram',
    'ram_plus_tree_reps_primary',
    'ram_plus_im_vectors_primary',
    'ram_plus_im_vp_vectors_primary',
]
ENC_MODEL_LABELS = {
    'thinker_tree_reps_primary':      'Thinker\ntree_reps',
    'thinker_im_vectors_primary':     'Thinker\nim_vectors',
    'thinker_im_vp_vectors_primary':  'Thinker\nim_vp',
    'ram':                            'RAM\n(baseline)',
    'ram_plus_tree_reps_primary':     'RAM +\ntree_reps',
    'ram_plus_im_vectors_primary':    'RAM +\nim_vectors',
    'ram_plus_im_vp_vectors_primary': 'RAM +\nim_vp',
}
ENC_COLORS = {
    'thinker_tree_reps_primary':      '#92C5DE',
    'thinker_im_vectors_primary':     '#F4A582',
    'thinker_im_vp_vectors_primary':  '#B8E186',
    'ram':                            '#7B3294',
    'ram_plus_tree_reps_primary':     '#2166AC',
    'ram_plus_im_vectors_primary':    '#D6604D',
    'ram_plus_im_vp_vectors_primary': '#4DAC26',
}
ENC_TARGET_ROIS   = ['hippocampus', 'right_hippocampus', 'pfc']
ENC_TARGET_LABELS = {
    'hippocampus':       'Hippocampus (bilateral)',
    'right_hippocampus': 'Right Hippocampus',
    'pfc':               'PFC',
}

fig, axes = plt.subplots(1, len(ENC_TARGET_ROIS), figsize=(16, 5))
fig.suptitle('sub001_game2 — Voxelwise encoding (Pearson r, within-run CV)\nMean across runs +/- 1 SE  |  dashed = RAM baseline',
             fontsize=12, fontweight='bold')

for ax, target_roi in zip(axes, ENC_TARGET_ROIS):
    means, stes = [], []
    for model in ENC_MODELS_ORDER:
        vals = [r['mean_r']
                for r in enc_rows
                if r['model'] == model and r['roi'] == target_roi
                and r['cv_scheme'] == 'within_run_block']
        if vals:
            m = np.mean(vals)
            se = np.std(vals, ddof=1) / math.sqrt(len(vals)) if len(vals) > 1 else 0
        else:
            m, se = float('nan'), 0.0
        means.append(m)
        stes.append(se)

    x = np.arange(len(ENC_MODELS_ORDER))
    bar_colors = [ENC_COLORS[m] for m in ENC_MODELS_ORDER]
    ax.bar(x, means, yerr=stes, color=bar_colors, capsize=4,
           edgecolor='white', linewidth=0.5, error_kw={'linewidth': 1.2})
    ax.axhline(0, color='black', linewidth=0.5)

    ram_idx = ENC_MODELS_ORDER.index('ram')
    ram_mean = means[ram_idx]
    ax.axhline(ram_mean, color='#7B3294', linewidth=1.2, linestyle='--', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([ENC_MODEL_LABELS[m] for m in ENC_MODELS_ORDER],
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean Pearson r', fontsize=9)
    ax.set_title(ENC_TARGET_LABELS[target_roi], fontsize=10, fontweight='bold')
    ax.tick_params(axis='y', labelsize=8)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/new_fig5_encoding_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  saved new_fig5_encoding_comparison.png")

print("\nAll done. Figures saved to:", OUT_DIR)
