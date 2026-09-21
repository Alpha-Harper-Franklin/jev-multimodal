"""Generate the repository figure from completed measurement artifacts only."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--summary', required=True)
    p.add_argument('--output', required=True, help='Filename stem for PNG and SVG')
    a = p.parse_args()
    summary = json.loads(Path(a.summary).read_text())
    modes = ['independent', 'batched', 'shared']
    names = ['Independent', 'Ordinary batch', 'Shared prefix']
    colors = ['#64748b', '#3b82f6', '#0d9488']
    variants = ['random', 'popular', 'adversarial']
    rows = summary['modes']
    assert all(rows[m]['images'] == 500 and rows[m]['all']['n'] == 9000 for m in modes)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={'width_ratios': [1, 1.3]})
    latency = [rows[m]['latency_p50_ms'] for m in modes]
    axes[0].barh(names, latency, color=colors, height=.58)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, max(latency)*1.24)
    axes[0].set_xlabel('Median milliseconds per image / 18 questions')
    axes[0].set_title('Measured request latency', loc='left', fontweight='bold', pad=14)
    for y, value in enumerate(latency):
        axes[0].text(value+max(latency)*.025, y, f'{value:,.0f} ms', va='center')
    positions = np.arange(3)
    for i, (mode, name, color) in enumerate(zip(modes, names, colors)):
        values = [100*rows[mode]['variants'][v]['all']['f1'] for v in variants]
        bars = axes[1].bar(positions+(i-1)*.25, values, width=.23, label=name, color=color)
        for bar, value in zip(bars, values):
            axes[1].text(bar.get_x()+bar.get_width()/2, value+1, f'{value:.1f}', ha='center', fontsize=8)
    axes[1].set_xticks(positions, [v.capitalize() for v in variants])
    axes[1].set_ylim(0, 105)
    axes[1].set_yticks([0, 25, 50, 75, 100])
    axes[1].set_ylabel('F1 (%)')
    axes[1].set_title('All 3,000 questions per COCO variant', loc='left', fontweight='bold', pad=14)
    axes[1].legend(loc='lower center', ncol=3, bbox_to_anchor=(.5, -.28), frameon=False, fontsize=9)
    fig.suptitle('POPE COCO · 500 images · 9,000 questions', x=.08, ha='left', fontweight='bold', fontsize=16)
    fig.text(.08, .025, 'Qwen2.5-VL-7B · one RTX 4090 · batch size 8 · same prompts and FP32 readout · no training', fontsize=9, color='#475569')
    fig.tight_layout(rect=[.02, .13, 1, .93])
    target = Path(a.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target.with_suffix('.png'), dpi=180)
    fig.savefig(target.with_suffix('.svg'))
    print(json.dumps({'png': target.with_suffix('.png').name, 'svg': target.with_suffix('.svg').name}))


if __name__ == '__main__':
    main()
