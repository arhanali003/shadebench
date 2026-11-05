"""Render committed experiment measurements; requires optional matplotlib."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
rows = json.loads((root / 'examples/results.json').read_text())
fig, ax = plt.subplots(figsize=(10, 4.5), layout='constrained')
fig.set_facecolor('#f4f0e8')
ax.set_facecolor('#f4f0e8')
styles = [('clean', '#203c2a', 'o', '-'), ('random', '#8a9063', 'x', '--'), ('optimized', '#b44e26', 's', '-')]
for arm, color, marker, line in styles:
    ax.plot([r['seed'] for r in rows], [r['training'][arm]['overall'] * 100 for r in rows],
            label=arm, color=color, marker=marker, linestyle=line, linewidth=2, markersize=8)
ax.set(ylim=(0, 105), xticks=range(5), xlabel='Seed', ylabel='Held-out accuracy (%)',
       title='Synthetic classifier: feature shifts do not guarantee training effects')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', alpha=.2)
ax.legend(loc='lower left', frameon=False)
fig.savefig(root / 'assets/results.png', dpi=160)
