"""Local-only experiment runner. No model downloads or network calls."""
import argparse
import base64
import html
import json
import platform
import sys
from pathlib import Path

import numpy as np
from PIL import Image, __version__ as pillow_version

from . import __version__
from .core import (projection, optimize, quantize, read_image, digest, metrics,
                   transformations, fixture, training_experiment)


def render_report(report, output):
    cards = []
    for name in ('source', 'target', 'random', 'optimized'):
        encoded = base64.b64encode((output / (name + '.png')).read_bytes()).decode()
        cards.append(f'<figure><img src="data:image/png;base64,{encoded}" alt="{name}"><figcaption>{name}</figcaption></figure>')
    payload = html.escape(json.dumps(report, indent=2, allow_nan=False))
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Shadebench experiment</title><style>body{max-width:1000px;margin:48px auto;padding:24px;background:#f4f0e8;color:#222;font:17px system-ui}h1{font-size:48px}section{display:flex;flex-wrap:wrap;gap:24px}figure{margin:0}img{width:192px;image-rendering:pixelated}figcaption{padding:12px 0}pre{background:#fff;padding:24px;overflow:auto;font-size:13px}aside{border-left:4px solid #70694d;padding:16px}</style>
<h1>Shadebench / experiment</h1><p>Read the result before trusting the effect.</p>
<aside>Educational fixed-feature experiment. Not Nightshade, not diffusion-model validation, and not evidence that an artwork is protected. Larger displayed pixels are intentional.</aside><section>''' + ''.join(cards) + '</section><h2>Recorded measurements</h2><pre>' + payload + '</pre></html>'


def run(output, seed=0, epsilon_pixels=8, steps=80, source_path=None, target_path=None):
    if type(seed) is not int or not 0 <= seed < 2**32 - 100:
        raise ValueError('seed must be an integer in [0, 2**32-100)')
    if type(epsilon_pixels) is not int or not 0 <= epsilon_pixels <= 32:
        raise ValueError('epsilon-pixels must be an integer from 0 to 32')
    if (source_path is None) != (target_path is None):
        raise ValueError('Provide both source and target, or neither for synthetic demo')
    rng = np.random.default_rng(seed)
    features = projection(seed)
    source = read_image(source_path) if source_path else quantize(fixture(0, rng)) / 255
    target = read_image(target_path) if target_path else quantize(fixture(1, rng)) / 255
    epsilon = epsilon_pixels / 255
    optimized, history = optimize(source, target, features, epsilon, steps)
    random = np.clip(source + rng.uniform(-epsilon, epsilon, source.shape), 0, 1)
    output = Path(output)
    # Exclusive directory creation prevents accidental replacement of evidence.
    output.mkdir(parents=True, exist_ok=False)
    arrays = {'source': source, 'target': target, 'random': random, 'optimized': optimized}
    for name, array in arrays.items():
        Image.fromarray(quantize(array)).save(output / (name + '.png'))
    stored = {name: read_image(output / (name + '.png')) for name in arrays}
    if np.max(np.abs(quantize(stored['source']).astype(int) - quantize(stored['optimized']).astype(int))) > epsilon_pixels:
        raise RuntimeError('Exported pixel bound failed')
    report = {'schema_version': 1, 'software': __version__, 'seed': seed,
              'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pillow': pillow_version},
              'scope': 'synthetic fixed-feature study; not diffusion or artist-protection validation',
              'parameters': {'epsilon_pixels': epsilon_pixels, 'steps': steps, 'size': 32, 'features': 32},
              'optimization': {'initial_loss': history[0], 'final_loss': history[-1], 'history': history},
              'images': {name: {'sha256': digest(output / (name + '.png')),
                                **metrics(stored['source'], array, stored['target'], features)}
                         for name, array in stored.items()},
              'transformations': {arm: {name: metrics(stored['source'], value, stored['target'], features)
                                       for name, value in transformations(stored[arm]).items()}
                                  for arm in ('source', 'random', 'optimized')},
              'limitations': ['Random linear encoder; no semantic pretrained model.',
                              'No perceptual study; pixel bounds do not establish invisibility.',
                              'Transformation distances do not prove training-time robustness.',
                              'Single seed is not statistical evidence of generality.']}
    if source_path is None:
        report['training'] = training_experiment(seed, features, epsilon, steps)
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (output / 'index.html').write_text(render_report(report, output), encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epsilon-pixels', type=int, default=8)
    parser.add_argument('--steps', type=int, default=80)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--target', type=Path)
    args = parser.parse_args(argv)
    try:
        report = run(args.output, args.seed, args.epsilon_pixels, args.steps, args.source, args.target)
    except (OSError, ValueError, RuntimeError) as error:
        print(json.dumps({'status': 'error', 'message': str(error)}), file=sys.stderr)
        return 1
    print(json.dumps({'status': 'ok', 'output': str(args.output), 'final_loss': report['optimization']['final_loss']}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
