import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

from shadebench.core import projection, optimize, metrics, read_image, quantize, transformations
from shadebench.cli import run


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.features = projection(2)
        cls.source = np.full((32, 32), .4)
        cls.target = np.full((32, 32), .6)

    def test_projected_optimization_decreases_loss_with_bounds(self):
        output, history = optimize(self.source, self.target, self.features, 8/255, 15)
        self.assertLess(history[-1], history[0])
        self.assertTrue(np.all(np.diff(history) <= 1e-10))
        self.assertLessEqual(np.max(np.abs(output-self.source)), 8/255 + 1e-12)
        self.assertTrue(np.all((output >= 0) & (output <= 1)))

    def test_zero_budget_is_identity(self):
        output, history = optimize(self.source, self.target, self.features, 0, 3)
        np.testing.assert_array_equal(output, self.source)
        self.assertEqual(history[0], history[-1])

    def test_known_identity_features_solution(self):
        matrix = np.eye(1024)[:32]
        output, _ = optimize(self.source, self.target, matrix, .1, 1)
        np.testing.assert_allclose(output[0], .5)
        np.testing.assert_allclose(output[1:], .4)

    def test_invalid_inputs(self):
        for epsilon in (-1, float('nan'), float('inf'), .5):
            with self.assertRaises(ValueError):
                optimize(self.source, self.target, self.features, epsilon)
        for steps in (0, 1001, True, 1.2):
            with self.assertRaises(ValueError):
                optimize(self.source, self.target, self.features, steps=steps)
        for source in (np.zeros((3, 3)), np.full((32,32), np.nan), np.full((32,32), 2)):
            with self.assertRaises(ValueError):
                optimize(source, self.target, self.features)

    def test_metric_exact_values(self):
        result = metrics(self.source, self.target, self.target, self.features)
        self.assertAlmostEqual(result['rmse'], .2)
        self.assertAlmostEqual(result['linf'], .2)
        self.assertAlmostEqual(result['target_feature_distance'], 0)
        identical = metrics(self.source, self.source, self.target, self.features)
        self.assertIsNone(identical['psnr_db'])
        json.dumps(identical, allow_nan=False)

    def test_transformations_are_valid_arrays(self):
        for array in transformations(self.source).values():
            self.assertEqual(array.shape, (32,32))
            self.assertTrue(np.all((array >= 0) & (array <= 1)))

    def test_deterministic_features(self):
        np.testing.assert_array_equal(projection(2), self.features)
        self.assertFalse(np.array_equal(projection(3), self.features))

    def test_reject_implicit_resize(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'input.png'
            Image.new('RGB', (64,64)).save(path)
            with self.assertRaises(ValueError):
                read_image(path)

    def test_end_to_end_repeatable_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/'a', Path(tmp)/'b'
            first = run(a, seed=3, steps=3)
            second = run(b, seed=3, steps=3)
            self.assertEqual(first, second)
            self.assertEqual(set(first['training']['arms']), {'clean','random','optimized'})
            self.assertTrue(first['training']['labels_unchanged'])
            self.assertIn('not diffusion', (a/'index.html').read_text())
            self.assertLessEqual(first['images']['optimized']['linf'], 8/255 + 1e-12)
            with self.assertRaises(FileExistsError):
                run(a, steps=1)
            self.assertEqual(json.loads((a/'report.json').read_text()), first)

    def test_invalid_run_creates_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'result'
            for args in ({'seed': -1}, {'epsilon_pixels': 40}, {'steps': 0}, {'source_path': 'missing'}):
                with self.assertRaises(ValueError):
                    run(path, **args)
                self.assertFalse(path.exists())

    def test_cli_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable,'-m','shadebench.cli','--output',tmp,'--seed','-1'], capture_output=True,text=True)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(json.loads(result.stderr)['status'], 'error')

    def test_custom_images_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, array in [('source',self.source),('target',self.target)]:
                Image.fromarray(quantize(array)).save(root/(name+'.png'))
            before = (root/'source.png').read_bytes()
            report = run(root/'out', source_path=root/'source.png', target_path=root/'target.png', steps=2)
            self.assertNotIn('training', report)
            self.assertEqual(before,(root/'source.png').read_bytes())


if __name__ == '__main__':
    unittest.main()
