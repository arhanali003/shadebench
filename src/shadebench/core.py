"""Fixed-feature collision experiment with an explicit pixel constraint."""
import hashlib
import io
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

SIZE = 32


def projection(seed=0):
    """Deterministic random orthonormal features, not a pretrained vision encoder."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(SIZE * SIZE, 32)))
    return q.T


def pixels(array):
    array = np.asarray(array, dtype=np.float64)
    if array.shape != (SIZE, SIZE) or not np.isfinite(array).all():
        raise ValueError('Expected a finite 32 by 32 grayscale array')
    if np.any(array < 0) or np.any(array > 1):
        raise ValueError('Pixel values must lie in [0,1]')
    return array


def optimize(source, target, features, epsilon=8 / 255, steps=80):
    source, target = pixels(source), pixels(target)
    features = np.asarray(features, dtype=np.float64)
    if features.shape != (32, SIZE * SIZE) or not np.isfinite(features).all():
        raise ValueError('Expected finite feature matrix of shape (32,1024)')
    if not math.isfinite(epsilon) or not 0 <= epsilon <= 32 / 255:
        raise ValueError('epsilon must be between 0 and 32/255')
    if type(steps) is not int or not 1 <= steps <= 1000:
        raise ValueError('steps must be an integer between 1 and 1000')
    x, goal = source.ravel().copy(), features @ target.ravel()
    lower, upper = np.maximum(0, x - epsilon), np.minimum(1, x + epsilon)
    # Lipschitz constant for gradient of squared feature distance.
    spectral = np.linalg.norm(features, ord=2)
    rate = 1 / max(2 * spectral * spectral, 1e-12)
    history = []
    for _ in range(steps):
        residual = features @ x - goal
        history.append(float(residual @ residual))
        x = np.clip(x - rate * 2 * features.T @ residual, lower, upper)
    residual = features @ x - goal
    history.append(float(residual @ residual))
    return x.reshape(source.shape), history


def quantize(array):
    return np.rint(pixels(array) * 255).astype(np.uint8)


def read_image(path):
    """No implicit resizing or frame selection: keep preprocessing explicit."""
    with Image.open(path) as image:
        if image.size != (SIZE, SIZE) or getattr(image, 'n_frames', 1) != 1:
            raise ValueError('Input must be a single-frame 32x32 image; prepare a copy explicitly')
        return np.asarray(image.convert('L'), dtype=np.float64) / 255


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(source, candidate, target, features):
    delta = pixels(candidate) - pixels(source)
    mse = float(np.mean(delta * delta))
    return {'linf': float(np.max(np.abs(delta))), 'rmse': math.sqrt(mse),
            'psnr_db': None if mse == 0 else -10 * math.log10(mse),
            'identical': mse == 0,
            'target_feature_distance': float(np.linalg.norm(features @ (candidate - target).ravel()))}


def transformations(array):
    image = Image.fromarray(quantize(array))
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=85)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        jpeg = np.asarray(compressed, dtype=np.float64) / 255
    resized = image.resize((16, 16), Image.Resampling.BILINEAR).resize((32, 32), Image.Resampling.BILINEAR)
    return {'jpeg85': jpeg, 'resize_half': np.asarray(resized, dtype=np.float64) / 255,
            'median3': np.asarray(image.filter(ImageFilter.MedianFilter(3)), dtype=np.float64) / 255}


def fixture(label, rng):
    """Original synthetic stripes; contrast and background vary independently."""
    y, x = np.mgrid[:SIZE, :SIZE]
    center = rng.uniform(12, 20)
    width = rng.uniform(3, 6)
    axis = x if label == 0 else y
    arr = rng.uniform(.25, .4) + rng.uniform(.2, .35) * np.exp(-((axis - center) / width) ** 2)
    return np.clip(arr + rng.normal(0, .015, arr.shape), 0, 1)


def fit(train, labels, features):
    z = np.stack([features @ image.ravel() for image in train])
    z = np.column_stack([z, np.ones(len(z))])
    penalty = np.eye(z.shape[1]) * .01
    penalty[-1, -1] = 0
    return np.linalg.solve(z.T @ z + penalty, z.T @ np.eye(2)[labels])


def accuracy(weights, images, labels, features):
    z = np.stack([features @ image.ravel() for image in images])
    predicted = (np.column_stack([z, np.ones(len(z))]) @ weights).argmax(axis=1)
    return {'overall': float(np.mean(predicted == labels)),
            'source_class': float(np.mean(predicted[labels == 0] == 0)),
            'target_class': float(np.mean(predicted[labels == 1] == 1))}


def training_experiment(seed, features, epsilon, steps):
    rng = np.random.default_rng(seed + 100)
    labels = np.repeat(np.arange(2), 16)
    train = np.stack([fixture(label, rng) for label in labels])
    test_labels = np.repeat(np.arange(2), 40)
    test = np.stack([fixture(label, rng) for label in test_labels])
    altered, random = train.copy(), train.copy()
    for i in range(8):
        altered[i], _ = optimize(train[i], train[16 + i], features, epsilon, steps)
        random[i] = np.clip(train[i] + rng.uniform(-epsilon, epsilon, train[i].shape), 0, 1)
    # The retraining comparison uses the same 8-bit representation as file exports.
    arms = {name: np.stack([quantize(x) / 255 for x in images])
            for name, images in [('clean', train), ('random', random), ('optimized', altered)]}
    test = np.stack([quantize(x) / 255 for x in test])
    return {'train_count': 32, 'test_count': 80, 'modified_count': 8,
            'labels_unchanged': True, 'model': 'ridge classifier on fixed random features',
            'arms': {name: accuracy(fit(images, labels, features), test, test_labels, features)
                     for name, images in arms.items()}}
