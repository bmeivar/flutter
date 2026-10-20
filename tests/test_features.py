import numpy as np
import pytest

from flutter_precursor import flutterFeatureExtractor

WINDOW = 128


def make_extractor(calibration_flag=False):
    return flutterFeatureExtractor("test", WINDOW, 0, calibrationFlag=calibration_flag)


def synthetic_segment(xi, n=1000, freq=0.5, fs=100, seed=0):
    """Sine (A=2) and white noise (sigma=1) mixed by xi, as in the paper."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    return xi * 2 * np.sin(2 * np.pi * freq * t) + (1 - xi) * rng.normal(size=n)


def stream(signal):
    ext = make_extractor()
    for v in signal:
        ext.get_measurement(v)
    return ext


def test_window_uses_all_samples(monkeypatch):
    ext = make_extractor()
    seen = []
    monkeypatch.setattr(ext, "calcArCod", lambda s: seen.append(np.array(s)))
    signal = np.arange(300, dtype=float)
    for v in signal:
        ext.get_measurement(v)
    assert all(len(w) == WINDOW for w in seen)
    # the newest sample is part of the window
    np.testing.assert_array_equal(seen[-1], signal[-WINDOW:])


def test_ar_dataset_has_all_rows():
    ext = make_extractor()
    ext.signal_to_dataset(list(np.arange(WINDOW, dtype=float)))
    assert ext.arDataset.shape == (WINDOW - ext.numArFeatures, ext.numArFeatures)
    assert ext.arLabels.shape == (WINDOW - ext.numArFeatures,)


def test_spectral_entropy_same_in_calibration_mode():
    signal = synthetic_segment(0.3, n=WINDOW)
    online = make_extractor()
    online.calcSpectralEntropy(signal)
    calib = make_extractor(calibration_flag=True)
    assert calib.calcSpectralEntropy(signal) == pytest.approx(online.spectralEntropy[-1])


def test_spectral_entropy_finite_with_empty_bins():
    # pure tone on an FFT bin: all other bins are (numerically) zero
    n = np.arange(WINDOW)
    signal = np.sin(2 * np.pi * 8 * n / WINDOW)
    ext = make_extractor()
    ext.calcSpectralEntropy(signal)
    value = ext.spectralEntropy[-1]
    assert np.isfinite(value)
    assert 0 <= value < 0.2


@pytest.mark.parametrize("feature, increasing", [
    ("arCod", True),
    ("spectralEntropy", False),
    ("uval", False),
    ("lzc", False),
])
def test_features_track_regularity(feature, increasing):
    """Paper Fig. 7: features change monotonically from noise to sine."""
    medians = [np.median(getattr(stream(synthetic_segment(xi)), feature))
               for xi in (0.0, 0.5, 1.0)]
    if increasing:
        assert medians[0] < medians[1] < medians[2]
    else:
        assert medians[0] > medians[1] > medians[2]


def test_arcod_range():
    noise = np.median(stream(synthetic_segment(0.0)).arCod)
    sine = np.median(stream(synthetic_segment(1.0)).arCod)
    assert noise < 0.1
    assert sine > 0.95
