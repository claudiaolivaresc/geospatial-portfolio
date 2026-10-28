"""Tests for harmonisation, spatial split, leakage safety and evaluation."""

import numpy as np
import pytest

from vod_clustering import (
    baseline_quantile_labels,
    build_pipeline,
    choose_k,
    evaluate_assignments,
    harmonise,
    make_scene,
    seed_stability,
    spatial_split,
)
from vod_clustering.synthetic import BAND_NAMES, CLASS_NAMES


def test_scene_is_seeded_and_has_all_classes():
    a = make_scene((40, 40), seed=3)
    b = make_scene((40, 40), seed=3)
    np.testing.assert_array_equal(a.land_cover, b.land_cover)
    np.testing.assert_array_equal(np.nan_to_num(a.bands), np.nan_to_num(b.bands))
    assert set(np.unique(a.land_cover)) == set(range(len(CLASS_NAMES)))
    assert a.bands.shape == (len(BAND_NAMES), 40, 40)
    assert 0.0 < np.isnan(a.bands).mean() < 0.06  # gaps near the configured 3%


def test_harmonise_drops_nan_and_out_of_range_pixels():
    bands = np.full((2, 2, 2), 0.5)
    bands[0, 0, 0] = np.nan  # missing band value
    bands[1, 1, 1] = 9.0  # out of physical range
    features, valid = harmonise(bands)
    assert valid.sum() == 2
    assert not valid[0, 0] and not valid[1, 1]
    assert features.shape == (2, 2)
    np.testing.assert_allclose(features, np.log(0.5))


def test_harmonise_rejects_wrong_dimensionality():
    with pytest.raises(ValueError):
        harmonise(np.zeros((4, 4)))


def test_spatial_split_is_disjoint_and_respects_buffer():
    train, test = spatial_split((20, 40), train_frac=0.5, buffer_cols=3)
    assert not np.any(train & test)
    cols_train = np.where(train.any(axis=0))[0]
    cols_test = np.where(test.any(axis=0))[0]
    # Gap between the last train column and the first test column must exceed the buffer.
    assert cols_test.min() - cols_train.max() > 3


def test_spatial_split_rejects_oversized_buffer():
    with pytest.raises(ValueError):
        spatial_split((10, 10), train_frac=0.5, buffer_cols=6)


def test_pipeline_scaler_is_fitted_on_training_data_only():
    rng = np.random.default_rng(0)
    X_train = rng.normal(0.0, 1.0, (500, 3))
    X_test = rng.normal(50.0, 1.0, (500, 3))  # very different distribution
    pipe = build_pipeline(3, seed=0).fit(X_train)
    np.testing.assert_allclose(pipe.named_steps["scale"].mean_, X_train.mean(axis=0))
    # Predicting on test must not change the fitted scaler.
    before = pipe.named_steps["scale"].mean_.copy()
    pipe.predict(X_test)
    np.testing.assert_array_equal(pipe.named_steps["scale"].mean_, before)


def test_ari_is_one_for_identical_partitions_and_seed_stability_is_high():
    scene = make_scene((60, 60), seed=1)
    feats, valid = harmonise(scene.bands)
    truth = scene.land_cover[valid]
    train_mask, test_mask = spatial_split(scene.land_cover.shape, 0.5, 3)
    X_train = feats[train_mask[valid]]
    X_test = feats[test_mask[valid]]
    y_test = truth[test_mask[valid]]

    pipe = build_pipeline(4, seed=0).fit(X_train)
    Z_test = pipe[:-1].transform(X_test)
    metrics = evaluate_assignments(y_test, y_test, Z_test)
    assert metrics["ARI_vs_land_cover"] == pytest.approx(1.0)

    assert seed_stability(X_train, X_test, k=4, seeds=(0, 1)) > 0.9


def test_baseline_returns_k_classes_on_test_data():
    rng = np.random.default_rng(2)
    X_train = rng.uniform(size=(200, 3))
    X_test = rng.uniform(size=(100, 3))
    labels = baseline_quantile_labels(X_train, X_test, band=1, k=4)
    assert set(np.unique(labels)) <= {0, 1, 2, 3}
    assert len(labels) == 100


def test_choose_k_returns_one_score_per_candidate():
    scene = make_scene((40, 40), seed=4)
    feats, _ = harmonise(scene.bands)
    scores = choose_k(feats, ks=[2, 3, 4], seed=0, sample_size=500)
    assert set(scores) == {2, 3, 4}
    assert all(-1.0 <= s <= 1.0 for s in scores.values())
