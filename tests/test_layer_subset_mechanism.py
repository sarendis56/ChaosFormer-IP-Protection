import json

from src.experiments.layer_subset_common import (
    layer_subset_descriptors,
    preregistered_subsets,
    stratified_indices,
    write_preregistration,
)


def test_preregistered_subsets_are_deterministic_and_unique():
    first = preregistered_subsets(12, [4, 6], random_count=20, seed=20260711)
    second = preregistered_subsets(12, [4, 6], random_count=20, seed=20260711)

    assert first == second
    assert len({(len(spec.layers), spec.layers) for spec in first}) == len(first)
    assert sum(spec.family == "random" for spec in first) == 40
    assert all(len(spec.layers) in {4, 6} for spec in first)


def test_descriptors_distinguish_clustered_and_spread_layers():
    clustered = layer_subset_descriptors((0, 1, 2, 3), 12)
    spread = layer_subset_descriptors((0, 4, 7, 11), 12)

    assert spread["depth_span"] > clustered["depth_span"]
    assert spread["mean_pairwise_distance"] > clustered["mean_pairwise_distance"]
    assert spread["adjacent_pairs"] < clustered["adjacent_pairs"]
    assert spread["depth_thirds_covered"] > clustered["depth_thirds_covered"]


def test_stratified_indices_are_balanced_and_reproducible():
    labels = [label for label in range(10) for _ in range(20)]
    first = stratified_indices(labels, 50, seed=7)
    second = stratified_indices(labels, 50, seed=7)

    assert first == second
    assert len(first) == len(set(first)) == 50
    selected_labels = [labels[index] for index in first]
    assert {label: selected_labels.count(label) for label in range(10)} == {
        label: 5 for label in range(10)
    }


def test_manifest_records_outcome_independent_full_selection(tmp_path):
    path = tmp_path / "manifest.json"
    write_preregistration(
        path,
        models=["model-a"],
        k_values=[4, 6],
        random_count=3,
        seed=11,
    )
    manifest = json.loads(path.read_text())

    assert manifest["full_retraining_selection_rule"]["selection_uses_screening_outcomes"] is False
    assert manifest["full_retraining_selection_rule"]["random_indices"] == [0, 1, 2]
