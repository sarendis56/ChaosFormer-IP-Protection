import json

from src.experiments.layer_subset_common import (
    layer_subset_descriptors,
    preregistered_imagenet_shards,
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
    assert sum(spec.family == "spread" for spec in first) == 6
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
    assert manifest["training_protocol"]["full_deit"]["seeds"] == [4101, 4102, 4103]
    assert manifest["training_protocol"]["oracle_deit"]["seeds"] == [3101]
    assert manifest["training_protocol"]["selection_uses_training_outcomes"] is False


def test_retraining_schedule_warms_up_then_decays():
    from src.experiments.layer_subset_retraining_experiment import cosine_schedule

    values = [cosine_schedule(step, total_steps=100, warmup_steps=10) for step in range(100)]
    assert values[0] < values[9]
    assert values[9] == 1.0
    assert values[10] == 1.0
    assert values[-1] < values[50]
    assert values[-1] >= 0.01


def test_soft_cross_entropy_accepts_hard_and_mixed_targets():
    import torch
    from src.experiments.layer_subset_retraining_experiment import soft_cross_entropy

    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    hard = torch.tensor([0, 1])
    mixed = torch.tensor([[0.75, 0.25], [0.25, 0.75]])
    assert torch.isfinite(soft_cross_entropy(logits, hard, 0.1))
    assert torch.isfinite(soft_cross_entropy(logits, mixed, 0.1))


def test_preregistered_launcher_phase_sizes():
    from src.experiments.run_layer_subset_study import jobs_for_phase

    assert len(jobs_for_phase("inference")) == 4
    assert len(jobs_for_phase("short")) == 17
    assert len(jobs_for_phase("full")) == 18
    assert len(jobs_for_phase("oracle")) == 6
    assert len(jobs_for_phase("vit_confirm")) == 6


def test_analysis_helpers_summarize_and_correlate():
    from src.experiments.analyze_layer_subset_study import aggregate, correlation

    rows = [{"x": 1, "y": 2}, {"x": 2, "y": 4}, {"x": 3, "y": 6}]
    summary = aggregate(rows, ("y",))
    assert summary["count"] == 3
    assert summary["y"]["mean"] == 4
    assert correlation(rows, "x", "y")["spearman_rho"] == 1.0


def test_imagenet_shard_preregistration_is_unique_and_deterministic():
    first = preregistered_imagenet_shards(seed=20260711)
    second = preregistered_imagenet_shards(seed=20260711)
    assert first == second
    assert len(first) == len(set(first)) == 60
    assert all(name.startswith("train-") and name.endswith("-of-00294.parquet") for name in first)
