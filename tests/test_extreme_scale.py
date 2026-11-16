"""Comprehensive pre-experiment test suite for `scripts/run_extreme_scale_showdown.py`."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import pytest
from scipy import stats
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_extreme_scale_showdown as ext

from emeans.ultra_emeans import UltraEMeans


class _StubApexEMeans:
    """Contract-faithful stub matching `@apex-solver`'s `#150` & `#154` `ApexEMeans` API."""

    def __init__(
        self,
        d: int,
        k: int,
        nepochs: float = 5.0,
        gear_mode: str = "auto",
        precision: str = "auto",
        devices: list[torch.device] | None = None,
        seed: int = 1234,
        **_: Any,
    ) -> None:
        self.d = int(d)
        self.k = int(k)
        self.nepochs = float(nepochs)
        self.gear_mode = gear_mode
        self.precision = precision
        self.devices = devices
        self.seed = int(seed)
        self.selected_gear = "multi_gpu" if (devices and len(devices) >= 2) else (
            "huge_k_ivf" if k >= 16_384 else ("small" if k <= 512 else "medium_fp8")
        )
        self.resolved_precision = "bf16" if k <= 512 else "croiss_fp8"
        self.init_time_sec = 0.0
        self.train_time_sec = 0.0
        self.total_time_sec = 0.0
        self.n_splits = 0
        self.n_protected = 0
        self.n_transplants = 0
        self.centroids: torch.Tensor | None = None

    def fit(
        self,
        x: torch.Tensor | list[torch.Tensor],
        init_centroids: torch.Tensor | None = None,
        capture_assignments: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(x, list):
            dev0 = x[0].device
            x_cat = torch.cat([s.to(dev0) for s in x], dim=0).contiguous()
        else:
            x_cat = x
        inner = UltraEMeans(d=self.d, k=self.k, nepochs=self.nepochs, wide_step_mult=2, seed=self.seed)
        c, a = inner.fit(x_cat, init_centroids=init_centroids, capture_assignments=capture_assignments)
        self.init_time_sec = float(inner.init_time_sec)
        self.train_time_sec = float(inner.train_time_sec)
        self.total_time_sec = self.init_time_sec + self.train_time_sec
        self.n_splits = int(inner.n_splits)
        self.n_protected = int(inner.n_protected)
        self.n_transplants = int(inner.n_transplants)
        self.centroids = c
        return c, a


# ==============================================================================
# 1. Safety, Ownership & Dedicated Shard Topology Tests
# ==============================================================================


def test_safety_guards_block_walnut_non_owned_hosts_and_tmux0() -> None:
    with pytest.raises(RuntimeError, match="SAFETY VIOLATION"):
        ext.assert_safe_fleet_target("walnut.stratabaremetal.com", "emeans_ext_s00")
    with pytest.raises(RuntimeError, match="OWNERSHIP VIOLATION"):
        ext.assert_safe_fleet_target("ripper", "emeans_ext_s00")
    with pytest.raises(RuntimeError, match="OWNERSHIP VIOLATION"):
        ext.assert_safe_fleet_target("loki", "emeans_ext_s00")
    with pytest.raises(RuntimeError, match="OWNERSHIP VIOLATION"):
        ext.assert_safe_fleet_target("foureyes", "emeans_ext_s00")
    with pytest.raises(RuntimeError, match="SAFETY VIOLATION"):
        ext.assert_safe_fleet_target("acorn.stratabaremetal.com", "0")
    with pytest.raises(RuntimeError, match="SAFETY VIOLATION"):
        ext.assert_safe_fleet_target("work", "other_session")

    # Valid dedicated hosts and session names pass cleanly
    ext.assert_safe_fleet_target("work", "emeans_ext_s00")
    ext.assert_safe_fleet_target("acorn.stratabaremetal.com", "emeans_ext_s02")
    ext.assert_safe_fleet_target("cashew.stratabaremetal.com", "emeans_ext_s07")


def test_dedicated_shard_topology_completeness_and_sku_pinning() -> None:
    cells = ext.build_scale_cells(smoke=False)
    shards = ext.build_dedicated_shards(seed_start=5000)
    assert len(cells) == 10
    assert len(shards) == 8

    # Stage 1 has 6 disjoint single-GPU shards across work:0,1 + acorn:0,1 + cashew:0,1
    s1 = [s for s in shards if s.stage == 1]
    s2 = [s for s in shards if s.stage == 2]
    assert len(s1) == 6
    assert len(s2) == 2
    assert len({(s.host, s.visible_gpus) for s in s1}) == 6
    assert {(s.host, s.visible_gpus) for s in s2} == {
        ("acorn.stratabaremetal.com", "0,1"),
        ("cashew.stratabaremetal.com", "0,1"),
    }

    # Verify exact seed coverage, 0 duplicates, single-SKU per cell, and 1G<->2G host pairing
    cell_by_id = {c.cell_id: c for c in cells}
    seen_cell_seeds: dict[str, list[int]] = {c.cell_id: [] for c in cells}
    cell_skus: dict[str, set[str]] = {}
    seed_host_1g: dict[tuple[str, int], str] = {}
    seed_host_2g: dict[tuple[str, int], str] = {}

    for sp in shards:
        for cid, s_tuple in sp.cell_seed_map.items():
            seen_cell_seeds[cid].extend(s_tuple)
            cell_skus.setdefault(cid, set()).add(sp.sku_tier)
            for sd in s_tuple:
                if cid in ("c4a_ext_hugek_2m_k65536_1gpu", "c4b_ext_10m_k16384_1gpu"):
                    stem = cid.replace("_1gpu", "")
                    seed_host_1g[(stem, sd)] = sp.host
                elif cid in ("c4c_ext_hugek_2m_k65536_2gpu", "c4d_ext_10m_k16384_2gpu"):
                    stem = cid.replace("c4c_", "c4a_").replace("c4d_", "c4b_").replace("_2gpu", "")
                    seed_host_2g[(stem, sd)] = sp.host

    for cid, c in cell_by_id.items():
        expected = [5000 + i for i in range(c.num_seeds)]
        assert sorted(seen_cell_seeds[cid]) == expected
        assert len(seen_cell_seeds[cid]) == len(set(seen_cell_seeds[cid]))
        assert cell_skus[cid] == {c.sku_tier}

    assert seed_host_1g == seed_host_2g


# ==============================================================================
# 2. Extreme Corpus Synthesis, Leak Guard & Shift-Equivariance Tests
# ==============================================================================


@pytest.mark.parametrize("d,k", [(17, 19), (100, 64), (129, 257)])
def test_extreme_corpus_determinism_shapes_and_prime_dims(d: int, k: int) -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr1, va1, te1, rc1, cut1 = ext.generate_extreme_multiscale_corpus_gpu(
        n_train=1027, n_val=131, n_test=137, d=d, k=k, seed=777, device=dev
    )
    tr2, va2, te2, rc2, cut2 = ext.generate_extreme_multiscale_corpus_gpu(
        n_train=1027, n_val=131, n_test=137, d=d, k=k, seed=777, device=dev
    )
    assert tr1.shape == (1027, d)
    assert va1.shape == (131, d)
    assert te1.shape == (137, d)
    assert rc1.shape[1] == d and rc1.shape[0] >= 4
    assert cut1 == cut2 > 0.0
    assert torch.equal(tr1, tr2)
    assert torch.equal(va1, va2)
    assert torch.equal(te1, te2)
    assert torch.equal(rc1, rc2)


def test_hash_leakage_detector_catches_injected_duplicates() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.Generator(device=dev).manual_seed(42)
    x_tr = torch.randn((512, 32), generator=g, device=dev)
    x_va = torch.randn((64, 32), generator=g, device=dev)
    x_te = torch.randn((64, 32), generator=g, device=dev)
    ext._verify_zero_split_hash_leakage(x_tr, x_va, x_te)

    x_te_leaked = x_te.clone()
    x_te_leaked[17] = x_tr[309]
    with pytest.raises(ValueError, match="DATA LEAKAGE"):
        ext._verify_zero_split_hash_leakage(x_tr, x_va, x_te_leaked)

    x_va_leaked = x_va.clone()
    x_va_leaked[3] = x_te[11]
    with pytest.raises(ValueError, match="DATA LEAKAGE"):
        ext._verify_zero_split_hash_leakage(x_tr, x_va_leaked, x_te)


def test_evaluator_coordinate_shift_invariance() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cell = ext.ScaleCellSpec(
        cell_id="shift_test",
        tier=4,
        tier_label="Shift Test",
        corpus_kind="extreme_zipf_rare_manifold",
        n_train=2048,
        n_val=256,
        n_test=256,
        d=64,
        k=64,
        num_gpus=1,
        geometry="euclidean",
        has_ivf_eval=False,
        has_rare_modes=True,
        corpus_seed=901,
        sku_tier="test",
        num_seeds=1,
    )
    corp = ext.load_scale_cell_data(cell, data_dir=ROOT / "outputs" / "phase4_million_scale" / "data", primary_device=dev)
    centroids = corp.x_train[:64].clone()
    m_base = ext._evaluate_fitted_centroids(cell, corp, centroids, device=dev)

    shift = torch.full((1, 64), 20_000.0, device=dev, dtype=torch.float32)
    corp_shifted = ext.LoadedScaleCorpus(
        name=corp.name,
        x_train=(corp.x_train.double() + shift.double()).float(),
        x_val=(corp.x_val.double() + shift.double()).float(),
        x_test=(corp.x_test.double() + shift.double()).float(),
        d=corp.d,
        geometry=corp.geometry,
        rare_centers=(corp.rare_centers.double() + shift.double()).float() if corp.rare_centers is not None else None,
        rare_cutoff_d2=corp.rare_cutoff_d2,
    )
    c_shifted = (centroids.double() + shift.double()).float()
    m_shift = ext._evaluate_fitted_centroids(cell, corp_shifted, c_shifted, device=dev)

    assert abs(m_base["test_mse"] - m_shift["test_mse"]) / max(m_base["test_mse"], 1e-9) < 1e-3
    assert abs(m_base["rare_preserved_frac"] - m_shift["rare_preserved_frac"]) < 1e-6
    assert m_base["empty_clusters"] == m_shift["empty_clusters"]


# ==============================================================================
# 3. Record Validator, Cold-JIT Spike Detector & Atomic JSON Tests
# ==============================================================================


def test_validator_rejects_nan_duplicates_confounding_and_cold_jit_detector(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ext.atomic_write_json(tmp_path / "bad.json", {"x": float("nan")})

    cells = [ext.build_scale_cells(smoke=True)[0]]
    budgets = (1.0, 2.0)
    recs = ext.execute_scale_cell(
        cells[0],
        seeds=[5000, 5001],
        budgets=budgets,
        wave="baselines",
        data_dir=ROOT / "outputs" / "phase4_million_scale" / "data",
        smoke=True,
    )
    ext.validate_wave_records(recs, cells, budgets, wave="baselines", seed_start=5000, apex_available=False, smoke=True)
    assert ext.detect_cold_jit_spikes(recs) == []

    # Duplicate record rejected
    with pytest.raises(RuntimeError, match="Duplicate record"):
        ext.validate_wave_records(
            recs + [copy.deepcopy(recs[0])],
            cells,
            budgets,
            wave="baselines",
            seed_start=5000,
            apex_available=False,
            smoke=True,
        )

    # Cross-GPU confounding on same (cell_id, seed) rejected
    confounded = copy.deepcopy(recs)
    confounded[0]["visible_gpu"] = "99"
    with pytest.raises(RuntimeError, match="CONFOUNDING VIOLATION"):
        ext.validate_wave_records(
            confounded, cells, budgets, wave="baselines", seed_start=5000, apex_available=False, smoke=True
        )

    # Cold-JIT spike detected when seed_order_idx==0 is artificially inflated by +500ms
    spiked = copy.deepcopy(recs)
    for r in spiked:
        if r["arm"] == "smeans" and r["nepochs"] == 1.0 and r["seed_order_idx"] == 0:
            r["train_time_sec"] += 0.600
            r["total_time_sec"] += 0.600
    detected = ext.detect_cold_jit_spikes(spiked)
    assert len(detected) == 1 and detected[0]["arm"] == "smeans"


# ==============================================================================
# 4. End-to-End Wave A -> Wave B Incremental Caching, Stats & Gitignore Hygiene
# ==============================================================================


def test_wave_a_then_wave_b_incremental_merge_stats_and_plots(tmp_path: Path) -> None:
    cells = ext.build_scale_cells(smoke=True)[:2]  # 1 tiny cell + 1 rare-outlier k=512 cell
    budgets = (1.0, 2.0)
    data_dir = ROOT / "outputs" / "phase4_million_scale" / "data"
    shards_dir = tmp_path / "shards"

    try:
        ext.set_apex_class_for_testing(_StubApexEMeans)

        # Simulate 2 shards: shard 0 runs seed 5000, shard 1 runs seed 5001
        for s_idx, sd in enumerate([5000, 5001]):
            recs_a: list[dict[str, Any]] = []
            for c in cells:
                recs_a.extend(ext.execute_scale_cell(c, [sd], budgets, wave="baselines", data_dir=data_dir, smoke=True))
                for r in recs_a:
                    if r["seed"] == sd:
                        r["seed_order_idx"] = s_idx
            ext.atomic_write_json(
                shards_dir / ext.shard_json_filename("baselines", s_idx, 2),
                {"shard_idx": s_idx, "records": recs_a},
            )

        # Wave A-only merge succeeds before Wave B runs
        payload_a = ext.merge_and_synthesize(tmp_path, cells, budgets, seed_start=5000, num_shards=2)
        assert payload_a["apex_included"] is False
        assert (tmp_path / "wave_a_baselines_results.json").exists()
        assert (tmp_path / "wave_a_baselines_ scale_ladder.png".replace(" ", "")).exists()

        # Now run Wave B ONLY (`--wave apex`) and merge with cached Wave A
        for s_idx, sd in enumerate([5000, 5001]):
            recs_b: list[dict[str, Any]] = []
            for c in cells:
                recs_b.extend(ext.execute_scale_cell(c, [sd], budgets, wave="apex", data_dir=data_dir, smoke=True))
                for r in recs_b:
                    if r["seed"] == sd:
                        r["seed_order_idx"] = s_idx
            ext.atomic_write_json(
                shards_dir / ext.shard_json_filename("apex", s_idx, 2),
                {"shard_idx": s_idx, "records": recs_b},
            )

        payload_ab = ext.merge_and_synthesize(tmp_path, cells, budgets, seed_start=5000, num_shards=2)
        assert payload_ab["apex_included"] is True
        assert (tmp_path / "extreme_scale_results.json").exists()
        flagship = tmp_path / "apex_4decade_scale_dominance.png"
        flagship_mirror = tmp_path / "plots" / "apex_4decade_scale_dominance.png"
        assert flagship.exists() and flagship_mirror.exists()
        with Image.open(flagship) as im:
            im.verify()

        # Independently verify paired Student-t p-values against scipy.stats.ttest_rel
        recs_all = payload_ab["records"]
        for comp in payload_ab["comparisons"]:
            if comp["arm"] == "apex_emeans" and comp["baseline"] == "faiss_emeans" and comp["metric"] == "test_mse":
                cid = comp["cell_id"]
                bt = comp["nepochs"]
                a_v = [r["test_mse"] for r in sorted(recs_all, key=lambda x: x["seed"]) if r["cell_id"] == cid and r["arm"] == "apex_emeans" and r["nepochs"] == bt]
                b_v = [r["test_mse"] for r in sorted(recs_all, key=lambda x: x["seed"]) if r["cell_id"] == cid and r["arm"] == "faiss_emeans" and r["nepochs"] == bt]
                if np.std(np.array(a_v) - np.array(b_v), ddof=1) > 1e-15:
                    _, exp_p = stats.ttest_rel(a_v, b_v)
                    assert abs(comp["p_ttest"] - float(exp_p)) < 1e-12
    finally:
        ext.set_apex_class_for_testing(None)


def test_gitignore_allows_extreme_scale_plots_and_ignores_shards() -> None:
    res_keep = subprocess.run(
        ["git", "check-ignore", "outputs/extreme_scale/apex_4decade_scale_dominance.png"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_keep.returncode == 1, "Flagship PNG must NOT be gitignored!"

    res_ign = subprocess.run(
        ["git", "check-ignore", "outputs/extreme_scale/shards/wave_a_shard_00_of_08.json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_ign.returncode == 0, "Raw shard dumps in outputs/extreme_scale/shards/ MUST be gitignored!"


def test_panel_a_strict_x_monotonicity_and_panel_c_stationary_lloyd_anchor() -> None:
    cells = ext.build_scale_cells(smoke=False)
    cells_1gpu = [c for c in cells if c.num_gpus == 1]
    # Construct synthetic records including duplicate N*K*D cells (`c2a` and `c2b`) and improving T=1->2->5 MSE
    synth_recs: list[dict[str, Any]] = []
    for c in cells_1gpu:
        synth_recs.append(
            {
                "cell_id": c.cell_id,
                "arm": "lloyd_25",
                "nepochs": 25.0,
                "n_train": c.n_train,
                "k": c.k,
                "d": c.d,
                "total_time_sec": 1.0,
                "test_mse": 100.0,
                "selected_gear": "baseline",
            }
        )
        for b_t in (1.0, 2.0, 5.0):
            synth_recs.append(
                {
                    "cell_id": c.cell_id,
                    "arm": "smeans",
                    "nepochs": b_t,
                    "n_train": c.n_train,
                    "k": c.k,
                    "d": c.d,
                    "total_time_sec": 0.1 * b_t,
                    "test_mse": 105.0 - 5.0 * b_t,  # strictly improves 100 -> 95 -> 80 as T grows
                    "selected_gear": "baseline",
                }
            )
            synth_recs.append(
                {
                    "cell_id": c.cell_id,
                    "arm": "apex_emeans",
                    "nepochs": b_t,
                    "n_train": c.n_train,
                    "k": c.k,
                    "d": c.d,
                    "total_time_sec": 0.05 * b_t,
                    "test_mse": 95.0 - 5.0 * b_t,
                    "selected_gear": "small_bf16" if c.tier == 1 else ("huge_k_ivf" if c.k >= 16384 else "med_fp8"),
                }
            )

    xs_a, ys_a, gears_a = ext._compute_panel_a_series(synth_recs, cells_1gpu, "apex_emeans", max_t=5.0)
    assert len(xs_a) == 7  # 8 1-GPU cells with c2a + c2b merged cleanly into 1 N*K*D bucket
    assert all(xs_a[i] < xs_a[i + 1] for i in range(len(xs_a) - 1))
    assert len(ys_a) == len(gears_a) == 7

    lx, ly = ext._compute_panel_c_pareto_series(synth_recs, cells_1gpu, (1.0, 2.0, 5.0), "lloyd_25")
    assert lx == [1.0] and ly == [0.0]

    sx, sy = ext._compute_panel_c_pareto_series(synth_recs, cells_1gpu, (1.0, 2.0, 5.0), "smeans")
    assert len(sx) == 3 and sy[0] > sy[1] > sy[2]  # strictly downward as T increases!
    assert abs(sy[0] - 0.0) < 1e-9 and abs(sy[1] - (-5.0)) < 1e-9 and abs(sy[2] - (-20.0)) < 1e-9


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires >= 2 visible CUDA devices")
def test_multi_gpu_non_primary_device_warmup_and_nccl_cell() -> None:
    ext.warmup_global_kernels(torch.device("cuda:1"))
    c4_smoke = ext.build_scale_cells(smoke=True)[-1]
    assert c4_smoke.num_gpus == 2
    recs = ext.execute_scale_cell(
        c4_smoke,
        seeds=[5000, 5001],
        budgets=(1.0, 2.0),
        wave="baselines",
        data_dir=ROOT / "outputs" / "phase4_million_scale" / "data",
        smoke=True,
    )
    assert all(r["active_gpus"] == 2 for r in recs)
    assert ext.detect_cold_jit_spikes(recs) == []

