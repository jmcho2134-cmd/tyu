#!/usr/bin/env python
"""
verify_subgoal.py — Stage 5 검증 V1~V4, 게이트 G4·G5
================================================================================

V1  경계 커버리지     각 데모의 phase 경계 상태가 추론된 조건(전체, persistent
                      만이 아니라)을 실제로 만족하는가. 조건이 데모 산포를
                      담지 못하면 여기서 무너진다.
V2  순환성 ablation   boundary feature(분할에 쓴 채널)를 빼고 다시 추론해도
                      조건이 생존하는가. ▶ GATE G4.
                      object 모드에서 move→post 경계는 정의 자체가 객체
                      채널(정착)이므로 그 phase 의 탈락은 설계의 동어반복이지
                      결함이 아니다. G4 는 "객체 채널이 정의하지 않는" 경계 —
                      pre(파지)와 마지막 phase(종료 상태) — 의 생존을 요구하고,
                      move 는 정보로만 보고한다.
V3  phase 역행률      phase_of 를 데모 궤적에 흘렸을 때 라벨이 뒤로 가는
                      스텝 비율 < 5%. ▶ GATE G5. (+ 세그먼터 z 와 일치율 보고)
V4  leave-one-out     데모 하나를 빼고 추론한 조건이 남은 데모의 경계를
                      커버하는가 — 일반화 보고 (데모 수가 적을 때 참고용).

    python verify_subgoal.py               # cache + artifacts 로 전부 실행
    python verify_subgoal.py --selftest    # 합성 end-to-end (numpy만)
"""

import argparse
import json

import numpy as np

from extract_features import load_cache
from rho import Subgoal, condition_holds
from subgoal_discover import DEFAULTS, discover, boundary_state

GATE_G5 = 0.05


def _used_entries(entries, spec):
    ids = set(spec["meta"]["demos_used"])
    return [e for e in entries if e["demo_id"] in ids]


# ---------------------------------------------------------------------------
def v1_boundary_coverage(spec, entries, boundaries, pad=2):
    used = _used_entries(entries, spec)
    K = len(spec["meta"]["canonical_labels"])
    per_phase = {}
    for k in range(K):
        p = spec["phases"][str(k)]
        hit = 0
        for e in used:
            names = e["names"]
            t = (boundaries[e["demo_id"]]["bounds"][k]
                 if k < K - 1 else len(e["F"]) - 1)
            ok = all(condition_holds(
                boundary_state(e["F"], names.index(n), t, pad, tail=k == K - 1),
                ty, th, mu)
                     for n, ty, th, mu in zip(p["features"], p["type"],
                                              p["thresh"], p["mean"]))
            hit += int(ok)
        per_phase[p["label"]] = hit / len(used)
    cov = float(np.mean(list(per_phase.values())))
    print(f"[V1] 경계 커버리지: {cov:.0%}  " +
          "  ".join(f"{l}={c:.0%}" for l, c in per_phase.items()))
    return cov, per_phase


# ---------------------------------------------------------------------------
def v2_circularity(entries, boundaries, params):
    p = dict(params); p["drop_boundary_features"] = True
    spec = discover(entries, boundaries, **p)
    theta = p["theta"]
    labels = spec["meta"]["canonical_labels"]
    survived = {}
    for k, ph in spec["phases"].items():
        alive = [(n, v) for n, v in zip(ph["features"], ph["v"]) if v <= theta]
        survived[ph["label"]] = alive
        tag = ", ".join(f"{n}(v={v:.3f})" for n, v in alive) or "—"
        print(f"[V2] {ph['label']:<9} ablation 생존 조건: {tag}")
    # G4: 객체 채널이 정의하지 않는 경계(첫 phase 와 마지막 phase)는 독립
    # 증거가 있어야 한다. move(객체 정착 그 자체)는 정보로만 본다.
    need = [labels[0], labels[-1]]
    ok = all(survived[l] for l in need)
    print(f"[G4] {'PASS' if ok else 'FAIL'}: 필수 경계 {need} 생존 "
          f"{'확인' if ok else '실패'} (move 는 객체 채널 정의라 참고만)")
    return ok, survived


# ---------------------------------------------------------------------------
def v3_regression(spec, entries, boundaries):
    sg = Subgoal(spec)
    used = _used_entries(entries, spec)
    reg = tot = agree = 0
    for e in used:
        z_hat = sg.phase_of(e["F"])
        reg += int(np.sum(np.diff(z_hat) < 0))
        tot += len(z_hat)
        b = boundaries[e["demo_id"]]["bounds"]
        z_true = np.zeros(len(e["F"]), int)
        for i, t in enumerate(b):
            z_true[t:] = i + 1
        agree += int(np.sum(z_hat == z_true))
    rate, acc = reg / tot, agree / tot
    ok = rate < GATE_G5
    print(f"[V3] phase 역행률 {rate:.2%} (스텝 {tot}개), "
          f"세그먼터 z 일치율 {acc:.1%}")
    print(f"[G5] {'PASS' if ok else 'FAIL'}: 역행 {rate:.2%} "
          f"{'<' if ok else '>='} {GATE_G5:.0%}")
    return ok, rate, acc


# ---------------------------------------------------------------------------
def v4_leave_one_out(entries, boundaries, params, pad=2):
    ids = [e["demo_id"] for e in entries if e["demo_id"] in boundaries]
    covs = []
    for held in ids:
        rest = [e for e in entries if e["demo_id"] != held]
        try:
            spec = discover(rest, {d: b for d, b in boundaries.items()
                                   if d != held}, **params)
        except ValueError:
            continue
        e = next(x for x in entries if x["demo_id"] == held)
        if boundaries[held]["labels"] != spec["meta"]["canonical_labels"]:
            continue
        K = len(spec["meta"]["canonical_labels"])
        hit = 0
        for k in range(K):
            p = spec["phases"][str(k)]
            t = (boundaries[held]["bounds"][k] if k < K - 1
                 else len(e["F"]) - 1)
            ok = all(condition_holds(
                boundary_state(e["F"], e["names"].index(n), t, pad,
                               tail=k == K - 1),
                ty, th, mu)
                     for n, ty, th, mu in zip(p["features"], p["type"],
                                              p["thresh"], p["mean"]))
            hit += int(ok)
        covs.append(hit / K)
    cov = float(np.mean(covs)) if covs else float("nan")
    print(f"[V4] leave-one-out 경계 커버리지: {cov:.0%} "
          f"({len(covs)} folds; 데모 수가 적으면 참고용)")
    return cov


# ---------------------------------------------------------------------------
def run_all(cache_dir="./cache", boundaries_path="./artifacts/boundaries.json",
            subgoal_path="./artifacts/subgoal.json", enforce=True, **params):
    params = {**DEFAULTS, **params}
    entries = load_cache(cache_dir)
    with open(boundaries_path) as f:
        boundaries = json.load(f)
    with open(subgoal_path) as f:
        spec = json.load(f)

    v1_boundary_coverage(spec, entries, boundaries, pad=params["pad"])
    g4, _ = v2_circularity(entries, boundaries, params)
    g5, _, _ = v3_regression(spec, entries, boundaries)
    v4_leave_one_out(entries, boundaries, params, pad=params["pad"])

    if enforce and not (g4 and g5):
        raise SystemExit("[verify] GATE 실패 — Stage 6 진행 금지")
    print(f"\n[verify] G4 {'PASS' if g4 else 'FAIL'} / "
          f"G5 {'PASS' if g5 else 'FAIL'}")
    return g4, g5


# ===========================================================================
def run_selftest():
    import tempfile
    from extract_features import synth
    from segment_hier import segment_object_centric

    print("=== verify_subgoal SELFTEST ===")
    tmp = tempfile.mkdtemp(prefix="verify_subgoal_selftest_")
    synth(tmp, n_demos=8, seed=31)
    entries = load_cache(tmp)
    boundaries = {}
    for e in entries:
        seg = segment_object_centric(e["F"], e["names"], dt=e["dt"])
        boundaries[e["demo_id"]] = dict(T=len(e["F"]), bounds=seg["bounds"],
                                        labels=seg["labels"])
    spec = discover(entries, boundaries, **DEFAULTS)

    cov, _ = v1_boundary_coverage(spec, entries, boundaries)
    g4, _ = v2_circularity(entries, boundaries, DEFAULTS)
    g5, rate, acc = v3_regression(spec, entries, boundaries)
    v4 = v4_leave_one_out(entries, boundaries, DEFAULTS)

    # acc 기준 0.8: phase_of 는 "파지 후 정지" 구간에서 모션 기반 세그먼터보다
    # 앞서 전이한다(의도된 차이) — subgoal_discover.run_selftest 참조.
    ok = cov >= 0.9 and g4 and g5 and acc >= 0.8 and v4 >= 0.8
    print(f"\n[selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cache-dir", default="./cache")
    ap.add_argument("--boundaries", default="./artifacts/boundaries.json")
    ap.add_argument("--subgoal", default="./artifacts/subgoal.json")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(0 if run_selftest() else 1)
    run_all(args.cache_dir, args.boundaries, args.subgoal,
            enforce=not args.no_gate)


if __name__ == "__main__":
    main()
