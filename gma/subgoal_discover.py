#!/usr/bin/env python
"""
subgoal_discover.py — Stage 5: 관측만으로 phase subgoal 자동 추론
================================================================================

하드코딩 없음. 각 phase 경계 시점의 cross-demo 통계가 조건을 결정한다:

    v_j = Var_i[φ_j(t_k)] / Var_{i,t}[φ_j]     분산비 → 조건 feature 선택
          (모든 데모가 그 경계에서 같은 값을 만들었으면 필연 → v 작음,
           우연이면 데모마다 제각각 → v 큼. 단일 데모로는 구분 불가 — D3)
    ε   = c·σ                                   허용 오차 (데모 산포 = 관용 — D4)
    persistence                                 satisfies 용 조건 분리 (D5)

조건 형태(le/ge/box/eq)도 데이터가 정한다:
    이진 event      → eq   (thresh = 경계값)
    경계로 하강 진입 → le   (thresh = mean + c·σ̃)
    경계로 상승 진입 → ge   (thresh = mean − c·σ̃)
    변화 없이 유지   → box  (|φ − mean| ≤ c·σ̃)

출력: artifacts/subgoal.json ★ 단일 진리원 (D10, PIPELINE_v4.md Sec 3.4).
실행기는 rho.py (ρ_k / satisfies / phase_of).

    python subgoal_discover.py                 # cache + boundaries -> subgoal.json
    python subgoal_discover.py --selftest      # numpy만; 합성 end-to-end
"""

import argparse
import json
import os
from collections import Counter

import numpy as np

import feature_select as fs
from extract_features import load_cache
from rho import condition_holds

DEFAULTS = dict(theta=0.15, c=2.0, alpha=1.0, max_feat=4, pad=2,
                sd_floor=0.001, persist_min=0.80)


def boundary_state(F, j, t, pad, tail=False):
    """경계 t 의 subgoal 상태값: [t, t+2·pad] 전방 중앙값.

    경계 프레임은 다음 phase 의 첫 프레임이고, phase 를 닫는 이벤트(파지
    완료 등)는 경계 직후 수 프레임 안에 완성된다 — 중심 윈도우로 읽으면
    객체가 contact 플래그보다 먼저 움직인 데모(그리퍼가 밀어서)에서 조건이
    깨진다 (실측: demo_001 은 t_start 가 contact 보다 2프레임 빠름).
    baseline phase_subgoal_set 의 pad 논리와 같은 이유. 마지막 phase(T−1)는
    앞이 없으므로 후방 [t−2·pad, t] 를 쓴다."""
    if tail:
        lo, hi = max(0, t - 2 * pad), t + 1
    else:
        lo, hi = t, min(len(F), t + 2 * pad + 1)
    return float(np.median(F[lo:hi, j]))


# 하위 호환 (verify_subgoal 이전 버전이 import 하던 이름)
_win = boundary_state


def _is_binary(col):
    return bool(np.all(np.isin(np.round(np.asarray(col, float), 6),
                               [0.0, 1.0])))


def select_features(v_by_name, theta, max_feat):
    """v ≤ theta 인 feature 를 v 오름차순으로 최대 max_feat 개.
    하나도 없으면 argmin 하나를 노트와 함께 반환한다 (게이트가 판단할 몫)."""
    ranked = sorted(v_by_name.items(), key=lambda kv: kv[1])
    picked = [(n, v) for n, v in ranked if v <= theta][:max_feat]
    if picked:
        return picked, None
    n, v = ranked[0]
    return [(n, v)], f"no feature under theta={theta}; kept argmin {n} (v={v:.3f})"


def discover(entries, boundaries, *, theta=0.15, c=2.0, alpha=1.0, max_feat=4,
             pad=2, sd_floor=0.001, persist_min=0.80,
             drop_boundary_features=False):
    """cache entries + boundaries dict -> subgoal spec (Sec 3.4)."""
    # ---- canonical 시퀀스와 데모 선별 ------------------------------------
    tally = Counter(" | ".join(boundaries[d]["labels"]) for d in boundaries)
    canonical = tally.most_common(1)[0][0].split(" | ")
    K = len(canonical)
    used, excluded = [], {}
    for e in entries:
        did = e["demo_id"]
        if did not in boundaries:
            excluded[did] = ["no boundaries"]
        elif boundaries[did]["labels"] != canonical:
            excluded[did] = boundaries[did]["labels"]
        else:
            used.append(e)
    if len(used) < 2:
        raise ValueError("cross-demo 분산에는 canonical 시퀀스 데모가 2편 "
                         f"이상 필요 (지금 {len(used)}편)")
    names = used[0]["names"]

    # ---- 후보 feature: subgoal-eligible − (ablation) − 상수 --------------
    eligible = [n for n in names if fs.SPEC[n].kind != fs.QUALITY]
    if drop_boundary_features:
        eligible = [n for n in eligible if not fs.SPEC[n].boundary]
    allF = np.concatenate([e["F"] for e in used], axis=0)
    gstd = {n: float(allF[:, names.index(n)].std()) for n in names}
    constant_dropped = [n for n in eligible if gstd[n] < sd_floor]
    eligible = [n for n in eligible if n not in constant_dropped]
    binary = {n: _is_binary(allF[:, names.index(n)]) for n in eligible}

    # ---- phase 별 통계 ----------------------------------------------------
    phases = {}
    for k in range(K):
        t_end, t_start = [], []                     # 데모별 (경계, phase 시작)
        for e in used:
            b = boundaries[e["demo_id"]]["bounds"]
            T = len(e["F"])
            t_end.append(int(b[k]) if k < K - 1 else T - 1)
            t_start.append(int(b[k - 1]) if k > 0 else 0)

        X = {}                                       # 경계값 (n_used,)
        S = {}                                       # phase 시작값
        tail = (k == K - 1)
        for n in eligible:
            j = names.index(n)
            X[n] = np.array([boundary_state(e["F"], j, te, pad, tail=tail)
                             for e, te in zip(used, t_end)])
            S[n] = np.array([boundary_state(e["F"], j, ts, pad)
                             for e, ts in zip(used, t_start)])

        v_by_name = {n: float(X[n].var() / (gstd[n] ** 2)) for n in eligible}
        picked, note = select_features(v_by_name, theta, max_feat)

        feats, mean, std, vv, typ, thr, g_std = [], [], [], [], [], [], []
        for n, v in picked:
            mu = float(X[n].mean()); sd = float(X[n].std())
            sd_t = max(sd, sd_floor)
            if binary[n]:
                t_, th = "eq", float(np.round(mu))
            else:
                delta = float((X[n] - S[n]).mean())
                if delta < -c * sd_t:
                    t_, th = "le", mu + c * sd_t
                elif delta > c * sd_t:
                    t_, th = "ge", mu - c * sd_t
                else:
                    t_, th = "box", c * sd_t
            feats.append(n); mean.append(mu); std.append(sd); vv.append(v)
            typ.append(t_); thr.append(th); g_std.append(gstd[n])

        # persistence: 경계 이후 [t_k, T) 에서 조건이 유지되는 비율
        pers = []
        for n, t_, th, mu in zip(feats, typ, thr, mean):
            j = names.index(n)
            fr = []
            for e, te in zip(used, t_end):
                seg = e["F"][te:, j]
                fr.append(float(np.mean(condition_holds(seg, t_, th, mu)))
                          if len(seg) else 1.0)
            pers.append(float(np.mean(fr)))
        persistent = [p >= persist_min for p in pers]

        # d_start: phase 시작 시점의 표준화 거리 (rho 의 스케일)
        scale = np.array([0.5 if t_ == "eq"
                          else max(sd, 0.25 * gs, sd_floor)
                          for t_, sd, gs in zip(typ, std, g_std)])
        d_i = []
        for i in range(len(used)):
            z = np.array([(S[n][i] - mu) for n, mu in zip(feats, mean)]) / scale
            d_i.append(float(np.sqrt(np.mean(z * z))))
        d_start = max(float(np.mean(d_i)), 1e-6)

        phases[str(k)] = dict(
            label=canonical[k], features=feats,
            mean=[round(x, 6) for x in mean],
            std=[round(x, 6) for x in std],
            v=[round(x, 6) for x in vv],
            type=typ, thresh=[round(x, 6) for x in thr],
            persistence=[round(x, 4) for x in pers],
            persistent=persistent,
            global_std=[round(x, 6) for x in g_std],
            boundary_t=t_end, d_start=round(d_start, 4), sd_floor=sd_floor,
            **({"note": note} if note else {}))

    meta = dict(
        n_demos_used=len(used), demos_used=[e["demo_id"] for e in used],
        demos_excluded=excluded,
        sequence_tally=dict(tally), canonical_labels=canonical,
        theta=theta, c=c, alpha=alpha, max_feat=max_feat, pad=pad,
        sd_floor=sd_floor, persist_min=persist_min,
        boundary_features_dropped=drop_boundary_features,
        constant_features_dropped=constant_dropped)
    return dict(meta=meta, phases=phases)


def run(cache_dir="./cache", boundaries_path="./artifacts/boundaries.json",
        out_path="./artifacts/subgoal.json", **params):
    entries = load_cache(cache_dir)
    with open(boundaries_path) as f:
        boundaries = json.load(f)
    spec = discover(entries, boundaries, **params)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(spec, f, indent=2)

    print(f"[discover] {spec['meta']['n_demos_used']} demos, "
          f"canonical = {' | '.join(spec['meta']['canonical_labels'])}")
    if spec["meta"]["demos_excluded"]:
        print(f"[discover] excluded: {spec['meta']['demos_excluded']}")
    for k, p in spec["phases"].items():
        conds = ", ".join(
            f"{n} {t} {th:.4g}{'*' if pr else ''}"
            for n, t, th, pr in zip(p["features"], p["type"], p["thresh"],
                                    p["persistent"]))
        print(f"  phase {k} [{p['label']:<9}] d_start={p['d_start']:<8} {conds}")
    print("  (* = persistent -> phase_of/satisfies 에 사용)")
    print(f"[out] {out_path}")
    return spec


# ===========================================================================
# selftest — 합성 캐시 end-to-end (numpy만)
# ===========================================================================
def run_selftest():
    import tempfile
    from extract_features import synth
    from segment_hier import segment_object_centric
    from rho import Subgoal

    print("=== subgoal_discover SELFTEST ===")
    ok = True
    tmp = tempfile.mkdtemp(prefix="subgoal_selftest_")
    synth(tmp, n_demos=8, seed=23)
    entries = load_cache(tmp)
    boundaries = {}
    for e in entries:
        seg = segment_object_centric(e["F"], e["names"], dt=e["dt"])
        boundaries[e["demo_id"]] = dict(T=len(e["F"]), bounds=seg["bounds"],
                                        labels=seg["labels"])

    spec = discover(entries, boundaries, **DEFAULTS)
    ph = {p["label"]: p for p in spec["phases"].values()}

    # pre 의 subgoal 은 "파지됨": contact eq 1 + eef_object_dist le ε 가 나와야
    pre = ph["pre"]
    cond = dict(zip(pre["features"], pre["type"]))
    if cond.get("contact") != "eq":
        ok = False; print(f"[FAIL] pre lacks contact eq: {cond}")
    if cond.get("eef_object_dist") != "le":
        ok = False; print(f"[FAIL] pre lacks eef_object_dist le: {cond}")
    j = pre["features"].index("contact")
    if abs(pre["thresh"][j] - 1.0) > 1e-6:
        ok = False; print(f"[FAIL] contact thresh {pre['thresh'][j]} != 1")
    print(f"pre  : {cond}  (파지 조건이 관측만으로 복원됨)")

    # move 의 subgoal 은 "객체가 goal": object_goal_dist le + persistent
    mv = ph["move"]
    if "object_goal_dist" not in mv["features"]:
        ok = False; print(f"[FAIL] move lacks object_goal_dist: {mv['features']}")
    else:
        j = mv["features"].index("object_goal_dist")
        if mv["type"][j] != "le" or not mv["persistent"][j]:
            ok = False; print(f"[FAIL] move object_goal_dist type/persistent: "
                              f"{mv['type'][j]}, {mv['persistent'][j]}")
    print(f"move : {dict(zip(mv['features'], mv['type']))}")

    # 실행기 왕복: phase_of 가 세그먼트 z 와 대체로 일치 + 역행 없음
    sg = Subgoal(spec)
    agree, reg, tot = 0, 0, 0
    for e in entries:
        z_true = segment_object_centric(e["F"], e["names"], dt=e["dt"])["z"]
        z_hat = sg.phase_of(e["F"])
        agree += int(np.sum(z_hat == z_true)); tot += len(z_true)
        reg += int(np.sum(np.diff(z_hat) < 0))
    agree /= tot; reg /= tot
    # 주의: phase_of(조건 기반)는 "파지 후 아직 안 움직인" 구간에서 세그먼터
    # (모션 기반)보다 한 phase 앞서 전이한다 — subgoal 이 달성되는 순간
    # potential 이 올라야 하므로 의도된 차이. 그래서 완전 일치가 아니라
    # 80% + 역행 0 를 요구한다.
    if agree < 0.8:
        ok = False; print(f"[FAIL] phase_of agreement {agree:.1%} < 80%")
    if reg > 0.05:
        ok = False; print(f"[FAIL] regression {reg:.2%} > 5%")
    print(f"phase_of vs segmenter z: {agree:.1%} 일치, 역행 {reg:.2%}")

    # rho 단조성: move 구간에서 ρ_1 이 전반적으로 상승
    e = entries[0]
    a, b = boundaries[e["demo_id"]]["bounds"]
    r = sg.rho(1, e["F"][a:b])
    up = float(np.mean(np.diff(r) >= -1e-9))
    if not (r[-1] > r[0] and r[-1] > 0.8):
        ok = False; print(f"[FAIL] rho_1 {r[0]:.2f} -> {r[-1]:.2f}")
    print(f"rho_1 over move: {r[0]:.3f} -> {r[-1]:.3f} "
          f"(비감소 스텝 {up:.0%})")

    # json 직렬화 왕복
    s = json.dumps(spec)
    sg2 = Subgoal(json.loads(s))
    if sg2.phase_of(e["F"][0]) != sg.phase_of(e["F"][0]):
        ok = False; print("[FAIL] json round-trip changed behavior")

    print(f"\n[selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cache-dir", default="./cache")
    ap.add_argument("--boundaries", default="./artifacts/boundaries.json")
    ap.add_argument("--out", default="./artifacts/subgoal.json")
    ap.add_argument("--theta", type=float, default=DEFAULTS["theta"])
    ap.add_argument("--c", type=float, default=DEFAULTS["c"])
    ap.add_argument("--alpha", type=float, default=DEFAULTS["alpha"])
    ap.add_argument("--max-feat", type=int, default=DEFAULTS["max_feat"])
    ap.add_argument("--pad", type=int, default=DEFAULTS["pad"])
    ap.add_argument("--sd-floor", type=float, default=DEFAULTS["sd_floor"])
    ap.add_argument("--persist-min", type=float, default=DEFAULTS["persist_min"])
    ap.add_argument("--drop-boundary-features", action="store_true",
                    help="V2 순환성 ablation 용")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(0 if run_selftest() else 1)
    run(args.cache_dir, args.boundaries, args.out,
        theta=args.theta, c=args.c, alpha=args.alpha, max_feat=args.max_feat,
        pad=args.pad, sd_floor=args.sd_floor, persist_min=args.persist_min,
        drop_boundary_features=args.drop_boundary_features)


if __name__ == "__main__":
    main()
