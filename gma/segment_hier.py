#!/usr/bin/env python
"""
segment_hier.py — Stage 4: 계층적 phase 분할 (pure numpy)
================================================================================

flat segmenter(phase_segment.segment_features)의 문제(설계결정 D2): 이벤트 병합
창이 0.03·T 라서 데모 길이가 phase 개수를 바꾼다. 여기서는 태스크 구조를 계층으로
읽는다 — 무엇이 언제 일어났는지를 "객체"가 먼저 말하고, "상호작용"이 그 주변을
자른다:

    L1  객체 채널   : |d/dt object_goal_dist| 2-means 임계 -> 객체가 움직인
                      구간 [t_lift, t_settle]. 객체가 움직이는 동안이 transport
                      의 몸통이다. 데모 길이와 무관.
    L2  상호작용    : gripper_open 의 전환 중 t_lift 에 가장 가까운 것 = b1
                      (파지), t_settle 이후 가장 가까운 것 = b4 (릴리즈).
                      극성(열림이 큰 값인지)이 로봇마다 달라 순서·근접으로만
                      앵커링한다.
    L2  knee        : [t_lift, b4] 안에서만 object_goal_dist 의 high-water-mark
                      에 Kneedle. 들어 올리는 동안 goal 거리는 커지다가(위로
                      멀어짐) 수평 이동이 시작되면 꺾인다. 그 무릎이 grasp ->
                      transport 경계 b2. 상승 편위가 없으면(거리가 단조 감소)
                      "처음으로 시작값 아래로 내려간 시점"으로 폴백.

    bounds = [b1, b2, b3=t_settle, b4]
    labels = [approach, grasp, transport, place, retreat]

분할은 Ψ(boundary feature: eef_object_dist, object_goal_dist, gripper_open)만
읽는다. 관측+goal 만으로 계산되므로 M5 정책 rollout 에도 그대로 쓸 수 있다.

    python segment_hier.py --selftest      # numpy만; 합성 캐시로 검증
"""

import argparse

import numpy as np

import feature_select as fs
from phase_segment import (event_transitions, high_water_mark, plateau_knee,
                           trend_ratio, two_means_threshold, _minmax)

CANONICAL_LABELS = ["approach", "grasp", "transport", "place", "retreat"]


# ===========================================================================
# SECTION 1 — L1: 객체 채널
# ===========================================================================
def object_motion_window(d_og, dt, min_run_frac=0.02, smooth_window=11):
    """|d/dt object_goal_dist| 로 객체가 실제로 움직인 구간을 찾는다.

    -> dict(t_lift, t_settle, threshold, n_runs, runs)  또는 움직임이 없으면
       t_lift = t_settle = None.

    임계는 2-means(데이터 유도), run 은 min_run_frac·T 로 디바운스한다. 여러
    run 이 나와도(운반 중 잠깐 멈춤) 첫 run 의 시작 ~ 마지막 run 의 끝을
    창으로 삼는다 — "객체가 움직이기 시작한 순간"과 "완전히 정착한 순간"이
    필요한 것이지 연속성이 필요한 것이 아니다.
    """
    d_og = np.asarray(d_og, float)
    T = len(d_og)
    g = fs.smooth_positions(d_og.reshape(-1, 1), window=smooth_window).ravel()
    v = np.zeros(T)
    if T > 1:
        v[1:] = np.abs(np.diff(g)) / dt
        v[0] = v[1]

    thr, _ = two_means_threshold(v)
    mask = (v > thr).astype(int)

    # debounce: 짧은 켜짐/꺼짐은 이웃에 흡수
    min_run = max(3, int(min_run_frac * T))
    i = 0
    while i < T:
        j = i
        while j < T and mask[j] == mask[i]:
            j += 1
        if (j - i) < min_run and i > 0:
            mask[i:j] = mask[i - 1]
        i = j

    idx = np.where(mask)[0]
    if len(idx) == 0:
        return dict(t_lift=None, t_settle=None, threshold=float(thr),
                    n_runs=0, runs=[])
    # contiguous runs (진단용)
    runs, s = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b != a + 1:
            runs.append((int(s), int(a))); s = b
    runs.append((int(s), int(idx[-1])))
    return dict(t_lift=int(idx[0]), t_settle=int(idx[-1]),
                threshold=float(thr), n_runs=len(runs), runs=runs)


# ===========================================================================
# SECTION 2 — L2: 상호작용 앵커 + knee
# ===========================================================================
def anchor_gripper(grip, t_lift, t_settle):
    """gripper 전환 중 t_lift 에 가장 가까운 것 = b1, t_settle 뒤에서 가장
    가까운 것 = b4. 전환이 2개보다 많아도(노이즈 토글) 앵커가 골라낸다."""
    trans = [t for t, _ in event_transitions(np.asarray(grip, float))]
    note = f"{len(trans)} transitions; anchored on t_lift/t_settle"
    if not trans:
        return None, None, trans, "no gripper transitions"
    b1 = int(min(trans, key=lambda t: abs(t - t_lift)))
    after = [t for t in trans if t > max(t_settle, b1)]
    b4 = int(min(after, key=lambda t: abs(t - t_settle))) if after else None
    if b4 is None:
        note += "; no release transition after t_settle"
    return b1, b4, [int(t) for t in trans], note


def lift_knee(d_og, t_lift, t_end, excursion_frac=0.05):
    """grasp -> transport 경계 b2. [t_lift, t_end] 안에서만 본다.

    들어 올리는 동안 object_goal_dist 는 보통 커진다(수직으로 goal 에서
    멀어짐). high-water-mark 는 '커지다 -> plateau' 곡선이 되고 Kneedle 무릎이
    상승이 끝나는 시점이다. 편위가 전체 스케일의 excursion_frac 미만이면
    (거리가 사실상 단조 감소 = 들자마자 goal 로 접근) 무릎이 정의되지 않으므로
    '처음으로 시작값 아래로 내려간 시점'으로 폴백한다.
    """
    d_og = np.asarray(d_og, float)
    w = d_og[t_lift:t_end + 1]
    if len(w) < 3:
        return t_lift, "degenerate window"
    hwm = high_water_mark(w)
    scale = float(d_og.max() - d_og.min())
    excursion = float(hwm[-1] - w[0])
    if excursion >= excursion_frac * max(scale, 1e-9):
        return t_lift + plateau_knee(hwm), "knee"
    below = np.where(w < w[0])[0]
    if len(below):
        return t_lift + int(below[0]), "fallback: first drop below start"
    return t_lift, "fallback: no excursion, no drop"


# ===========================================================================
# SECTION 3 — 조립
# ===========================================================================
def _phase_subgoal(F, names, a, b, main, pad=2):
    """phase [a,b) 의 (feature, degrade_sign). sign = -sign(데모가 몬 방향).
    이벤트(gripper) phase 는 경계 자체가 전환이므로 pad 로 창을 넓혀 읽는다."""
    T = len(F)
    j = names.index(main)
    lo, hi = max(a - pad, 0), min(b + pad, T)
    seg = F[lo:hi, j]
    net = float(seg[-1] - seg[0])
    if net == 0.0:
        return (main, 1.0)
    return (main, -float(np.sign(net)))


def _tail_subgoal(F, names, a, b, min_trend=0.15):
    """이벤트가 닫지 않은 꼬리(retreat): 그 구간에서 데모가 가장 세게 몰던
    boundary feature 를 읽는다 (phase_segment._tail_subgoal 과 같은 논리)."""
    best, best_tr = None, min_trend
    for n in fs.BOUNDARY:
        seg = F[a:b, names.index(n)]
        tr = trend_ratio(seg)
        if tr > best_tr:
            net = float(seg[-1] - seg[0])
            if net != 0:
                best, best_tr = (n, -float(np.sign(net))), tr
    return best if best is not None else (fs.BOUNDARY[0], 1.0)


def segment_hierarchical(F, names=None, dt=0.05):
    """(T, k) feature 행렬 -> 계층 분할.

    Returns dict:
      z                  (T,) phase index per timestep
      bounds             [b1, b2, b3, b4] (성립한 것만; 아래 labels 와 쌍)
      labels             canonical 부분열 (전부 성립하면 5개)
      subgoal_per_phase  [(feature, degrade_sign), ...] phase 당 하나
      diag               level1/level2 진단 (boundaries_diag.json 에 그대로)
    """
    F = np.asarray(F, float)
    T = len(F)
    names = list(names) if names is not None else list(fs.NAMES)
    d_og = F[:, names.index("object_goal_dist")]
    grip = F[:, names.index("gripper_open")]

    # ---- L1 ------------------------------------------------------------
    l1 = object_motion_window(d_og, dt)
    diag = dict(level1={k: l1[k] for k in ("threshold", "n_runs",
                                           "t_lift", "t_settle")})
    if l1["t_lift"] is None:
        # 객체가 아예 움직이지 않은 데모: 분할 불가 (실패 데모)
        z = np.zeros(T, dtype=int)
        return dict(z=z, bounds=[], labels=["approach"],
                    subgoal_per_phase=[_tail_subgoal(F, names, 0, T)],
                    diag=dict(**diag, level2=dict(note="no object motion")))
    t_lift, t_settle = l1["t_lift"], l1["t_settle"]

    # ---- L2 ------------------------------------------------------------
    b1, b4, trans, note_b1 = anchor_gripper(grip, t_lift, t_settle)
    knee_end = b4 if b4 is not None else t_settle
    b2, knee_mode = lift_knee(d_og, t_lift, knee_end)
    b3 = int(t_settle)
    diag["level2"] = dict(transitions=trans, b1=b1, b4=b4,
                          note_b1=note_b1, knee=dict(b2=int(b2),
                                                     window=[int(t_lift),
                                                             int(knee_end)],
                                                     mode=knee_mode))

    # ---- 경계 조립: 순서가 깨진 경계는 버리고 라벨을 함께 줄인다 --------
    cand = [("grasp", b1), ("transport", b2), ("place", b3), ("retreat", b4)]
    bounds, labels = [], ["approach"]
    prev = 0
    for lab, b in cand:
        if b is None:
            diag["level2"][f"dropped_{lab}"] = "boundary unavailable"
            continue
        b = int(b)
        if b <= prev or b >= T:
            diag["level2"][f"dropped_{lab}"] = f"out of order (b={b}, prev={prev})"
            continue
        bounds.append(b); labels.append(lab); prev = b

    # ---- z, subgoal ------------------------------------------------------
    z = np.zeros(T, dtype=int)
    edges = [0] + bounds + [T]
    MAIN = dict(approach="eef_object_dist", grasp="gripper_open",
                transport="object_goal_dist", place="gripper_open")
    subgoal_per_phase = []
    for s, lab in enumerate(labels):
        a, b = edges[s], edges[s + 1]
        z[a:b] = s
        if lab == "retreat":
            subgoal_per_phase.append(_tail_subgoal(F, names, a, b))
        else:
            subgoal_per_phase.append(_phase_subgoal(F, names, a, b, MAIN[lab]))

    return dict(z=z, bounds=bounds, labels=labels,
                subgoal_per_phase=subgoal_per_phase, diag=diag)


# ===========================================================================
# SECTION 3b — 객체 중심 분할: pre | move | post  ★ 기본 모드
# ===========================================================================
# 계층 분할(위)의 약점이 실데이터에서 확인됐다: L1 의 2-means 속도 임계가
# "빠른 운반" 모드에 맞춰지면 느린 lift 가 t_lift 앞에 남고, grasp phase 가
# transport 를 침투한다. 근본 원인은 b2(knee)가 물리 이벤트 없는 곡률 경계라는
# 것. 그래서 기본 모드는 경계가 2개뿐이고 둘 다 물리 이벤트인 객체 중심 분할:
#
#     pre  = [0, t_start)        객체가 아직 안 움직임 (접근 + 파지)
#     move = [t_start, t_settle) 객체가 움직이는 중   (들기 + 운반 + 놓기)
#     post = [t_settle, T)       객체가 goal 에 정착   (릴리즈 + 후퇴)
#
# "파지됨"은 phase 가 아니라 pre→move 경계의 subgoal 조건이 된다 (contact=1,
# eef_object_dist≈0 등은 Stage 5 가 경계 시점 통계로 자동 추론).
#
# 검출은 속도가 아니라 변위: |d_og - 정지값| 이 노이즈 바닥에서 유도한 ε 를
# 넘는 첫 시점(지속 조건) = t_start, |d_og - 최종값| 이 ε 아래로 마지막으로
# 내려온 시점 = t_settle. 변위는 누적이라 느린 lift 도 즉시 잡힌다.
# object_goal_dist(Ψ 열)만 읽으므로 M5 rollout 에도 그대로 쓸 수 있다.

OBJECT_LABELS = ["pre", "move", "post"]


def _first_sustained(mask, min_run):
    """mask 가 min_run 프레임 연속 True 가 되는 첫 시작 인덱스 (없으면 None)."""
    run = 0
    for i, m in enumerate(mask):
        run = run + 1 if m else 0
        if run >= min_run:
            return i - min_run + 1
    return None


def _backtrack_onset(dev, t_det, eps_lo):
    """탐지 시점 t_det 에서 뒤로 걸어가 |변위| 가 eps_lo(노이즈 바닥) 아래로
    내려가는 지점 직후 = 실제 움직임 시작점. 이중 임계 hysteresis 의 아래턱."""
    t = t_det
    while t > 0 and dev[t - 1] > eps_lo:
        t -= 1
    return t


def segment_object_centric(F, names=None, dt=0.05, *, eps_min=0.005,
                           eps_lo_min=0.001, k_frac=0.05, min_run_frac=0.01,
                           noise_mult=4.0, noise_mult_lo=1.5):
    """(T, k) feature 행렬 -> pre | move | post.

    이중 임계 hysteresis:
      eps_hi = max(eps_min, noise_mult · σ)      확실한 움직임 탐지 (min_run 지속)
      eps_lo = max(eps_lo_min, noise_mult_lo · σ) 탐지점에서 온셋으로 역추적
    σ 는 처음/끝 k_frac·T 프레임(객체가 확실히 정지한 구간)의 MAD 기반.
    높은 턱이 노이즈 오검출을 막고, 낮은 턱으로의 역추적이 경계를 실제
    움직임 시작(정착 끝)에 붙인다 — 느린 lift 도 지연 없이 잡힌다.
    Returns: segment_hierarchical 과 같은 dict 계약.
    """
    F = np.asarray(F, float)
    T = len(F)
    names = list(names) if names is not None else list(fs.NAMES)
    d_og = F[:, names.index("object_goal_dist")]

    k = max(5, int(k_frac * T))
    rest = float(np.median(d_og[:k]))
    final = float(np.median(d_og[-k:]))
    mad = max(np.median(np.abs(d_og[:k] - rest)),
              np.median(np.abs(d_og[-k:] - final)))
    sigma = 1.4826 * float(mad)
    eps_hi = max(eps_min, noise_mult * sigma)
    eps_lo = max(eps_lo_min, noise_mult_lo * sigma)
    min_run = max(3, int(min_run_frac * T))

    dev0 = np.abs(d_og - rest)
    t_det = _first_sustained(dev0 > eps_hi, min_run)
    t_start = _backtrack_onset(dev0, t_det, eps_lo) if t_det is not None else None

    devF = np.abs(d_og - final)[::-1]              # 뒤에서부터 본 최종값 이탈
    r_det = _first_sustained(devF > eps_hi, min_run)
    t_settle = None
    if r_det is not None:
        r_on = _backtrack_onset(devF, r_det, eps_lo)
        t_settle = T - r_on            # 정착이 시작된 프레임 = post 시작

    diag = dict(mode="object_centric", eps_hi=round(eps_hi, 5),
                eps_lo=round(eps_lo, 5), noise_mad=round(float(mad), 6),
                rest=round(rest, 4), final=round(final, 4),
                t_detect=t_det, t_start=t_start, t_settle=t_settle,
                min_run=min_run)

    if t_start is None or t_settle is None or not (0 < t_start < t_settle < T):
        # 객체가 안 움직였거나(실패 데모) 끝까지 정착하지 않은 데모
        z = np.zeros(T, dtype=int)
        diag["note"] = "no valid object motion window"
        return dict(z=z, bounds=[], labels=["pre"],
                    subgoal_per_phase=[_tail_subgoal(F, names, 0, T)],
                    diag=dict(level1=diag, level2={}))

    bounds = [int(t_start), int(t_settle)]
    z = np.zeros(T, dtype=int)
    z[t_start:t_settle] = 1
    z[t_settle:] = 2
    subgoal_per_phase = [
        _phase_subgoal(F, names, 0, t_start, "eef_object_dist"),
        _phase_subgoal(F, names, t_start, t_settle, "object_goal_dist"),
        _tail_subgoal(F, names, t_settle, T),
    ]
    return dict(z=z, bounds=bounds, labels=list(OBJECT_LABELS),
                subgoal_per_phase=subgoal_per_phase,
                diag=dict(level1=diag, level2={}))


# ===========================================================================
# SECTION 4 — selftest (numpy만; 합성 캐시로 end-to-end)
# ===========================================================================
def run_selftest():
    import tempfile
    from extract_features import synth, load_cache

    print("=== segment_hier SELFTEST ===")
    ok = True
    tmp = tempfile.mkdtemp(prefix="segment_hier_selftest_")
    synth(tmp, n_demos=8, seed=11)
    entries = load_cache(tmp)

    seqs = []
    for e in entries:
        seg = segment_hierarchical(e["F"], e["names"], dt=e["dt"])
        seqs.append(tuple(seg["labels"]))
        tb = e["meta"]["true_bounds"]          # [b1, b2, b3, b4] 합성 진리
        T = len(e["F"])

        if seg["labels"] != CANONICAL_LABELS:
            ok = False
            print(f"[FAIL] {e['demo_id']}: labels {seg['labels']}")
            continue
        if len(seg["bounds"]) != 4 or list(seg["bounds"]) != sorted(seg["bounds"]):
            ok = False
            print(f"[FAIL] {e['demo_id']}: bounds {seg['bounds']}")
            continue
        b1, b2, b3, b4 = seg["bounds"]
        # 합성 데모의 구조: grip 닫힘 = tb0+(tb1-tb0)/2, 객체 이동 시작 = tb1,
        # 정착 = tb2, grip 열림 = tb2+(tb3-tb2)/2. 앵커가 그 근방(±8%T)인가.
        tol = max(4, int(0.08 * T))
        checks = [
            ("b1~grip close", b1, tb[0] + (tb[1] - tb[0]) // 2),
            ("b2 in [t_lift, settle]", b2, None),
            ("b3~settle", b3, tb[2]),
            ("b4~grip open", b4, tb[2] + (tb[3] - tb[2]) // 2),
        ]
        for nm, got, want in checks:
            if want is not None and abs(got - want) > tol:
                ok = False
                print(f"[FAIL] {e['demo_id']}: {nm}: {got} vs {want} (tol {tol})")
        if not (tb[1] - tol <= b2 <= tb[2] + tol):
            ok = False
            print(f"[FAIL] {e['demo_id']}: b2={b2} outside lift window "
                  f"[{tb[1]},{tb[2]}]")
        # z 는 경계와 일치해야 한다
        zz = seg["z"]
        if not (zz[0] == 0 and zz[-1] == 4 and
                all(zz[b] == i + 1 for i, b in enumerate(seg["bounds"]))):
            ok = False; print(f"[FAIL] {e['demo_id']}: z/bounds mismatch")

    if len(set(seqs)) == 1:
        print(f"8/8 demos -> canonical sequence {' | '.join(CANONICAL_LABELS)}")
    else:
        ok = False; print(f"[FAIL] label sequences diverge: {set(seqs)}")

    # subgoal sanity: approach 는 (eef_object_dist, +1) — 거리를 키우면 열화
    e = entries[0]
    seg = segment_hierarchical(e["F"], e["names"], dt=e["dt"])
    sg = dict(zip(seg["labels"], seg["subgoal_per_phase"]))
    if sg["approach"] != ("eef_object_dist", 1.0):
        ok = False; print(f"[FAIL] approach subgoal {sg['approach']}")
    if sg["transport"][0] != "object_goal_dist":
        ok = False; print(f"[FAIL] transport subgoal {sg['transport']}")
    print("subgoals:", {k: (v[0], v[1]) for k, v in sg.items()})

    # 길이 강건성 (D2): 같은 시드 합성 데모를 2배로 늘려도 라벨 시퀀스 동일
    ee = entries[0]
    F2 = np.repeat(ee["F"], 2, axis=0)
    seg2 = segment_hierarchical(F2, ee["names"], dt=ee["dt"])
    if seg2["labels"] != CANONICAL_LABELS:
        ok = False
        print(f"[FAIL] 2x-length demo changed the sequence: {seg2['labels']}")
    else:
        print("2x-length demo -> same 5-phase sequence (D2: 길이 불변)")

    # ---- object-centric 모드 (기본): pre | move | post -------------------
    print("\n--- object-centric (pre | move | post) ---")
    for e in entries:
        seg = segment_object_centric(e["F"], e["names"], dt=e["dt"])
        tb = e["meta"]["true_bounds"]
        T = len(e["F"])
        tol = max(4, int(0.05 * T))
        if seg["labels"] != OBJECT_LABELS:
            ok = False
            print(f"[FAIL] {e['demo_id']}: labels {seg['labels']}")
            continue
        t_start, t_settle = seg["bounds"]
        # 합성 진리: 객체는 정확히 [tb1, tb2] 에서만 움직인다
        if abs(t_start - tb[1]) > tol:
            ok = False
            print(f"[FAIL] {e['demo_id']}: t_start {t_start} vs {tb[1]} (tol {tol})")
        if abs(t_settle - tb[2]) > tol:
            ok = False
            print(f"[FAIL] {e['demo_id']}: t_settle {t_settle} vs {tb[2]} (tol {tol})")
    print(f"{len(entries)}/{len(entries)} demos -> pre | move | post, "
          f"경계가 합성 진리(객체 이동 구간) ±5%T 이내")

    seg3 = segment_object_centric(np.repeat(ee["F"], 2, axis=0), ee["names"],
                                  dt=ee["dt"])
    if seg3["labels"] != OBJECT_LABELS:
        ok = False
        print(f"[FAIL] 2x-length demo (object mode): {seg3['labels']}")
    else:
        print("2x-length demo -> same 3-phase sequence (길이 불변)")

    sgo = dict(zip(seg3["labels"],
                   segment_object_centric(ee["F"], ee["names"],
                                          dt=ee["dt"])["subgoal_per_phase"]))
    if sgo["pre"][0] != "eef_object_dist" or sgo["move"][0] != "object_goal_dist":
        ok = False; print(f"[FAIL] object-mode subgoals: {sgo}")
    print("subgoals:", sgo)

    print(f"\n[selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(0 if run_selftest() else 1)
    print("this module is a library; run run_boundaries.py for the pipeline "
          "step, or --selftest to verify")


if __name__ == "__main__":
    main()
