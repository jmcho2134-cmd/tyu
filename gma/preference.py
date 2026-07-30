#!/usr/bin/env python
"""
preference.py — Stage 9: simulator 실측 + 선호 랭킹 (PIPELINE_v4 §3.8, G8)
================================================================================

입력은 degradation.npz 하나다. Stage 8 이 rung 마다 실행 궤적 전체(eef/obj/
grip/contact/actions/F/k_index/success)를 저장했으므로 여기서는 재시뮬레이션
없이 실측 라벨을 계산한다 (문서의 "rollout_exec 실측"은 Stage 8 실행 시점에
이미 일어난 것).

실측 메트릭 (rung 당):
    success         stepper 판정 (저장값)
    final_goal_err  ‖obj[-1] − goal‖
    path_length     Σ‖Δeef‖
    jerk            mean F[:, eef_jerk]
    slip            mean F[:, object_slip]
    rho_end         주입 phase 의 실행 끝점 ρ_raw (Stage 8 저장값)

랭킹 스칼라 (λ=0 rung 대비):
    damage = 2·(1−success) + Δfinal_goal_err/0.10 + max(0, ρ₀−ρ)

    성분이 세 개인 이유: success 하나면 성공 rung 끼리 등수가 없고(gradedness
    소멸), ρ 하나면 own-phase 는 멀쩡한데 downstream 에서 태스크가 죽는
    family(pre 회전형)가 안 보인다. 가중치는 "실패 = goal 에서 20cm 이탈과
    등가, goal 오차는 bin 스케일 0.10 m 로 정규화" — λ 는 어디에도 없다 (D8).
    jerk/slip/path 는 라벨에 안 섞고 실측값 그대로 hdf5 에 실어 Stage 10 의
    입력/분석에 넘긴다.

G8 (family 단위 — 실패 = candidate reject, 그 family 의 pair 만 버린다):
    R1 λ=0 rung 이 성공하고 데모를 재현한다 (max eef err ≤ max_repro)
    R2 success 가 level 에 대해 downward-closed (한번 죽으면 계속 죽어야)
    R3 spearman(level, damage) ≥ min_spearman  (단조성)
    R4 max(damage)−min(damage) ≥ min_spread    (사다리가 실제로 움직임)
전체 G8 PASS = 수락률 ≥ accept_frac 이고 모든 phase 에 생존 family ≥ 1 이고
pair 총수 ≥ min_pairs.

pair 는 nested 만 (야말/제안서 Sec 8.2): 같은 family 안에서 level_i < level_j
이고 damage 차이 ≥ min_gap 인 (i, j) 만. cross-family 비교는 라벨 없이
불가능하고, 차이가 min_gap 미만인 쌍에 순서를 주장하는 것은 라벨 노이즈다.

출력: artifacts/preference.hdf5 (§3.8) + preference_diag.json + G8 시각화.

    python preference.py --selftest      # mock family; 파일 불필요
    python preference.py --plot          # 실데이터
"""

import argparse
import json
import os

import numpy as np

import feature_select as fs
from degradation import CLIFF_LEVEL, LEVELS, load_families


# ===========================================================================
# SECTION 1 — rung 실측
# ===========================================================================
def measured_metrics(traj, goal, rho_end):
    eef = np.asarray(traj["eef"], float)
    obj = np.asarray(traj["obj"], float)
    F = np.asarray(traj["F"], float)
    ji, si = fs.index_of("eef_jerk"), fs.index_of("object_slip")
    return dict(
        success=bool(traj["success"]),
        final_goal_err=float(np.linalg.norm(obj[-1] - np.asarray(goal))),
        path_length=float(np.linalg.norm(np.diff(eef, axis=0), axis=1).sum()),
        jerk=float(F[:, ji].mean()),
        slip=float(F[:, si].mean()),
        rho_end=float(rho_end),
        length=int(traj["length"]),
        lam=float(traj["lam"]))


def damage_of(m, m0, goal_scale=0.10):
    """λ=0 rung (m0) 대비 실측 열화 스칼라. 클수록 나쁘다."""
    d_goal = (m["final_goal_err"] - m0["final_goal_err"]) / goal_scale
    d_rho = max(0.0, m0["rho_end"] - m["rho_end"])
    return 2.0 * (1.0 - float(m["success"])) + d_goal + d_rho


def spearman(x, y):
    """동순위 평균랭크 spearman (scipy 없이)."""
    def rank(v):
        v = np.asarray(v, float)
        order = np.argsort(v)
        r = np.empty(len(v))
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            r[order[i:j + 1]] = 0.5 * (i + j) + 1
            i = j + 1
        return r
    rx_, ry = rank(x), rank(y)
    sx, sy = rx_.std(), ry.std()
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    return float(((rx_ - rx_.mean()) * (ry - ry.mean())).mean() / (sx * sy))


# ===========================================================================
# SECTION 2 — G8: family 게이트 + nested pairs
# ===========================================================================
def rung_kinds(fam):
    """family 의 rung 종류. 구 npz(표시 없음)는 전부 'eff' 로 본다."""
    n = len(fam["trajectories"])
    return list(fam.get("rung_kind", ["eff"] * n))


def gradedness(dmg):
    """사다리 계단성. -> (min_step_frac, 정규화 rung 벡터)

    문서 §5 의 G8 은 "monotonicity + gradedness + success preservation" 인데
    gradedness 검사가 구현에 없었다. 실측 결과 수락된 19 family 의 평균 정규화
    rung 이 [0, .06, .17, .48, .98] 로, 아래 두 칸이 사실상 붙어 있었다
    (5단 사다리가 실질 2단). 인접 rung 증분의 최솟값을 전체 폭으로 나눠 본다 —
    균등 사다리면 0.25, 위 실측이면 0.06 이다.
    """
    d = np.asarray(dmg, float)
    spread = float(d.max() - d.min())
    if spread <= 0:
        return 0.0, np.zeros_like(d)
    steps = np.diff(d)
    return float(steps.min() / spread), (d - d.min()) / spread


def gate_family(fam, metrics, demo_eef, args):
    """(accepted, reason). metrics 는 rung 순서의 measured_metrics 목록."""
    m0 = metrics[0]
    t0 = fam["trajectories"][0]
    kinds = rung_kinds(fam)
    if not m0["success"]:
        return False, "λ=0 rung 이 실패"
    if demo_eef is not None:
        e = np.asarray(t0["eef"], float)
        n = min(len(e), len(demo_eef))
        err = float(np.abs(e[:n] - demo_eef[:n]).max())
        if err > args.max_repro:
            return False, f"λ=0 재현 오차 {err*100:.1f}cm > {args.max_repro*100:.0f}cm"
    succ = [m["success"] for m in metrics]
    for a, b in zip(succ, succ[1:]):
        if b and not a:
            return False, f"success 역전 {''.join('o' if s else 'x' for s in succ)}"
    dmg = [damage_of(m, m0, args.goal_scale) for m in metrics]
    # 순위상관은 rung 인덱스로 본다 (level 값이 family 마다 달라도 무관하고,
    # cliff rung 이 붙어 rung 수가 변해도 성립한다).
    rho = spearman(list(range(len(dmg))), dmg)
    if not np.isfinite(rho) or rho < args.min_spearman:
        return False, f"단조성 위반 spearman={rho:+.2f} < {args.min_spearman}"
    spread = float(np.max(dmg) - np.min(dmg))
    if spread < args.min_spread:
        return False, f"사다리 무반응 spread={spread:.3f} < {args.min_spread}"

    # R5 gradedness — 효율 rung 에서만 본다. cliff rung 은 정의상 도약이므로
    # 여기 섞으면 계단성 검사가 무의미해진다.
    eff = [d for d, k in zip(dmg, kinds) if k == "eff"]
    if len(eff) >= 3:
        step, norm = gradedness(eff)
        if step < args.min_step_frac:
            return False, (f"계단성 위반 min_step={step:+.3f} < "
                           f"{args.min_step_frac} (정규화 rung "
                           f"{np.round(norm, 2).tolist()})")
    else:
        step = float("nan")
    return True, (f"ok (spearman={rho:+.2f}, spread={spread:.2f}, "
                  f"min_step={step:.2f})")


PAIR_EFF, PAIR_CLIFF = 0, 1          # pairs/kind 값


def make_pairs(row_ids, metrics, args, kinds=None):
    """nested pairs: (better_row, worse_row, margin, kind).

    kind 를 나누는 이유: 두 rung 모두 성공한 pair 는 '효율' 선호를 가르치고,
    한쪽이 실패한 pair 는 '성공 여부'를 가르친다. 섞어서 하나의 ranking loss 에
    넣으면 실패 항(damage 가중치 2.0)이 마진을 지배해 R_θ 가 성공 판별기로
    수렴한다 (실측: pair 의 54% 가 성공/실패 대비, margin 중앙값 3.41).
    Stage 10 이 두 집합에 다른 가중치를 줄 수 있게 라벨만 붙여 둔다.
    """
    m0 = metrics[0]
    dmg = [damage_of(m, m0, args.goal_scale) for m in metrics]
    kinds = list(kinds) if kinds is not None else ["eff"] * len(metrics)
    pairs, skipped = [], 0
    for i in range(len(metrics)):
        for j in range(i + 1, len(metrics)):
            gap = dmg[j] - dmg[i]
            if gap < args.min_gap:
                skipped += 1
                continue
            both_ok = metrics[i]["success"] and metrics[j]["success"]
            both_eff = kinds[i] == "eff" and kinds[j] == "eff"
            kd = PAIR_EFF if (both_ok and both_eff) else PAIR_CLIFF
            pairs.append((row_ids[i], row_ids[j], float(gap), kd))
    return pairs, skipped


# ===========================================================================
# SECTION 3 — 전체 조립 + hdf5
# ===========================================================================
def build(families, goals, demo_eefs, args, log=print):
    """families → (rows, pairs, diag). rows 는 수락 family 의 rung 만.

    λ=0 rung 은 데모별로 하나만 저장한다 (Stage 8 이 데모마다 한 번 실행해 그
    데모의 모든 family 가 같은 객체를 공유한다). 이전에는 family 마다 다시
    넣어 95행 중 19행이 5개 궤적의 반복이었고, pairwise loss 에서 λ=0 앵커가
    데모당 최대 4배 과가중됐다.
    """
    rows, pairs, diag = [], [], []
    n_skip_pairs, n_dedup = 0, 0
    zero_row = {}                      # demo_id -> λ=0 rung 의 행 번호
    for fam in families:
        did = fam["demo_id"]
        kinds = rung_kinds(fam)
        fracs = list(fam.get("level_fracs", LEVELS))[:len(fam["trajectories"])]
        metrics = [measured_metrics(t, goals[did], r)
                   for t, r in zip(fam["trajectories"], fam["rho_endpoint"])]
        okay, reason = gate_family(fam, metrics, demo_eefs.get(did), args)
        m0 = metrics[0]
        step = gradedness([damage_of(m, m0, args.goal_scale)
                           for m, k in zip(metrics, kinds) if k == "eff"])[0]
        diag.append(dict(
            family_id=fam["family_id"], phase_id=int(fam["phase_id"]),
            accepted=bool(okay), reason=reason,
            lambda_ok=float(fam.get("lambda_ok", fam.get("lambda_max", 0.0))),
            lambda_fail=(None if fam.get("lambda_fail") is None
                         else float(fam["lambda_fail"])),
            rung_kind=kinds, level_fracs=fracs, min_step_frac=float(step),
            damage=[damage_of(m, m0, args.goal_scale) for m in metrics],
            success=[m["success"] for m in metrics]))
        log(f"  {'ACCEPT' if okay else 'reject':<7} {fam['family_id']:<26} "
            f"{reason}")
        if not okay:
            continue
        row_ids = []
        for lv, t, m, kd in zip(fracs, fam["trajectories"], metrics, kinds):
            if lv == 0.0 and did in zero_row:
                row_ids.append(zero_row[did])       # 공유 (중복 저장 안 함)
                n_dedup += 1
                continue
            rid = len(rows)
            row_ids.append(rid)
            if lv == 0.0:
                zero_row[did] = rid
            rows.append(dict(traj=t, m=m, level=float(lv), rung_kind=kd,
                             phase_id=int(fam["phase_id"]),
                             family_id=fam["family_id"], demo_id=did))
        p, sk = make_pairs(row_ids, metrics, args, kinds=kinds)
        n_skip_pairs += sk
        pairs.extend((bi, wj, int(fam["phase_id"]), mg, kd)
                     for bi, wj, mg, kd in p)
    if n_dedup:
        log(f"  [dedup] λ=0 rung {n_dedup}개를 데모별 1행으로 통합 "
            f"({len(zero_row)}개 고유)")
    return rows, pairs, diag, n_skip_pairs


def save_hdf5(path, rows, pairs, goals, meta):
    import h5py
    M = len(rows)
    if M == 0:
        # 수락 family 가 하나도 없을 수 있다 (게이트가 전부 기각). 그때도
        # 진단을 남기고 정상 종료해야 한다 — 예전에는 max() 가 빈 시퀀스로
        # ValueError 를 던져 G8 FAIL 리포트 직후 크래시했다.
        with h5py.File(path, "w") as h:
            g = h.create_group("pairs")
            for k, dt_ in [("i", np.int64), ("j", np.int64),
                           ("phase_id", np.int64), ("margin", np.float64),
                           ("kind", np.int8)]:
                g.create_dataset(k, data=np.zeros(0, dt_))
            h.create_group("trajs")
            h.attrs["meta"] = json.dumps(meta)
        return
    Tmax = max(len(r["traj"]["F"]) for r in rows)
    NF = len(rows[0]["traj"]["F"][0])
    adim = np.asarray(rows[0]["traj"]["actions"]).shape[1]
    F = np.zeros((M, Tmax, NF), np.float32)
    A = np.zeros((M, Tmax, adim), np.float32)
    L = np.zeros(M, np.int32)
    for i, r in enumerate(rows):
        f_ = np.asarray(r["traj"]["F"], np.float32)
        a_ = np.asarray(r["traj"]["actions"], np.float32)
        F[i, :len(f_)] = f_
        A[i, :len(a_)] = a_
        L[i] = len(f_)
    with h5py.File(path, "w") as h:
        g = h.create_group("pairs")
        g.create_dataset("i", data=np.array([p[0] for p in pairs], np.int64))
        g.create_dataset("j", data=np.array([p[1] for p in pairs], np.int64))
        g.create_dataset("phase_id",
                         data=np.array([p[2] for p in pairs], np.int64))
        g.create_dataset("margin",
                         data=np.array([p[3] for p in pairs], np.float64))
        # kind: 0 = 효율 pair (양쪽 성공·양쪽 효율 rung), 1 = cliff pair.
        # Stage 10 이 두 집합에 다른 가중치를 줄 수 있게 한다.
        g.create_dataset("kind",
                         data=np.array([p[4] for p in pairs], np.int8))
        t = h.create_group("trajs")
        t.create_dataset("F", data=F, compression="gzip")
        t.create_dataset("actions", data=A, compression="gzip")
        t.create_dataset("length", data=L)
        t.create_dataset("goal", data=np.array(
            [goals[r["demo_id"]] for r in rows], np.float64))
        t.create_dataset("lam", data=np.array(
            [r["m"]["lam"] for r in rows], np.float64))
        t.create_dataset("level", data=np.array(
            [r["level"] for r in rows], np.float64))
        t.create_dataset("phase_id", data=np.array(
            [r["phase_id"] for r in rows], np.int64))
        t.create_dataset("rung_kind", data=np.array(
            [r.get("rung_kind", "eff") for r in rows],
            dtype=h5py.string_dtype()))
        t.create_dataset("family_id", data=np.array(
            [r["family_id"] for r in rows], dtype=h5py.string_dtype()))
        t.create_dataset("demo_id", data=np.array(
            [r["demo_id"] for r in rows], dtype=h5py.string_dtype()))
        mg = t.create_group("measured")
        for k, dt_ in [("success", bool), ("final_goal_err", np.float64),
                       ("path_length", np.float64), ("jerk", np.float64),
                       ("slip", np.float64), ("rho_end", np.float64)]:
            mg.create_dataset(k, data=np.array(
                [r["m"][k] for r in rows], dtype=dt_))
        h.attrs["meta"] = json.dumps(meta)


def gate_g8(diag, pairs, args, log=print):
    n_acc = sum(1 for d in diag if d["accepted"])
    frac = n_acc / max(1, len(diag))
    phases = sorted(set(d["phase_id"] for d in diag))
    phase_ok = {p: any(d["accepted"] for d in diag if d["phase_id"] == p)
                for p in phases}
    n_eff = sum(1 for p in pairs if p[4] == PAIR_EFF)
    eff_frac = n_eff / max(1, len(pairs))
    ok = (frac >= args.accept_frac and all(phase_ok.values())
          and len(pairs) >= args.min_pairs
          and eff_frac >= args.min_eff_pair_frac)
    log(f"\n[G8] families {n_acc}/{len(diag)} 수락 ({frac:.0%}), "
        f"pairs {len(pairs)}, phase 생존 "
        f"{{{', '.join(f'{p}:{int(v)}' for p, v in phase_ok.items())}}}")
    log(f"[G8] pair 구성: 효율 {n_eff} ({eff_frac:.0%}) / "
        f"cliff {len(pairs) - n_eff} ({1 - eff_frac:.0%})"
        f"   — 효율 pair 가 R_θ 에게 '효율'을 가르치는 쪽이다")
    acc = [d for d in diag if d["accepted"]]
    if acc:
        st = [d["min_step_frac"] for d in acc if np.isfinite(d["min_step_frac"])]
        if st:
            log(f"[G8] 계단성 min_step_frac: 중앙값 {np.median(st):.3f}, "
                f"최소 {min(st):.3f} (균등 사다리 = 0.25, 기준 "
                f"≥ {args.min_step_frac})")
    log(f"[G8] {'PASS' if ok else 'FAIL'} (기준: 수락률 ≥ {args.accept_frac:.0%}"
        f", 전 phase ≥ 1, pairs ≥ {args.min_pairs}, "
        f"효율 pair ≥ {args.min_eff_pair_frac:.0%})")
    return ok


def run(args):
    from extract_features import load_cache
    families, meta8 = load_families(args.degradation)
    entries = load_cache(args.cache_dir)
    goals = {e["demo_id"]: np.asarray(e["goal"], float) for e in entries}
    demo_eefs = {e["demo_id"]: np.asarray(e["eef_pos"], float)
                 for e in entries}

    rows, pairs, diag, n_skip = build(families, goals, demo_eefs, args)
    ok = gate_g8(diag, pairs, args)
    print(f"[pair] nested 후보 중 min_gap 미달로 제외 {n_skip}")

    meta = dict(source=args.degradation, stage8=meta8, n_rows=len(rows),
                n_pairs=len(pairs), min_gap=args.min_gap,
                goal_scale=args.goal_scale, min_spearman=args.min_spearman,
                min_spread=args.min_spread,
                min_step_frac=args.min_step_frac,
                min_eff_pair_frac=args.min_eff_pair_frac,
                n_pairs_eff=int(sum(1 for p in pairs if p[4] == PAIR_EFF)),
                pair_kind_legend={"0": "efficiency", "1": "cliff"},
                g8_pass=bool(ok))
    out = os.path.join(args.out_dir, "preference.hdf5")
    save_hdf5(out, rows, pairs, goals, meta)
    with open(os.path.join(args.out_dir, "preference_diag.json"), "w") as f:
        json.dump(dict(g8_pass=bool(ok), families=diag), f,
                  ensure_ascii=False, indent=1)
    print(f"[out] {out}  ({len(rows)} rows, {len(pairs)} pairs)")
    if not rows:
        print("[preference] 수락 family 0개 — Stage 10 으로 넘길 pair 가 없다. "
              "preference_diag.json 의 reason 을 보고 Stage 7·8 을 조정하라.")

    if args.plot:
        viz_g8(diag, pairs, os.path.join(args.out_dir, "preference_g8.png"))
    return ok


# ===========================================================================
# SECTION 4 — 시각화
# ===========================================================================
def viz_g8(diag, pairs, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    phases = sorted(set(d["phase_id"] for d in diag))
    fig, axes = plt.subplots(1, len(phases) + 1,
                             figsize=(4.6 * (len(phases) + 1), 4.0))
    for ax, ph in zip(axes, phases):
        for d in [x for x in diag if x["phase_id"] == ph]:
            c = "tab:green" if d["accepted"] else "tab:red"
            xs = d.get("level_fracs", LEVELS)[:len(d["damage"])]
            kinds = d.get("rung_kind", ["eff"] * len(d["damage"]))
            ax.plot(xs, d["damage"], color=c, alpha=0.6, lw=1.3)
            for x, y, s, kd in zip(xs, d["damage"], d["success"], kinds):
                ax.scatter(x, y, marker="o" if s else "x", color=c,
                           s=44 if kd == "cliff" else 26,
                           edgecolors="k" if kd == "cliff" else "none",
                           linewidths=0.6, zorder=3)
        ax.axvline(1.0, color="0.7", lw=0.6, ls=":")
        ax.set_title(f"phase {ph}: measured damage vs level\n"
                     f"(green=accepted, red=rejected)", fontsize=9)
        ax.set_xlabel("level (× λ_ok;  1.25 = cliff rung)")
        ax.set_ylabel("damage (measured)")
        ax.grid(alpha=0.3)
    eff = [p[3] for p in pairs if p[4] == PAIR_EFF]
    clf = [p[3] for p in pairs if p[4] != PAIR_EFF]
    axes[-1].hist([eff, clf], bins=30, stacked=True,
                  color=["tab:blue", "tab:orange"], alpha=0.85,
                  label=[f"efficiency ({len(eff)})", f"cliff ({len(clf)})"])
    axes[-1].set_title(f"pair margins (n={len(pairs)})", fontsize=9)
    axes[-1].set_xlabel("margin (damage diff)")
    axes[-1].legend(fontsize=7)
    axes[-1].grid(alpha=0.3)
    fig.suptitle("G8: family gate + nested preference pairs")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_png}")


# ===========================================================================
# SECTION 5 — selftest (mock family; 파일 불필요)
# ===========================================================================
def _mock_family(fid, phase, dmg_profile, succ_profile, goal, T=40,
                 kinds=None, fracs=None):
    """damage/success 프로파일을 실측 메트릭으로 역산한 가짜 family."""
    ji, si = fs.index_of("eef_jerk"), fs.index_of("object_slip")
    NF = fs.N_FEATURES
    fracs = list(fracs) if fracs is not None else list(LEVELS)
    kinds = list(kinds) if kinds is not None else ["eff"] * len(dmg_profile)
    trajs, rhos = [], []
    for lv, dg, sc in zip(fracs, dmg_profile, succ_profile):
        eef = np.cumsum(np.full((T, 3), 0.01), axis=0)
        obj = np.zeros((T, 3))
        # damage 는 final_goal_err 로 실어 나른다 (성공 항은 succ 로)
        residual = dg - 2.0 * (1.0 - float(sc))
        obj[-1] = goal + np.array([0.10 * residual, 0, 0])
        F = np.zeros((T, NF))
        F[:, ji], F[:, si] = 0.1 + lv, 0.05 * lv
        trajs.append(dict(eef=eef, obj=obj, grip=np.zeros(T),
                          contact=np.zeros(T, bool), actions=np.zeros((T, 7)),
                          F=F, k_index=np.arange(T), success=bool(sc),
                          length=T, realised=1.0, lam=lv))
        rhos.append(1.0)
    return dict(family_id=fid, candidate=dict(candidate_id=fid), demo_id="d0",
                phase_id=phase, lambda_max=1.0, lambda_ok=1.0,
                lambda_fail=1.25, lambda_kind="cliff",
                lambda_levels=list(fracs), level_fracs=fracs,
                rung_kind=kinds, probes=[],
                trajectories=trajs, rho_endpoint=rhos)


def run_selftest():
    import tempfile
    print("=== preference SELFTEST (mock families) ===")
    ok = True
    goal = np.array([0.5, 0.0, 0.85])

    class A:
        min_gap, goal_scale = 0.05, 0.10
        min_spearman, min_spread = 0.6, 0.15
        max_repro, accept_frac, min_pairs = 0.02, 0.4, 3
        min_step_frac, min_eff_pair_frac = 0.08, 0.5

    EFF5 = ["eff"] * 5
    EFF5C = ["eff"] * 5 + ["cliff"]
    F5C = list(LEVELS) + [CLIFF_LEVEL]
    fams = [
        # 균등 사다리 + cliff rung: 효율 5개 전부 성공, 마지막만 실패
        _mock_family("d0:good", 0, [0, .5, 1.0, 1.5, 2.0, 4.0],
                     [1, 1, 1, 1, 1, 0], goal, kinds=EFF5C, fracs=F5C),
        _mock_family("d0:flat", 1, [0, .01, .02, .01, .03],
                     [1, 1, 1, 1, 1], goal, kinds=EFF5),   # R4 spread 기각
        _mock_family("d0:nonmono", 1, [0, 2.0, .5, 1.0, .2],
                     [1, 1, 1, 1, 1], goal, kinds=EFF5),   # R3 단조성 기각
        _mock_family("d0:inv", 2, [0, .5, 1.0, 1.5, 2.0],
                     [1, 0, 1, 1, 0], goal, kinds=EFF5),   # R2 역전 기각
        _mock_family("d0:good2", 1, [0, .5, 1.0, 1.5, 2.0],
                     [1, 1, 1, 1, 1], goal, kinds=EFF5),
        # ★ R5: 단조·spread 는 통과하지만 아래 3칸이 붙은 실측형 사다리.
        #   구 게이트는 이것을 수락했다 (실측 평균 [0,.06,.17,.48,.98]).
        _mock_family("d0:cliffy", 2, [0, .06, .17, .48, .98],
                     [1, 1, 1, 1, 1], goal, kinds=EFF5),
        _mock_family("d0:good3", 2, [0, .5, 1.0, 1.5, 2.0],
                     [1, 1, 1, 1, 1], goal, kinds=EFF5),
    ]
    goals = {"d0": goal}
    demo_eefs = {"d0": np.cumsum(np.full((40, 3), 0.01), axis=0)}

    rows, pairs, diag, _ = build(fams, goals, demo_eefs, A,
                                 log=lambda *a: None)
    acc = {d["family_id"]: d["accepted"] for d in diag}
    expect = {"d0:good": True, "d0:flat": False, "d0:nonmono": False,
              "d0:inv": False, "d0:good2": True, "d0:cliffy": False,
              "d0:good3": True}
    for k, v in expect.items():
        if acc[k] != v:
            ok = False
            print(f"[FAIL] {k}: accepted={acc[k]}, 기대 {v} "
                  f"({next(d['reason'] for d in diag if d['family_id']==k)})")
    n_expect = sum(1 for v in expect.values() if v)
    print(f"gate: {sum(acc.values())}/{len(fams)} 수락 (기대 {n_expect})")
    print(f"  R5 계단성: d0:cliffy 기각 — "
          f"{next(d['reason'] for d in diag if d['family_id']=='d0:cliffy')}")

    # λ=0 dedupe: 수락 family 4개인데 λ=0 행은 데모당 1개여야 한다
    n_zero = sum(1 for r in rows if r["level"] == 0.0)
    if n_zero != 1:
        ok = False; print(f"[FAIL] λ=0 행 {n_zero}개 (기대 1 — 데모 d0 하나)")
    else:
        print(f"λ=0 dedupe: 수락 {sum(acc.values())} family, λ=0 행 1개 "
              f"(총 {len(rows)}행)")

    # pair kind: 양쪽 성공 = 효율, 실패 포함 = cliff
    n_eff = sum(1 for p in pairs if p[4] == PAIR_EFF)
    bad_kind = [p for p in pairs
                if (rows[p[0]]["m"]["success"] and rows[p[1]]["m"]["success"]
                    and rows[p[0]].get("rung_kind") == "eff"
                    and rows[p[1]].get("rung_kind") == "eff")
                != (p[4] == PAIR_EFF)]
    if bad_kind:
        ok = False; print(f"[FAIL] pair kind 오분류 {len(bad_kind)}개")
    else:
        print(f"pair kind: 효율 {n_eff} / cliff {len(pairs) - n_eff} "
              f"(분류 정확)")

    # pair 방향·마진: 낮은 level 이 항상 i (우세)
    bad = [p for p in pairs if not (rows[p[0]]["level"] < rows[p[1]]["level"])]
    if bad:
        ok = False; print(f"[FAIL] level 역방향 pair {len(bad)}개")
    if any(p[3] < A.min_gap for p in pairs):
        ok = False; print("[FAIL] margin < min_gap pair 존재")
    print(f"pairs: {len(pairs)}개, margin [{min(p[3] for p in pairs):.2f}, "
          f"{max(p[3] for p in pairs):.2f}], 방향 전부 정상")

    # 실패 rung 이 성공 rung 보다 항상 열세인가 (d0:good 의 최상단)
    for p in pairs:
        if rows[p[0]]["m"]["success"] < rows[p[1]]["m"]["success"]:
            pass  # i 성공, j 실패 = 정상
        if rows[p[1]]["m"]["success"] > rows[p[0]]["m"]["success"]:
            ok = False; print("[FAIL] 실패 rung 이 성공 rung 을 이김")

    g8 = gate_g8(diag, pairs, A, log=lambda *a: None)
    if not g8:
        ok = False; print("[FAIL] mock 구성으로 G8 이 PASS 여야 함")

    # hdf5 왕복
    tmp = tempfile.mktemp(suffix=".hdf5")
    save_hdf5(tmp, rows, pairs, goals, dict(test=True))
    import h5py
    with h5py.File(tmp, "r") as h:
        M = h["trajs/F"].shape[0]
        P = h["pairs/i"].shape[0]
        m_ok = (M == len(rows) and P == len(pairs)
                and json.loads(h.attrs["meta"])["test"] is True
                and h["trajs/measured/success"].dtype == bool)
    os.remove(tmp)
    if not m_ok:
        ok = False; print("[FAIL] hdf5 round-trip")
    else:
        print(f"hdf5 round-trip OK ({M} rows, {P} pairs)")

    print(f"\n[selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--degradation", default="./artifacts/degradation.npz")
    ap.add_argument("--cache-dir", default="./cache")
    ap.add_argument("--out-dir", default="./artifacts")
    ap.add_argument("--min-gap", type=float, default=0.05,
                    help="pair 성립에 필요한 최소 damage 차이")
    ap.add_argument("--goal-scale", type=float, default=0.10,
                    help="final_goal_err 정규화 스케일 [m]")
    ap.add_argument("--min-spearman", type=float, default=0.6)
    ap.add_argument("--min-spread", type=float, default=0.15)
    ap.add_argument("--min-step-frac", type=float, default=0.08,
                    help="G8 R5 계단성: 효율 rung 인접 증분의 최솟값 / 전체 폭. "
                         "균등 사다리는 0.25, 실측 실패 사례는 0.06")
    ap.add_argument("--min-eff-pair-frac", type=float, default=0.5,
                    help="G8: 효율 pair(양쪽 성공)가 전체 pair 중 차지할 최소 비율")
    ap.add_argument("--max-repro", type=float, default=0.02,
                    help="λ=0 rung 의 허용 데모 재현 오차 [m]")
    ap.add_argument("--accept-frac", type=float, default=0.5)
    ap.add_argument("--min-pairs", type=int, default=30)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(0 if run_selftest() else 1)
    raise SystemExit(0 if run(args) else 1)


if __name__ == "__main__":
    main()
