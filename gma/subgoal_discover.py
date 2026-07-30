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
from rho import condition_holds, condition_violation

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

        # d_start_cond: phase 시작 시점의 "조건별" 위반량 (rho_worst 정규화용).
        # 전역 d_start 하나로 정규화하면 스케일이 큰 조건(멀리서 시작하는 거리
        # feature)이 눈금을 독점해 이진 조건 파괴가 안 보인다 — rho.py 헤더의
        # 실측(파지 실패 Δρ=−0.148) 참조.
        vio = np.zeros((len(used), len(feats)))
        for i in range(len(used)):
            for j, (n, t_, th, mu) in enumerate(zip(feats, typ, thr, mean)):
                vio[i, j] = float(condition_violation(S[n][i], t_, th, mu))
        raw_dsc = (vio / scale[None, :]).mean(axis=0)
        # 바닥값. 두 가지를 동시에 막아야 한다:
        #   (a) 시작 위반량 0 인 조건 (phase 시작에 이미 성립) → 0 으로 나눌 수 없음
        #   (b) 0 은 아니지만 극히 작은 값 → 1/dsc 증폭이 폭발한다. 실측:
        #       post phase 의 object_goal_dist 가 0.009 로 나와 Δρ_worst 가 −6.7
        #       까지 튀었고, phase 0(−0.02)과 눈금이 3백 배 어긋났다.
        # 기준을 최댓값이 아니라 '0 이 아닌 값들의 중앙값'으로 잡는다. 최댓값
        # 기준이면 큰 거리 feature 가 바닥을 밀어올려 이진 조건의 정당한 시작
        # 위반량(contact: 정확히 1.0)까지 덮어버린다 (실측: 1.0 → 1.299).
        pos = raw_dsc[raw_dsc > 1e-9]
        ref = float(np.median(pos)) if len(pos) else 1.0
        d_start_cond = np.maximum(raw_dsc, max(0.10 * ref, 1e-6))

        phases[str(k)] = dict(
            label=canonical[k], features=feats,
            mean=[round(x, 6) for x in mean],
            std=[round(x, 6) for x in std],
            v=[round(x, 6) for x in vv],
            type=typ, thresh=[round(x, 6) for x in thr],
            persistence=[round(x, 4) for x in pers],
            persistent=persistent,
            global_std=[round(x, 6) for x in g_std],
            boundary_t=t_end, d_start=round(d_start, 4),
            d_start_cond=[round(float(x), 6) for x in d_start_cond],
            sd_floor=sd_floor,
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


# ===========================================================================
# 리포트 — 추론된 subgoal 을 사람이 읽는 형태로 (터미널)
# ===========================================================================
_OP = dict(eq="==", le="<=", ge=">=", box="≈")

W = 78


def _cond_str(n, typ, th, mu):
    """조건 하나를 사람이 읽는 식으로. box 는 mean±tol 로 펼친다."""
    if typ == "box":
        return f"|{n} − {mu:.4g}| <= {th:.4g}"
    if typ == "eq":
        return f"{n} == {th:.0f}"
    return f"{n} {_OP[typ]} {th:.4g}"


def _broken_value(typ, mu, th, sc, dsc):
    """그 조건을 'phase 시작과 같은 수준'으로 위반시킨 값.

    조건마다 파괴량을 임의로 정하면(예: 10·scale) 비교가 무의미하다. phase 시작
    위반량 d_start_cond 로 맞추면 "그 phase 가 시작될 때만큼 이 조건이 깨진
    상태" — 조건들 사이에 동등하게 심각한 사건이 된다. 이 기준에서 ρ 가 조건마다
    다른 값을 주면 그것이 곧 RMS·전역 d_start 정규화의 결함이다.
    """
    v = float(dsc) * float(sc)                      # raw 단위 위반량
    if typ == "eq":
        return (1.0 - th) if th in (0.0, 1.0) else th + v + 0.5
    if typ == "le":
        return th + v
    if typ == "ge":
        return th - v
    return mu + th + v                              # box


def report(spec, names=None, out=print):
    """추론된 subgoal 전체를 터미널에 출력한다.

    보여주는 것: phase 별 조건 조합(AND), persistent 표시, 선택 근거 v,
    ρ 스케일, 그리고 "조건을 하나씩 완전히 파괴했을 때 ρ / ρ_worst 가 얼마나
    떨어지는가" — 마지막 항목이 Stage 7 이 실제로 최적화할 신호의 크기다.
    """
    from rho import Subgoal

    sg = Subgoal(spec, names=names)
    m = spec["meta"]
    NF = len(sg.names)

    out("=" * W)
    out("추론된 SUBGOAL  (Stage 5 산출물 · 단일 진리원)")
    out("=" * W)
    out(f"데모 {m['n_demos_used']}편 사용"
        + (f" / 제외 {len(m['demos_excluded'])}편" if m.get("demos_excluded")
           else " / 제외 없음")
        + f"      canonical: {' | '.join(m['canonical_labels'])}")
    if m.get("demos_excluded"):
        for d, lab in m["demos_excluded"].items():
            out(f"    제외 {d}: {' | '.join(lab) if isinstance(lab, list) else lab}")
    out(f"하이퍼파라미터: theta={m['theta']}  c={m['c']}  alpha={m['alpha']}  "
        f"max_feat={m['max_feat']}  persist_min={m['persist_min']}")
    if m.get("constant_features_dropped"):
        out(f"상수로 탈락한 feature: {m['constant_features_dropped']}")
    if m.get("boundary_features_dropped"):
        out("※ boundary feature 제외 모드 (V2 순환성 ablation)")

    for k in range(sg.K):
        p = spec["phases"][str(k)]
        ph = sg._ph[k]
        out("")
        out(f"── phase {k}  [{p['label']}]  "
            + "─" * max(1, W - 20 - len(p["label"]))
            + f" 경계 t={p['boundary_t']}")

        out("  달성 조건 (AND — 하나라도 깨지면 subgoal 미달):")
        for j, n in enumerate(p["features"]):
            star = "★" if p["persistent"][j] else " "
            out(f"    {star} {_cond_str(n, p['type'][j], p['thresh'][j], p['mean'][j]):<34}"
                f" v={p['v'][j]:.4f}  persist={p['persistence'][j]:.2f}"
                f"  scale={ph['scale'][j]:.5g}")
        n_pers = int(np.sum(p["persistent"]))
        out(f"    ★ = persistent ({n_pers}/{len(p['features'])}) → "
            f"phase_of · satisfies 판정에 사용")
        expr = "  AND  ".join(
            _cond_str(n, t, th, mu)
            for n, t, th, mu, pr in zip(p["features"], p["type"], p["thresh"],
                                        p["mean"], p["persistent"]) if pr)
        out(f"    satisfies({k}) ≡ {expr or '(persistent 조건 없음 → 전체 조건 폴백)'}")

        out(f"  ρ 정규화: d_start={p['d_start']:.4f} (RMS·전역)   "
            f"d_start_cond={np.round(ph['d_start_cond'], 3).tolist()} (조건별)")

        # 민감도: subgoal 달성 상태에서 조건을 하나씩 파괴
        base = np.zeros(NF)
        for n, mu in zip(p["features"], p["mean"]):
            base[sg.names.index(n)] = mu
        r0, w0 = sg.rho(k, base), sg.rho_worst(k, base)
        out(f"  민감도 (달성 상태 ρ={r0:.3f} / ρ_worst={w0:.3f}). 각 조건을"
            f" 'phase 시작과 같은 수준'으로")
        out(f"           위반시킨다 — 조건들 사이에 동등하게 심각한 사건이다:")
        dr, dw = [], []
        for j, n in enumerate(p["features"]):
            q = base.copy()
            q[sg.names.index(n)] = _broken_value(
                p["type"][j], p["mean"][j], p["thresh"][j], ph["scale"][j],
                ph["d_start_cond"][j])
            a, b = sg.rho(k, q) - r0, sg.rho_worst(k, q) - w0
            # 이진(eq) 조건은 '반전' 외의 파괴가 없다. phase 시작 위반량이 1.0
            # 미만이면(경계에서 이미 일부 데모가 반대 상태) 반전은 시작보다 더
            # 심한 사건이므로 동등 비교에서 빼고 배수를 함께 적는다.
            vs = float(sg.violation(k, q)[j] / ph["d_start_cond"][j])
            comparable = abs(vs - 1.0) < 1e-6
            mark = "" if comparable else f"   [시작의 {vs:.1f}배 — 이진 반전]"
            if comparable:
                dr.append(abs(a)); dw.append(abs(b))
            out(f"    {n:<18} Δρ={a:+.3f}   Δρ_worst={b:+.3f}"
                f"   satisfies→{sg.satisfies(k, q, persistent_only=False)}{mark}")
        if len(dr) > 1:
            sr = max(dr) / max(min(dr), 1e-9)
            sw = max(dw) / max(min(dw), 1e-9)
            out(f"    → 동등 심각도 조건 {len(dr)}개의 편차:  ρ {sr:.1f}배  vs  "
                f"ρ_worst {sw:.1f}배  (1.0 에 가까울수록 공정)")
        if p.get("note"):
            out(f"  note: {p['note']}")

    out("")
    out("=" * W)
    out("Δρ_worst 가 Stage 7 의 최적화 신호입니다 (rho.py 헤더 참조).")
    out("|Δρ_worst| 가 작은 조건은 FCM screening 이 사실상 볼 수 없습니다.")
    out("=" * W)
    return sg


def run(cache_dir="./cache", boundaries_path="./artifacts/boundaries.json",
        out_path="./artifacts/subgoal.json", show_report=True, **params):
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
    print(f"[out] {out_path}\n")
    if show_report:
        report(spec)
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
    ap.add_argument("--report", action="store_true",
                    help="추론은 하지 않고 기존 subgoal.json 을 읽어 리포트만 출력")
    ap.add_argument("--no-report", action="store_true",
                    help="추론 후 리포트 출력 생략")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(0 if run_selftest() else 1)
    if args.report:
        with open(args.out) as f:
            report(json.load(f))
        return
    run(args.cache_dir, args.boundaries, args.out,
        show_report=not args.no_report,
        theta=args.theta, c=args.c, alpha=args.alpha, max_feat=args.max_feat,
        pad=args.pad, sd_floor=args.sd_floor, persist_min=args.persist_min,
        drop_boundary_features=args.drop_boundary_features)


if __name__ == "__main__":
    main()
