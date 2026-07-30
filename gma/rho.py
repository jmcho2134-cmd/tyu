#!/usr/bin/env python
"""
rho.py — Stage 5 산출물 subgoal.json 의 실행기 (pure numpy)
================================================================================

subgoal.json ★ 단일 진리원(D10)을 읽어 세 함수를 제공한다:

    rho(k, Phi)        phase k 안에서의 진행도 ρ_k ∈ [0,1]
    satisfies(k, Phi)  phase k 의 subgoal 조건 판정 (기본: persistent 조건만)
    phase_of(Phi)      현재 phase 라벨 z
    rho_worst(k, Phi)  조건 조합 위반 기준 진행도 (Stage 7 screening 용)

Phi 는 feature 행 (N_FEATURES,) 또는 행렬 (T, N_FEATURES) — feature_select
.NAMES 순서. 관측+goal 로 계산되는 열만 읽으므로 M1 데모와 M5 rollout 이
같은 코드를 쓴다.

조건 형태 (subgoal_discover 가 데이터에서 결정):
    eq  : |x − thresh| ≤ 0.5          (이진 event; thresh ∈ {0,1})
    le  : x ≤ thresh                   (thresh = mean + c·σ̃)
    ge  : x ≥ thresh                   (thresh = mean − c·σ̃)
    box : |x − mean| ≤ thresh          (thresh = c·σ̃)

phase_of 는 "satisfies(k−1) 이 참인 가장 높은 k" 를 취한다. 앞 phase 의
조건이 나중에 풀려도(릴리즈로 contact=0) 뒤 phase 의 조건이 유지되는 한
라벨은 되돌아가지 않는다 — D5(persistence 분리)와 함께 역행을 막는
두 번째 장치다.

ρ 의 표준화 스케일: 연속 feature 는 max(경계 σ, 0.25·전역 σ, sd_floor),
이진(eq) 은 0.5 로 고정한다. 경계 σ 만 쓰면 산포가 가장 좁은 feature 가
거리를 독점해 ρ 가 이벤트 순간까지 평평해지고(dense shaping 소멸),
전역 σ 만 쓰면 허용 오차의 의미가 사라진다. d_start(phase 시작 시점의
평균 거리)로 나눠 ρ = clip(1 − d/d_start, 0, 1)^alpha 로 정규화한다.

ρ vs ρ_worst — 목적이 다르면 거리도 달라야 한다
------------------------------------------------
subgoal 은 조건들의 AND 다 (satisfies 가 그렇게 판정한다). 그런데 ρ 의 거리는
RMS 평균이고, 전역 스칼라 d_start 하나로 정규화한다. 그래서 조건 하나를 완전히
파괴해도 나머지가 0 이면 √(1/J) 로 희석되고, 스케일이 큰 조건(멀리서 시작하는
거리 feature)이 d_start 를 독점한다. 실측 (PickPlaceBread, pre phase):

    파지 완전 실패 (contact 1→0)   Δρ = −0.148     ← 태스크가 죽는 사건
    eef 를 5cm 떼기                Δρ = −0.139     ← 그냥 손이 조금 떨어진 것

두 사건이 구별되지 않는다. Stage 7 은 argmin_d ρ 로 열화 방향을 고르므로,
"조합을 망가뜨리는 방향"의 신호 상한이 0.148 로 눌리고 앙상블 불확실성
(0.01~0.02) 대비 SNR 이 무너져 랜덤 방향이 steepest 를 이긴다 (실측: 3 phase
전부에서 steepest 탈락).

ρ_worst 는 두 가지를 바꾼다:

  (1) 거리를 "임계를 넘은 양"(violation margin)으로 정의한다. 조건이 성립하면
      정확히 0 이므로  satisfies == (모든 viol == 0)  이 되어 screening 목적함수와
      phase 판정이 같은 정의를 쓴다. ρ 는 조건 성립 여부와 무관한 mean 거리라
      이 정합이 없었다.
  (2) 조건별로 정규화한다 (d_start_cond[j] = phase 시작 시점 그 조건의 평균
      위반량). 이것이 지배적인 개선이다 — 위 예에서 contact 는 자기 스케일로
      정규화되어 Δρ_worst = −0.84 가 되고 eef 5cm 는 −0.07 에 머문다.

  집계는 p-norm:  d = ( mean_j (viol_j / d_start_cond_j)^p )^(1/p)
      p = 1    평균 (희석)
      p = 2    RMS (현행 ρ 와 같은 집계)
      p = 4    기본값. max 에 가까우면서 미분 가능 — steepest 의 FD 가 필요하다
      p → ∞    정확히 max (= AND 위반과 등가)
  정규화가 mean 기반이라 phase 시작 상태(모든 조건이 자기 시작 위반량)에서
  d = 1, 즉 ρ_worst = 0 이 되고 subgoal 달성 시 1 이 된다. 클립하지 않으므로
  "시작보다 더 나쁨"은 음수로 보인다 (rho_raw 와 같은 이유).

용도 분리:
    rho        클립 RMS      → Stage 11 PBRS shaping (매끄러움이 필요)
    rho_raw    비클립 RMS    → 하위호환 / 진단
    rho_worst  위반마진 p-norm → Stage 7 screening 목적함수
"""

import json

import numpy as np

import feature_select as fs


def condition_holds(x, typ, thresh, mean):
    """x 는 스칼라 또는 배열. -> bool 또는 bool 배열."""
    x = np.asarray(x, float)
    if typ == "eq":
        return np.abs(x - thresh) < 0.5
    if typ == "le":
        return x <= thresh
    if typ == "ge":
        return x >= thresh
    if typ == "box":
        return np.abs(x - mean) <= thresh
    raise ValueError(f"unknown condition type: {typ}")


EQ_TOL = 0.5          # condition_holds 의 eq 허용폭과 같아야 한다


def condition_violation(x, typ, thresh, mean):
    """조건을 얼마나 넘겼는가 (raw 단위, 성립하면 정확히 0).

    condition_holds 와 같은 임계를 쓰므로  violation == 0  ⟺  condition_holds
    가 성립한다. 이 등가성이 ρ_worst 를 satisfies 와 정합시키는 근거다.
    """
    x = np.asarray(x, float)
    if typ == "eq":
        dev = np.abs(x - thresh) - EQ_TOL
    elif typ == "le":
        dev = x - thresh
    elif typ == "ge":
        dev = thresh - x
    elif typ == "box":
        dev = np.abs(x - mean) - thresh
    else:
        raise ValueError(f"unknown condition type: {typ}")
    return np.maximum(dev, 0.0)


class Subgoal:
    """subgoal.json -> ρ / satisfies / phase_of."""

    def __init__(self, spec, names=None, p_norm=4.0):
        self.spec = spec
        self.names = list(names) if names is not None else list(fs.NAMES)
        meta = spec["meta"]
        self.labels = list(meta["canonical_labels"])
        self.K = len(self.labels)
        self.alpha = float(meta.get("alpha", 1.0))
        self.sd_floor = float(meta.get("sd_floor", 0.001))
        self.p_norm = float(p_norm)
        self._legacy_dsc = False

        self._ph = []
        for k in range(self.K):
            p = spec["phases"][str(k)]
            idx = np.array([self.names.index(n) for n in p["features"]], int)
            mu = np.asarray(p["mean"], float)
            sd = np.asarray(p["std"], float)
            gsd = np.asarray(p.get("global_std", sd), float)
            types = list(p["type"])
            scale = np.empty(len(idx))
            for j, t in enumerate(types):
                if t == "eq":
                    scale[j] = 0.5
                else:
                    scale[j] = max(sd[j], 0.25 * gsd[j], self.sd_floor)
            d_start = max(float(p["d_start"]), 1e-9)
            # 조건별 시작 위반량. 없으면(구 subgoal.json) 전역 d_start 를 공유해
            # 하위호환으로 돈다 — 이 경우 ρ_worst 의 조건별 정규화 이득이 없으니
            # subgoal_discover 를 다시 돌리는 것이 맞다.
            if "d_start_cond" in p:
                dsc = np.maximum(np.asarray(p["d_start_cond"], float), 1e-9)
            else:
                dsc = np.full(len(idx), d_start)
                self._legacy_dsc = True
            self._ph.append(dict(
                label=p["label"], idx=idx, mu=mu, scale=scale, types=types,
                thresh=np.asarray(p["thresh"], float),
                persistent=np.asarray(p["persistent"], bool),
                d_start=d_start, d_start_cond=dsc))

    @classmethod
    def load(cls, path, names=None):
        with open(path) as f:
            return cls(json.load(f), names=names)

    # -- 내부: (T?, N) -> (T?, n_feat) ------------------------------------
    def _cols(self, k, Phi):
        Phi = np.asarray(Phi, float)
        one = Phi.ndim == 1
        X = Phi[None, :] if one else Phi
        return X[:, self._ph[k]["idx"]], one

    def dist(self, k, Phi):
        """phase k subgoal 까지의 표준화 거리 (RMS over 조건 feature)."""
        X, one = self._cols(k, Phi)
        p = self._ph[k]
        z = (X - p["mu"][None, :]) / p["scale"][None, :]
        d = np.sqrt(np.mean(z * z, axis=1))
        return float(d[0]) if one else d

    def rho(self, k, Phi):
        d = self.dist(k, Phi)
        r = np.clip(1.0 - d / self._ph[k]["d_start"], 0.0, 1.0) ** self.alpha
        return float(r) if np.isscalar(d) else r

    def rho_raw(self, k, Phi):
        """클립 없는 선형 진행도 1 − d/d_start (음수 허용, alpha 미적용).

        screening(Stage 7)과 실측 Δρ 용. 클립판 rho 는 subgoal 에서 멀면 0
        에 포화되어 "더 나빠짐"이 보이지 않는다 — 열화 방향을 고르려면
        포화 없는 눈금이 필요하다. shaping(Φ)에는 클립판 rho 를 쓴다."""
        d = self.dist(k, Phi)
        r = 1.0 - d / self._ph[k]["d_start"]
        return float(r) if np.isscalar(d) else r

    # -- 위반마진 계열 (Stage 7 screening) ---------------------------------
    def violation(self, k, Phi, scaled=True):
        """phase k 조건별 위반량. -> (n_feat,) 또는 (T, n_feat).

        scaled=True 면 조건 스케일(연속: max(σ, 0.25σ_g, sd_floor) / eq: 0.5)로
        나눈 무차원 값. 전부 0 이면 satisfies(persistent_only=False) 와 등가.
        """
        X, one = self._cols(k, Phi)
        p = self._ph[k]
        V = np.empty_like(X)
        for j in range(X.shape[1]):
            V[:, j] = condition_violation(X[:, j], p["types"][j],
                                          p["thresh"][j], p["mu"][j])
        if scaled:
            V = V / p["scale"][None, :]
        return V[0] if one else V

    def dist_worst(self, k, Phi, p_norm=None):
        """조건별 정규화 위반량의 p-norm.  0 = subgoal 달성, 1 ≈ phase 시작."""
        V = self.violation(k, Phi, scaled=True)
        one = V.ndim == 1
        V = V[None, :] if one else V
        p = self._ph[k]
        t = V / p["d_start_cond"][None, :]
        q = float(self.p_norm if p_norm is None else p_norm)
        if not np.isfinite(q):                      # p → ∞ : 정확히 max
            d = t.max(axis=1)
        else:
            d = np.mean(t ** q, axis=1) ** (1.0 / q)
        return float(d[0]) if one else d

    def rho_worst(self, k, Phi, p_norm=None):
        """조건 조합 기준 진행도 1 − d_worst. 클립하지 않는다.

        Stage 7 의 목적함수:  d* = argmin_d ρ_worst( φ + FCM(s, a, λ·d) ).
        ρ(RMS·전역 d_start)와 달리 조건 하나를 파괴하면 그 조건 자신의 스케일로
        온전히 반영된다 — 모듈 헤더의 실측 대비 참조.
        """
        d = self.dist_worst(k, Phi, p_norm=p_norm)
        r = 1.0 - d
        return float(r) if np.isscalar(d) else r

    def satisfies(self, k, Phi, persistent_only=True):
        """phase k 의 subgoal 조건 전부 성립? persistent_only=True 가
        phase_of 용(D5). persistent 조건이 하나도 없으면 전체 조건으로
        폴백한다 (라벨이 영원히 못 올라가는 것보다는 낫다)."""
        X, one = self._cols(k, Phi)
        p = self._ph[k]
        sel = p["persistent"] if (persistent_only and p["persistent"].any()) \
            else np.ones(len(p["idx"]), bool)
        ok = np.ones(X.shape[0], bool)
        for j in np.where(sel)[0]:
            ok &= condition_holds(X[:, j], p["types"][j], p["thresh"][j],
                                  p["mu"][j])
        return bool(ok[0]) if one else ok

    def phase_of(self, Phi):
        """satisfies(k−1) 이 참인 가장 높은 k (없으면 0)."""
        Phi = np.asarray(Phi, float)
        one = Phi.ndim == 1
        X = Phi[None, :] if one else Phi
        z = np.zeros(len(X), int)
        for k in range(self.K - 1):
            z = np.maximum(z, (k + 1) * self.satisfies(k, X).astype(int))
        return int(z[0]) if one else z

    def phase_rho(self, Phi):
        """(z, ρ_z) 를 한 번에 — Stage 11 의 Φ = c·(z + ρ_z) 용."""
        Phi = np.asarray(Phi, float)
        one = Phi.ndim == 1
        X = Phi[None, :] if one else Phi
        z = self.phase_of(X)
        r = np.zeros(len(X))
        for k in range(self.K):
            m = z == k
            if m.any():
                r[m] = self.rho(k, X[m])
        return (int(z[0]), float(r[0])) if one else (z, r)


# ===========================================================================
# selftest — 손으로 만든 spec 으로 순수 로직 검증 (파일/데모 불필요)
# ===========================================================================
def _toy_spec():
    return dict(
        meta=dict(canonical_labels=["pre", "move", "post"], alpha=1.0,
                  sd_floor=0.001),
        phases={
            "0": dict(label="pre", features=["eef_object_dist", "contact"],
                      mean=[0.0, 1.0], std=[0.01, 0.0], global_std=[0.1, 0.5],
                      type=["le", "eq"], thresh=[0.02, 1.0],
                      persistent=[True, True], d_start=5.0, v=[0, 0],
                      d_start_cond=[10.0, 1.0]),
            "1": dict(label="move", features=["object_goal_dist"],
                      mean=[0.0], std=[0.01], global_std=[0.1],
                      type=["le"], thresh=[0.05],
                      persistent=[True], d_start=10.0, v=[0],
                      d_start_cond=[8.0]),
            "2": dict(label="post", features=["eef_object_dist"],
                      mean=[0.2], std=[0.02], global_std=[0.1],
                      type=["ge"], thresh=[0.15],
                      persistent=[True], d_start=8.0, v=[0],
                      d_start_cond=[4.0]),
        })


def run_selftest():
    print("=== rho SELFTEST ===")
    ok = True
    sg = Subgoal(_toy_spec())

    def row(eo, og, con):
        phi = np.zeros(fs.N_FEATURES)
        phi[fs.index_of("eef_object_dist")] = eo
        phi[fs.index_of("object_goal_dist")] = og
        phi[fs.index_of("contact")] = con
        return phi

    # condition_holds 4형
    checks = [condition_holds(0.5, "eq", 1.0, None) == False,
              condition_holds(0.9, "eq", 1.0, None) == True,
              condition_holds(0.01, "le", 0.02, None) == True,
              condition_holds(0.16, "ge", 0.15, None) == True,
              condition_holds(0.24, "box", 0.05, 0.2) == True,
              condition_holds(0.26, "box", 0.05, 0.2) == False]
    if not all(checks):
        ok = False; print(f"[FAIL] condition_holds: {checks}")
    else:
        print("condition_holds eq/le/ge/box OK")

    # phase_of: 시나리오 궤적
    far = row(0.3, 0.5, 0)          # 시작
    grasped = row(0.005, 0.5, 1)    # 파지됨
    at_goal = row(0.005, 0.02, 1)   # 운반 완료 (아직 파지)
    released = row(0.2, 0.02, 0)    # 릴리즈 + 후퇴 (contact 풀림)
    zs = [sg.phase_of(x) for x in (far, grasped, at_goal, released)]
    if zs != [0, 1, 2, 2]:
        ok = False; print(f"[FAIL] phase_of trajectory {zs} != [0,1,2,2]")
    else:
        print(f"phase_of: far/grasped/at_goal/released -> {zs} "
              f"(릴리즈로 contact 풀려도 역행 없음)")

    # rho 단조성: pre 에서 eef 가 다가올수록 ρ_0 증가, 경계에서 ~1
    ds = [0.3, 0.2, 0.1, 0.02, 0.0]
    rs = [sg.rho(0, row(d, 0.5, 1)) for d in ds]
    if not all(a <= b + 1e-9 for a, b in zip(rs, rs[1:])):
        ok = False; print(f"[FAIL] rho not monotone: {np.round(rs, 3)}")
    else:
        print(f"rho_0 monotone as eef approaches: {np.round(rs, 3)}")
    if not (0.0 <= min(rs) and max(rs) <= 1.0):
        ok = False; print("[FAIL] rho out of [0,1]")

    # ---- ρ_worst: 위반마진 계열 ------------------------------------------
    # (a) violation == 0  ⟺  condition_holds. 두 함수가 같은 임계를 써야 한다.
    tests = [(0.9, "eq", 1.0, None), (0.4, "eq", 1.0, None),
             (0.01, "le", 0.02, None), (0.03, "le", 0.02, None),
             (0.16, "ge", 0.15, None), (0.14, "ge", 0.15, None),
             (0.24, "box", 0.05, 0.2), (0.26, "box", 0.05, 0.2)]
    bad = [t for t in tests
           if bool(condition_holds(*t)) != (float(condition_violation(*t)) == 0.0)]
    if bad:
        ok = False; print(f"[FAIL] violation/holds 불일치: {bad}")
    else:
        print("condition_violation == 0  <=>  condition_holds  (4형 전부)")

    # (b) subgoal 달성 -> ρ_worst = 1, 모든 위반량 0
    achieved = row(0.0, 0.5, 1)
    if not (abs(sg.rho_worst(0, achieved) - 1.0) < 1e-9
            and np.allclose(sg.violation(0, achieved), 0.0)):
        ok = False; print(f"[FAIL] 달성 상태 ρ_worst={sg.rho_worst(0, achieved)}")
    else:
        print(f"달성 상태: ρ_worst=1.000, violation={sg.violation(0, achieved)}")

    # (c) 공정성 — 핵심. 두 조건을 각자의 phase 시작 수준으로 위반시키면
    #     ρ 는 크게 어긋나고 ρ_worst 는 같아야 한다 (모듈 헤더의 실측 근거).
    #     phase0: eef le(thresh .02, scale max(.01,.025,.001)=.025, dsc 10)
    #             contact eq(scale .5, dsc 1.0)
    brk_eef = row(0.02 + 10.0 * 0.025, 0.5, 1)      # 시작 수준 위반
    brk_con = row(0.0, 0.5, 0)                       # 반전 = 시작 수준(dsc 1.0)
    dr = [abs(sg.rho(0, x) - sg.rho(0, achieved)) for x in (brk_eef, brk_con)]
    dw = [abs(sg.rho_worst(0, x) - 1.0) for x in (brk_eef, brk_con)]
    spread_r = max(dr) / max(min(dr), 1e-12)
    spread_w = max(dw) / max(min(dw), 1e-12)
    print(f"동등 심각도 위반 2건: |Δρ|={np.round(dr, 3)} (편차 {spread_r:.1f}x), "
          f"|Δρ_worst|={np.round(dw, 3)} (편차 {spread_w:.1f}x)")
    if spread_w > 1.05:
        ok = False
        print("    [FAIL] ρ_worst 가 동등 심각도에 다른 값을 준다 — 조건별 "
              "정규화가 깨졌다")
    if spread_r < 2.0:
        print("    [note] 이 toy spec 에서는 ρ 의 편차가 작다 (실데이터는 6.7~94x)")

    # (d) 조건이 하나라도 깨지면 ρ_worst < 1, 전부 성립하면 = 1
    if not (sg.rho_worst(0, brk_con) < 1.0 and sg.satisfies(0, brk_con) is False):
        ok = False; print("[FAIL] 조건 파괴가 ρ_worst 에 반영되지 않음")

    # (e) p_norm=inf 는 정확히 max
    both = row(0.02 + 10.0 * 0.025, 0.5, 0)          # 두 조건 동시 위반
    d_max = sg.dist_worst(0, both, p_norm=float("inf"))
    t = sg.violation(0, both) / sg._ph[0]["d_start_cond"]
    if abs(d_max - float(np.max(t))) > 1e-9:
        ok = False; print(f"[FAIL] p=inf != max: {d_max} vs {np.max(t)}")
    else:
        print(f"p=inf -> max 일치 ({d_max:.3f}); p=4 -> {sg.dist_worst(0, both):.3f}")

    # (f) 구 subgoal.json(d_start_cond 없음) 하위호환
    legacy = _toy_spec()
    for k in legacy["phases"]:
        legacy["phases"][k].pop("d_start_cond")
    sg_l = Subgoal(legacy)
    if not sg_l._legacy_dsc or not np.isfinite(sg_l.rho_worst(0, achieved)):
        ok = False; print("[FAIL] legacy spec 하위호환 실패")
    else:
        print("legacy spec (d_start_cond 없음) -> 전역 d_start 공유로 동작")

    # 벡터화 = 스칼라 경로 일치
    M = np.stack([far, grasped, at_goal, released])
    if list(sg.phase_of(M)) != zs:
        ok = False; print("[FAIL] vectorized phase_of differs")
    z, r = sg.phase_rho(M)
    if list(z) != zs or r.shape != (4,):
        ok = False; print("[FAIL] phase_rho")
    else:
        print(f"vectorized phase_rho OK: z={list(z)}, rho={np.round(r, 3)}")

    print(f"\n[selftest] {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    raise SystemExit(0 if run_selftest() else 1) if a.selftest else \
        print("library module; --selftest to verify")
