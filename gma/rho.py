#!/usr/bin/env python
"""
rho.py — Stage 5 산출물 subgoal.json 의 실행기 (pure numpy)
================================================================================

subgoal.json ★ 단일 진리원(D10)을 읽어 세 함수를 제공한다:

    rho(k, Phi)        phase k 안에서의 진행도 ρ_k ∈ [0,1]
    satisfies(k, Phi)  phase k 의 subgoal 조건 판정 (기본: persistent 조건만)
    phase_of(Phi)      현재 phase 라벨 z

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


class Subgoal:
    """subgoal.json -> ρ / satisfies / phase_of."""

    def __init__(self, spec, names=None):
        self.spec = spec
        self.names = list(names) if names is not None else list(fs.NAMES)
        meta = spec["meta"]
        self.labels = list(meta["canonical_labels"])
        self.K = len(self.labels)
        self.alpha = float(meta.get("alpha", 1.0))
        self.sd_floor = float(meta.get("sd_floor", 0.001))

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
            self._ph.append(dict(
                label=p["label"], idx=idx, mu=mu, scale=scale, types=types,
                thresh=np.asarray(p["thresh"], float),
                persistent=np.asarray(p["persistent"], bool),
                d_start=max(float(p["d_start"]), 1e-9)))

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
                      persistent=[True, True], d_start=5.0, v=[0, 0]),
            "1": dict(label="move", features=["object_goal_dist"],
                      mean=[0.0], std=[0.01], global_std=[0.1],
                      type=["le"], thresh=[0.05],
                      persistent=[True], d_start=10.0, v=[0]),
            "2": dict(label="post", features=["eef_object_dist"],
                      mean=[0.2], std=[0.02], global_std=[0.1],
                      type=["ge"], thresh=[0.15],
                      persistent=[True], d_start=8.0, v=[0]),
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
