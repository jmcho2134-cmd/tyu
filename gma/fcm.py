#!/usr/bin/env python
"""
fcm.py — Stage 6 (FCM 데이터 수집 + 학습) & Stage 7 (방해 action 집합 도출)
================================================================================

야말 fcm.py 의 검증된 부품(state 분기 rollout, sklearn MLP, feasible cone)을
PIPELINE_v4 계약으로 재구성한 것. v4 에서 달라진 점:

  * 목적함수가 rho.Subgoal (★ 통합 지점). 단일 데모 phase_subgoal_set 은
    쓰지 않는다. 이전 시그니처가 필요한 곳을 위해 score_terms(sg, k) 어댑터만
    남겼다.
  * residual 정의가 v4 Sec 3.5:  r(t,h) = φ_pert(t+h) − φ_demo(t+h).
    (야말은 φ(t+h) − φ(t) — 앵커 기준 변화량이었다.)
  * FCM 은 seed 만 다른 앙상블 → 예측 평균 + 표준편차(불확실성 penalty).
  * screening:  d*_z = argmin_d  ρ_z( φ_t + FCM(s,a,λ_probe·d) )   (Sec 6)
    불확실성 penalty + cone 투영 + cos 중복 제거 + phase 별 Top-K.

산출물
------
    artifacts/fcm_dataset.hdf5     FCMSample 평탄화 (Sec 3.5)
    artifacts/fcm_ensemble.pkl     sklearn 앙상블 (문서의 .pt 자리; torch 아님)
    artifacts/action_sets.json     Sec 3.6
    artifacts/fcm_*.png            G6 / fit R^2 / screening / direction set

게이트
------
    G6  λ=0 → residual ≈ 0        (분기·리플레이가 깨졌으면 학습 진행 금지)
    G7  Top-K recall > random     (FCM 랭킹이 실측 랭킹을 이겨야 의미)

    python fcm.py --selftest                  # numpy 만: mock 물리로 전 구간
    python fcm.py collect|train|screen|all    # 실데모 (robosuite 필요)
"""

import argparse
import json
import os
import pickle

import numpy as np

import feature_select as fs
from rho import Subgoal

try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    _HAS_SK = True
except Exception:                                             # pragma: no cover
    _HAS_SK = False

def fcm_input_mask():
    """FCM 입력으로 쓸 feature 열. R_θ 의 reward_input_mask 와 값은 같지만
    근거가 다르므로 이름을 분리한다: 여기서 action_magnitude / energy 를 빼는
    이유는 "λ 의 대리변수" (그건 Stage 10 의 문제) 가 아니라, 둘이 δ 로부터
    거의 결정론적으로 정해져 잔차 예측에 새 정보를 주지 않기 때문이다."""
    return fs.reward_input_mask()


N_PHASE_SLOTS = 8                     # one-hot 자리 (phase 3개여도 고정폭)
# obs 는 절대 좌표만 (quat 제외): screening 은 cache(F, eef_pos, obj_pos)만
# 으로 돌아야 하므로, 학습 입력도 같은 채널로 제한해 분포를 일치시킨다.
N_OBS = 3 + 3                         # eef_pos, obj_pos
AXIS_LABELS = ["dx", "dy", "dz", "drx", "dry", "drz", "grip"]


# ===========================================================================
# SECTION 1 — 통합 지점: rho.Subgoal → (구형) score_terms
# ===========================================================================
def score_terms(sg, k, sub_weight=1.0):
    """rho.Subgoal 의 phase k 조건 → [(feature_idx, coef, kind), ...].

    구형 소비자(subgoal_score 류)와의 호환 어댑터. 조건 형태가 부호를 정한다:
        le  → 값을 키우면 열화     (+1, signed)
        ge  → 값을 줄이면 열화     (−1, signed)
        eq/box → 어느 쪽이든 이탈이 열화 (abs)
    새 코드는 이 어댑터 대신 ρ 를 직접 쓴다 (score 함수가 곧 −Δρ).
    """
    p = sg.spec["phases"][str(k)]
    terms = []
    for n, typ in zip(p["features"], p["type"]):
        idx = fs.index_of(n)
        if typ == "le":
            terms.append((idx, +float(sub_weight), "signed"))
        elif typ == "ge":
            terms.append((idx, -float(sub_weight), "signed"))
        else:
            terms.append((idx, float(sub_weight), "abs"))
    return terms


# ===========================================================================
# SECTION 2 — Stage 6: state 분기 데이터 수집
# ===========================================================================
class DemoRollout:
    """데모 state t 에 앵커를 박고, 같은 state 에서 두 branch 를 민다.

        Branch A (λ=0)  : demo action 그대로  → 리플레이 검증 (G6)
        Branch B        : demo action + delta → residual 라벨

    open-loop 인 이유(야말 주석 그대로): "여기서 다르게 행동했다면"의 delta 가
    정확히 식별되어야 하고, 추적 제어기를 끼우면 그 보정이 action 에 섞여
    귀속이 불가능해진다. H 가 짧아(기본 8 스텝 = 0.4 s) drift 는 유계.

    residual 은 v4 정의: r(t,h) = φ_branch(t+h) − φ_demo(t+h).
    양쪽 φ 를 같은 코드로(pre 프레임 포함, 같은 윈도우) 계산해 미분계열
    feature 의 수치 차이가 residual 에 새지 않게 한다.
    """

    def __init__(self, env, states, actions, fr, dt, object_type=None, pre=3,
                 warmup=5):
        import frame_extract as fx
        self.env = env
        self.states = np.asarray(states, float)
        self.actions = np.asarray(actions, float)
        self.fr = fr                              # frame_extract.Frames (demo)
        self.dt, self.pre = float(dt), int(pre)
        self.warmup = int(warmup)
        self.adim = self.actions.shape[1]
        self.T = len(states)
        self.goal = np.asarray(fr.goal, float)
        self.obj0_z = float(fr.obj0_z)
        self.ex = fx.FrameExtractor(env, object_type, dt=dt)
        self.grip_model = self._resolve_gripper()
        self.ca_series = self._integrate_gripper()

    def _resolve_gripper(self):
        try:
            g = self.env.robots[0].gripper
            if isinstance(g, dict):
                g = g.get("right") or next(iter(g.values()))
            return g if hasattr(g, "current_action") else None
        except Exception:
            return None

    def _integrate_gripper(self):
        """gripper 의 current_action 적분기 궤적을 데모 액션으로 재구성.

        robosuite 그리퍼는 format_action 안에서 current_action 을 적분하는데
        (Robotiq85: ca += 0.2·sign(a)), 이 값은 파이썬 객체에 살지 MuJoCo
        state 가 아니라서 set_state_from_flattened 가 복원하지 못한다.
        복원 없이 분기하면 직전 branch 가 남긴 값에서 이어가 그리퍼가 엉뚱한
        목표로 움직인다 (실측: anchor 63% 드롭, gripper_open drift 폭발).
        그리퍼 자신의 format_action 을 0 에서부터 다시 돌려 t 시점 값을
        얻는다 — 수집도 에피소드 reset(=0)에서 시작했으므로 정확히 일치."""
        g = self.grip_model
        if g is None:
            return None
        saved = np.copy(g.current_action)
        g.current_action = np.zeros(g.dof)
        series = [np.copy(g.current_action)]
        for a in self.actions:
            try:
                g.format_action(np.atleast_1d(a[-g.dof:]))
            except Exception:
                g.current_action = saved
                return None
            series.append(np.copy(g.current_action))
        g.current_action = saved
        return series

    def obs_row(self, t):
        return np.concatenate([self.fr.eef_pos[t], self.fr.obj_pos[t]])

    def phi_at(self, t):
        """anchor 시점의 feature 행 (데모 프레임에서 직접)."""
        f = self.fr
        return fs.compute_trajectory(
            f.eef_pos[t:t + 1], f.obj_pos[t:t + 1], f.grip[t:t + 1], self.dt,
            f.eef_quat[t:t + 1], f.obj_quat[t:t + 1], self.goal, self.obj0_z,
            contact=f.contact[t:t + 1])[0]

    def _feat(self, eef, obj, grip, eq, oq, con, acts):
        return fs.compute_trajectory(
            np.array(eef), np.array(obj), np.array(grip), self.dt,
            np.array(eq), np.array(oq), self.goal, self.obj0_z,
            actions=np.array(acts), contact=np.array(con))

    def _win_start(self, t):
        """윈도우 시작 인덱스. pre(미분계열용 과거)와 warmup(리플레이 구간)의
        더 이른 쪽을 쓴다 — branch 와 demo_window_features 가 같은 길이의
        윈도우를 써야 residual 이 index 정렬된다."""
        return max(0, t - max(self.pre, self.warmup))

    def _window_history(self, t):
        lo = self._win_start(t)
        f = self.fr
        return (lo, [list(f.eef_pos[lo:t + 1]), list(f.obj_pos[lo:t + 1]),
                     list(f.grip[lo:t + 1]), list(f.eef_quat[lo:t + 1]),
                     list(f.obj_quat[lo:t + 1]), list(f.contact[lo:t + 1]),
                     [self.actions[j] for j in range(lo, t)]])

    def demo_window_features(self, t, H):
        """데모 자신을 같은 윈도우 방식으로 계산한 φ_demo(t+h). branch 와
        수치적으로 동일한 파이프라인 → residual 에 계산 아티팩트가 없다."""
        lo = self._win_start(t)
        i0 = t - lo
        hi = min(self.T, t + H + 1)
        f = self.fr
        F = self._feat(f.eef_pos[lo:hi], f.obj_pos[lo:hi], f.grip[lo:hi],
                       f.eef_quat[lo:hi], f.obj_quat[lo:hi], f.contact[lo:hi],
                       [self.actions[j] for j in range(lo, hi - 1)])
        out = np.zeros((H, fs.N_FEATURES))
        for h in range(1, H + 1):
            out[h - 1] = F[min(i0 + h, len(F) - 1)]
        return out

    def branch(self, t, delta, H):
        """state t 에서 delta 를 H 스텝 유지 주입.
        -> (φ_branch(t+1..t+H), info). delta=0 이 Branch A.
        info["phi0"] = 분기 시작점의 **실측** feature 행 (아래 참조)."""
        lo, (eef, obj, grip, eq, oq, con, acts) = self._window_history(t)
        i0 = t - lo
        # 웜업 리플레이: t−W 에서 state 를 넣고 데모 액션으로 W 스텝 밀어
        # 분기 시작점 t 에 도달한다. set_state 는 qacc_warmstart 나 controller
        # 내부값까지 복원하지 못해 첫 스텝에 미세 트랜지언트가 생기는데
        # (grasp_align ~0.9° 실측), 웜업이 그 상태들을 자연스럽게 데모와
        # 일치시킨다. 그리퍼 적분기는 웜업 시작점 기준으로 복원.
        tw = max(0, t - self.warmup)
        self.env.sim.set_state_from_flattened(self.states[tw])
        self.env.sim.forward()
        if self.grip_model is not None and self.ca_series is not None:
            self.grip_model.current_action = np.copy(self.ca_series[tw])

        # 윈도우 [tw, t] 를 리플레이 **실측** 프레임으로 교체한다.
        # 이전에는 이 구간이 데모 프레임이었다. 그러면 학습 입력 C = φ(t) 는
        # "데모의 t 상태"인데 라벨 Y 는 "리플레이된 t 상태에서 일어난 일"이 되어
        # 초기조건이 어긋난다. G6 가 재던 드리프트의 정체가 바로 이 불일치이고,
        # 리플레이가 가장 안 맞는 채널(gripper_open·contact)에서 가장 커진다.
        # 교체하면 C 와 Y 가 같은 초기조건을 가리키므로 δ 의 귀속이 정확해진다.
        n_keep = max(0, tw - lo)               # [lo, tw) 는 관측 불가 → 데모 유지
        eef, obj, grip = eef[:n_keep], obj[:n_keep], grip[:n_keep]
        eq, oq, con = eq[:n_keep], oq[:n_keep], con[:n_keep]

        def _push_frame(frm):
            eef.append(frm["eef_pos"]); obj.append(frm["obj_pos"])
            grip.append(frm["grip"]); eq.append(frm["eef_quat"])
            oq.append(frm["obj_quat"]); con.append(frm["contact"])

        _push_frame(self.ex._frame(with_dynamics=False))        # 프레임 tw
        for j in range(tw, t):
            try:
                self.env.step(self.actions[j])
            except Exception:
                return None, dict(clip=0.0, ok=False)
            _push_frame(self.ex._frame(with_dynamics=False))    # tw+1 .. t
        clip_amt, ok = 0.0, True
        for h in range(H):
            a_base = self.actions[min(t + h, len(self.actions) - 1)]
            clip_amt = max(clip_amt, fs.clip_fraction(a_base, delta, self.adim))
            a = fs.perturb_action(a_base, delta, self.adim,
                                  proper_rotation=True)
            try:
                self.env.step(a)
            except Exception as ex:
                ok = False
                print(f"[warn] step failed at t={t}: {ex}")
                break
            frm = self.ex._frame(with_dynamics=False)
            eef.append(frm["eef_pos"]); obj.append(frm["obj_pos"])
            grip.append(frm["grip"]); eq.append(frm["eef_quat"])
            oq.append(frm["obj_quat"]); con.append(frm["contact"])
            acts.append(a)
        n = len(eef)
        if not ok or n <= i0 + 1:
            return None, dict(clip=clip_amt, ok=False)
        F = self._feat(eef, obj, grip, eq, oq, con, acts)
        out = np.zeros((H, fs.N_FEATURES))
        for h in range(1, H + 1):
            out[h - 1] = F[min(i0 + h, n - 1)]
        return out, dict(clip=clip_amt, ok=True, phi0=F[i0].copy())


def graded_levels(base, n_levels=4, lo=0.25, hi=4.0):
    return base * np.exp(np.linspace(np.log(lo), np.log(hi), n_levels))


def phase_anchors(z, phase, n, T, horizon, inset_frac=0.08):
    """phase 내부의 균등 anchor. 경계에서 inset 만큼 안쪽으로 들여 배치한다:
    경계 프레임은 접촉 형성/해제의 카오스 순간이라 open-loop 분기가 원리적으로
    재현 불가하고 (실측: 경계 위 anchor 만 drift 폭발), Stage 8 의 주입 창도
    phase 내부이므로 학습 분포를 거기에 맞춘다."""
    idx = np.where(np.asarray(z)[:T] == phase)[0]
    idx = idx[idx < T - horizon - 1]
    if len(idx) == 0:
        return []
    inset = max(2, int(inset_frac * len(idx)))
    if len(idx) > 2 * inset + 1:
        idx = idx[inset:-inset]
    if len(idx) <= n:
        return [int(i) for i in idx]
    return [int(idx[i]) for i in np.linspace(0, len(idx) - 1, n).astype(int)]


def z_from_bounds(bounds, T):
    z = np.zeros(T, int)
    for i, b in enumerate(bounds):
        z[b:] = i + 1
    return z


def sample_directions(rng, adim, n, demo_dir=None):
    """v4 Sec 3.5 의 주입 종류를 섞는다: Gaussian / 좌표축 / 랜덤 단위벡터 /
    데모 평행(±). 전부 단위벡터로 반환."""
    out = []
    kinds = ["gauss", "axis", "unit", "para"]
    for i in range(n):
        kind = kinds[i % len(kinds)]
        if kind == "gauss":
            d = rng.normal(size=adim)
        elif kind == "axis":
            d = np.zeros(adim)
            d[rng.integers(adim)] = rng.choice([-1.0, 1.0])
        elif kind == "unit":
            d = rng.normal(size=adim)
            d[rng.integers(adim)] *= 3.0          # 축 치우친 랜덤
        else:                                      # demo 평행/역평행
            if demo_dir is None or np.linalg.norm(demo_dir) < 1e-9:
                d = rng.normal(size=adim)
            else:
                d = rng.choice([-1.0, 1.0]) * np.asarray(demo_dir, float)
        n_ = np.linalg.norm(d)
        out.append(d / n_ if n_ > 1e-12 else np.eye(adim)[0])
    return out


def collect(rollers, boundaries, sg, args, rng, scales=None, log=print):
    """모든 데모 × phase × anchor 에서 분기 데이터 수집 (Sec 3.5).

    라벨은 Branch A(λ=0 리플레이) 기준:  Y = φ_pert − φ_replay.
    set_state 직후의 controller 웜업 트랜지언트가 양 branch 에 공유되므로
    이 차분에서 상쇄되고, δ 의 귀속이 정확해진다. λ=0 행의 Y 에는 대신
    리플레이 드리프트(φ_replay − φ_demo)를 담는다 — G6 진단 전용이고
    학습 시에는 0 으로 치환된다 (cmd_train).

    anchor 자가 검증: Branch A 가 데모를 재현하지 못하는 anchor(정규화
    드리프트 > anchor_drift_tol)는 통째로 버린다. 접촉 형성 순간은 리플레이가
    1~2 프레임만 어긋나도 이진 채널(contact/gripper)이 뒤집혀 라벨이
    쓰레기가 되는데, 그런 곳에서 학습하지 않는 것이 맞다. 버린 비율은
    G6 가 함께 판정한다 (>50% 면 분기 자체가 깨진 것).
    """
    rows = dict(C=[], O=[], A=[], D=[], lam=[], Z=[], rho=[], h=[], Y=[],
                clip=[], ep=[], t=[], G=[])
    n_drop, n_anchor, n_anchor_drop, n_drop_rho = 0, 0, 0, 0
    tol = getattr(args, "anchor_drift_tol", 0.3)
    anchor_rho_tol = getattr(args, "anchor_rho_tol", None)
    for did, roller in rollers.items():
        T = roller.T
        zseg = z_from_bounds(boundaries[did]["bounds"], T)
        levels = graded_levels(args.delta_scale, args.levels)
        for phase in sorted(set(zseg.tolist())):
            anchors = phase_anchors(zseg, phase, args.anchors, T, args.horizon)
            if not anchors:
                continue
            a_ = int(np.argmax(zseg == phase))
            b_ = T - int(np.argmax(zseg[::-1] == phase))
            demo_dir = roller.actions[a_:b_].mean(axis=0)
            for t in anchors:
                n_anchor += 1
                base = roller.demo_window_features(t, args.horizon)
                outA, infoA = roller.branch(t, np.zeros(roller.adim),
                                            args.horizon)
                if outA is None:
                    n_anchor_drop += 1
                    continue
                drift = outA - base
                # ρ 공간 anchor 필터. 위치 채널 기준(아래)만으로는 접촉·그리퍼
                # 리플레이 오차가 걸러지지 않는데, ρ_worst 는 contact 를
                # 조건 스케일 0.5 / d_start_cond 1.0 으로 읽으므로 1 프레임
                # 뒤집힘이 곧 전체 스케일 이동이다. Stage 7 이 최적화하는 양이
                # 오염된 anchor 는 여기서 버리는 것이 맞다.
                phi_rep = infoA.get("phi0")
                if anchor_rho_tol is not None and phi_rep is not None:
                    zc0 = sg.phase_of(phi_rep)
                    d_rho = abs(sg.rho_worst(zc0, phi_rep + drift[-1])
                                - sg.rho_worst(zc0, phi_rep))
                    if d_rho > anchor_rho_tol:
                        n_anchor_drop += 1
                        n_drop_rho += 1
                        continue
                if scales is not None:
                    # anchor 필터는 위치 기반 progress 채널만 본다.
                    # 이벤트 채널(contact/gripper)은 1 프레임 지터로 RMS 가
                    # 튀고, 속도/저크류 quality 채널은 정지 물체의 미세
                    # 떨림이 재현되지 않아 노이즈비가 1 을 넘는다 (실측:
                    # 드롭의 전부가 object_ang_speed 류였고 위치 채널은 전부
                    # 깨끗). 그 채널들의 체계적 드리프트는 G6 집계가 잡는다.
                    prog = np.array([fs.SPEC[n].kind == fs.PROGRESS
                                     for n in fs.NAMES])
                    sc = np.where(prog, scales, 0.0)
                    dr = np.sqrt((drift ** 2).mean(axis=0))
                    ratio = np.where(sc > 1e-9, dr / np.maximum(sc, 1e-9), 0.0)
                    if float(np.nanmax(ratio)) > tol:
                        n_anchor_drop += 1
                        continue
                # 학습 입력 φ 는 데모 프레임이 아니라 **분기가 실제로 출발한**
                # 리플레이 상태다 (branch 의 phi0). 데모 φ 를 쓰면 C 와 Y 의
                # 초기조건이 어긋난다 — branch() 주석 참조.
                phi = infoA.get("phi0")
                if phi is None:
                    phi = roller.phi_at(t)
                zc = sg.phase_of(phi)
                rc = sg.rho(zc, phi)

                goal_row = np.asarray(roller.goal, float).ravel()[:3]

                def push(delta, lam, resid, clip):
                    for h in range(args.horizon):
                        rows["C"].append(phi)
                        rows["O"].append(roller.obs_row(t))
                        rows["A"].append(roller.actions[t])
                        rows["D"].append(delta)
                        rows["lam"].append(lam)
                        rows["Z"].append(zc)
                        rows["rho"].append(rc)
                        rows["h"].append((h + 1) * roller.dt)
                        rows["Y"].append(resid[h])
                        rows["clip"].append(clip)
                        rows["ep"].append(did)
                        rows["t"].append(t)
                        # goal 을 실제 입력으로 넣는다. 이전에는 fit 이 goal 을
                        # 받지 않아 3열이 항상 0 이었고, 데모마다 객체·bin 기하가
                        # 다른 20편 규모에서는 FCM 이 goal-상대 방향("bin 반대쪽
                        # 으로 밀기")을 학습할 근거가 없어진다.
                        rows["G"].append(goal_row)

                push(np.zeros(roller.adim), 0.0, drift, infoA["clip"])
                for lv in levels:
                    for d in sample_directions(rng, roller.adim,
                                               args.dirs_per_level, demo_dir):
                        delta = lv * d
                        out, info = roller.branch(t, delta, args.horizon)
                        if out is None or info["clip"] > args.max_clip:
                            n_drop += 1
                            continue
                        push(delta, float(lv), out - outA, info["clip"])
        log(f"  {did}: rows so far {len(rows['Y'])}")
    data = {k: np.asarray(v) for k, v in rows.items()}
    data["ep"] = np.asarray(rows["ep"], dtype="S16")
    data["anchor_drop_rate"] = np.array(
        n_anchor_drop / max(1, n_anchor))
    log(f"[collect] {len(data['Y'])} rows ({n_drop} branches dropped; "
        f"anchors kept {n_anchor - n_anchor_drop}/{n_anchor}"
        + (f", ρ 필터로 {n_drop_rho}개" if n_drop_rho else "") + ")")
    return data


def subgoal_channels(sg):
    """subgoal.json 의 어느 phase 에서든 조건으로 쓰인 feature 의 마스크.

    G6 판정 범위가 이것이어야 하는 이유: ρ / ρ_worst / satisfies / phase_of 가
    전부 이 열들만 읽는다. 리플레이가 이 열에서 드리프트하면 라벨이 오염되고,
    다른 열이 흔들리는 것은 (라벨에 안 들어가므로) 상대적으로 무해하다.
    """
    m = np.zeros(fs.N_FEATURES, bool)
    if sg is None:
        return m
    for k in range(sg.K):
        for n in sg.spec["phases"][str(k)]["features"]:
            m[fs.index_of(n)] = True
    return m


def rho_drift_effect(data, sg, per_axis=None):
    """phase 별 (drift_ρ, effect_ρ, n0, n1) — ρ_worst 공간에서 재는 리플레이 오차.

    per_axis: (adim,) bool 또는 None. 주면 λ>0 행 중 δ 가 그 축들에 유의미한
    성분을 가진 행만으로 effect 를 잰다 (분모 희석 방지).
    """
    m0 = data["lam"] == 0.0
    C, Y, Z = data["C"], data["Y"], data["Z"].astype(int)
    out = {}
    for k in range(sg.K):
        s0, s1 = m0 & (Z == k), (~m0) & (Z == k)
        if s0.sum() == 0 or s1.sum() == 0:
            continue

        def rms(sel):
            base = np.array([sg.rho_worst(k, c) for c in C[sel]])
            moved = np.array([sg.rho_worst(k, c + y)
                              for c, y in zip(C[sel], Y[sel])])
            return float(np.sqrt(((moved - base) ** 2).mean()))
        out[k] = (rms(s0), rms(s1), int(s0.sum()), int(s1.sum()))
    return out


def gate_g6(data, tol_ratio=0.3, max_anchor_drop=0.5, log=print, sg=None,
            rho_tol=0.3):
    """G6: 분기·리플레이가 믿을 만한가.

    판정 기준은 **ρ_worst 공간의 phase 별 비율**이다 (sg 를 주면):

        drift_ρ(k) = RMS | ρ_worst(φ + Y_λ=0) − ρ_worst(φ) |      리플레이 오차
        effect_ρ(k) = RMS | ρ_worst(φ + Y_λ>0) − ρ_worst(φ) |     섭동 효과
        모든 phase 에서 drift_ρ / effect_ρ ≤ rho_tol

    왜 raw feature 단위 비율이 아닌가 (실측으로 확인된 두 가지 결함):

      (1) 분모 희석. effect 를 λ>0 행 전체로 평균하면 그 채널을 아예 건드리지
          않는 방향까지 들어간다. 실측: gripper_open 의 |Y| 는 δ 에 grip 성분이
          있는 행에서 0.114, 없는 행에서 0.024 (4.7배). contact 은 0.063 vs
          0.0017 (36배). 그래서 per-feature 비율이 0.63 / 0.51 로 부풀었다.
      (2) 채널별 중요도 미반영. ρ_worst 는 조건별 스케일(eq=0.5)과
          d_start_cond 로 정규화하므로, raw 단위 드리프트가 같아도 ρ 에 미치는
          영향은 채널마다 다르다. 게이트가 지켜야 하는 것은 "Stage 7 이
          최적화하는 양이 리플레이 오차에 오염되지 않는가"이고 그것이 ρ 다.

    실측(5편)에서 이 기준은 실패를 phase 0 하나로 국소화했다:
        phase0 0.931 (FAIL)  /  phase1 0.001  /  phase2 0.085
    per-feature 기준은 같은 데이터에서 전체를 FAIL 로 만들어, 어느 phase 의
    데이터가 못 쓸 것인지 알려주지 못했다.

    raw per-feature 비율은 진단으로 계속 출력한다. sg 가 없으면 (mock 등)
    예전 per-feature PROGRESS 판정으로 폴백한다.

    두 번째 조건은 그대로: anchor 드롭률 ≤ max_anchor_drop.
    """
    m0 = data["lam"] == 0.0
    if not m0.any():
        log("[G6] FAIL: λ=0 행이 없음")
        return False, np.full(fs.N_FEATURES, np.nan)
    scale = data["C"].std(axis=0)
    drift = np.sqrt((data["Y"][m0] ** 2).mean(axis=0))
    effect = np.sqrt((data["Y"][~m0] ** 2).mean(axis=0))
    active = effect > (1e-9 + 0.01 * scale)
    ratio = np.where(active, drift / np.maximum(effect, 1e-12), np.nan)
    drop = float(data.get("anchor_drop_rate", 0.0))
    drop_ok = drop <= max_anchor_drop

    gated = subgoal_channels(sg)
    if not gated.any():
        # 폴백: 구 per-feature PROGRESS 판정
        gated = np.array([fs.SPEC[n].kind == fs.PROGRESS for n in fs.NAMES])
        r_gate = np.where(gated, ratio, np.nan)
        if not np.isfinite(r_gate).any():
            log("[G6] FAIL: 판정 대상에 유효한 비율이 없음")
            return False, ratio
        worst = float(np.nanmax(r_gate))
        j = int(np.nanargmax(r_gate))
        ok = (worst <= tol_ratio) and drop_ok
        log(f"[G6] {'PASS' if ok else 'FAIL'}: λ=0 drift/effect (progress 채널 "
            f"(subgoal 미제공 폴백)) 최대 {worst:.3f} ({fs.NAMES[j]}) "
            f"{'<=' if worst <= tol_ratio else '>'} {tol_ratio}, "
            f"anchor drop {drop:.0%} "
            f"{'<=' if drop_ok else '>'} {max_anchor_drop:.0%}")
        log("     판정 대상: " + ", ".join(
            f"{fs.NAMES[k]}={ratio[k]:.3f}" for k in range(fs.N_FEATURES)
            if gated[k] and np.isfinite(ratio[k])))
        return ok, ratio

    # ---- 본 판정: ρ_worst 공간, phase 별 -----------------------------------
    per = rho_drift_effect(data, sg)
    if not per:
        log("[G6] FAIL: phase 별 ρ 비율을 잴 수 있는 행이 없음")
        return False, ratio
    rr = {k: (d / max(e, 1e-12)) for k, (d, e, _, _) in per.items()}
    worst_k = max(rr, key=lambda k: rr[k])
    ok = all(v <= rho_tol for v in rr.values()) and drop_ok
    log(f"[G6] {'PASS' if ok else 'FAIL'}: ρ_worst 공간 drift/effect "
        f"최대 {rr[worst_k]:.3f} (phase {worst_k} [{sg.labels[worst_k]}]) "
        f"{'<=' if rr[worst_k] <= rho_tol else '>'} {rho_tol}, "
        f"anchor drop {drop:.0%} {'<=' if drop_ok else '>'} "
        f"{max_anchor_drop:.0%}")
    for k in sorted(per):
        d, e, n0, n1 = per[k]
        mark = "  <-- 이 phase 데이터는 못 쓴다" if rr[k] > rho_tol else ""
        log(f"     phase {k} [{sg.labels[k]:<5}] drift_ρ={d:.4f} "
            f"effect_ρ={e:.4f} ratio={rr[k]:.3f} "
            f"(λ=0 {n0}행 / λ>0 {n1}행){mark}")
    log("     [진단] raw feature 비율 (분모가 δ 무관 방향까지 포함해 희석됨): "
        + ", ".join(f"{fs.NAMES[k]}={ratio[k]:.3f}"
                    for k in range(fs.N_FEATURES)
                    if gated[k] and np.isfinite(ratio[k])))
    return ok, ratio


# ===========================================================================
# SECTION 3 — FCM 앙상블
# ===========================================================================
class FCM:
    """단일 net: (φ, a, δ, h, obs, phase onehot, goal) -> residual (F,)."""

    def __init__(self, hidden=(96, 96), max_iter=1200, seed=0):
        self.xs, self.ys = StandardScaler(), StandardScaler()
        self.net = MLPRegressor(hidden_layer_sizes=tuple(hidden),
                                max_iter=max_iter, random_state=seed,
                                early_stopping=True, n_iter_no_change=25,
                                validation_fraction=0.1)
        # NOTE: reward_input_mask 는 원래 R_θ 용이다 (action_magnitude/energy 를
        # 빼는 이유는 "λ 의 대리변수라 ranking loss 를 자기 자신으로 푼다"이고
        # 그것은 Stage 10 의 문제다). FCM 입력에는 그 논거가 적용되지 않지만,
        # 두 채널은 여기서도 δ 로부터 거의 결정론적이라 잔차 예측에 정보를 주지
        # 않는다. 마스크를 분리해 둔다 — 의미를 섞지 않기 위해서다.
        self.in_mask = fcm_input_mask()
        self.goal = None
        self.has_goal = False      # 학습 시 goal 열에 실제 값이 들어갔는가

    def _x(self, C, A, D, h, O, Z, g=None):
        C = np.atleast_2d(np.asarray(C, float))[:, self.in_mask]
        A = np.atleast_2d(np.asarray(A, float))
        D = np.atleast_2d(np.asarray(D, float))
        O = np.atleast_2d(np.asarray(O, float))
        n = max(len(C), len(A), len(D), len(O))

        def rep(M):
            return np.repeat(M, n, axis=0) if len(M) == 1 and n > 1 else M
        C, A, D, O = rep(C), rep(A), rep(D), rep(O)
        h = np.atleast_1d(np.asarray(h, float)).reshape(-1, 1)
        if len(h) == 1 and n > 1:
            h = np.repeat(h, n, axis=0)
        zz = np.asarray(Z, int).ravel()
        if len(zz) == 1 and n > 1:
            zz = np.repeat(zz, n)
        oh = np.zeros((n, N_PHASE_SLOTS))
        oh[np.arange(n), np.clip(zz, 0, N_PHASE_SLOTS - 1)] = 1.0
        # goal: 행별 (n,3) 또는 단일 (3,). None 이면 0 (구 동작).
        gg = self.goal if g is None else g
        if gg is None:
            G = np.zeros((n, 3))
        else:
            G = np.atleast_2d(np.asarray(gg, float))
            if len(G) == 1 and n > 1:
                G = np.repeat(G, n, axis=0)
            if len(G) != n:
                raise ValueError(f"goal rows {len(G)} != batch {n}")
        return np.hstack([C, A, D, h, O, oh, G])

    def fit(self, data, goal=None):
        """goal 은 data["G"] (행별) 를 우선 쓰고, 없으면 인자 goal (단일)."""
        g = data.get("G") if isinstance(data, dict) else None
        if g is None and goal is not None:
            g = np.asarray(goal, float).ravel()
        self.goal = None if g is None else np.asarray(g, float)
        self.has_goal = g is not None
        X = self._x(data["C"], data["A"], data["D"], data["h"],
                    data["O"], data["Z"], g=g)
        self.net.fit(self.xs.fit_transform(X), self.ys.fit_transform(data["Y"]))
        # 배치 전체 goal 을 self.goal 에 남기면 추론 시 행수가 안 맞는다.
        # 학습에 goal 을 썼다는 사실만 남기고 값은 호출자가 넘긴다.
        self.goal = None
        return self

    def predict(self, C, A, D, h, O, Z, g=None):
        X = self._x(C, A, D, h, O, Z, g=g)
        return self.ys.inverse_transform(self.net.predict(self.xs.transform(X)))


class FCMEnsemble:
    """seed 만 다른 FCM n개 — 평균이 예측, 분산이 불확실성 penalty."""

    def __init__(self, n=5, hidden=(96, 96), max_iter=1200):
        self.members = [FCM(hidden=hidden, max_iter=max_iter, seed=s)
                        for s in range(n)]

    def fit(self, data, goal=None, log=print):
        for i, m in enumerate(self.members):
            m.fit(data, goal=goal)
            log(f"  member {i}: fitted (val loss {m.net.best_validation_score_:.4f})"
                if hasattr(m.net, "best_validation_score_") else f"  member {i} fitted")
        return self

    def predict(self, *a, **kw):
        P = np.stack([m.predict(*a, **kw) for m in self.members])
        return P.mean(axis=0), P.std(axis=0)

    @property
    def has_goal(self):
        """멤버 전부가 goal 열을 실제 값으로 학습했는가. 구 pkl 은 False —
        그 경우 screening 에서 goal 을 넘기면 학습 분포와 어긋난다."""
        return all(getattr(m, "has_goal", False) for m in self.members)

    def save(self, path):
        """sklearn 객체만 pickle 한다: 커스텀 클래스를 통째로 담으면
        `python fcm.py` 실행 시 __main__ 네임스페이스로 저장되어 다른
        프로세스가 로드하지 못한다 (실측)."""
        state = dict(members=[dict(xs=m.xs, ys=m.ys, net=m.net, goal=m.goal,
                                   has_goal=bool(m.has_goal))
                              for m in self.members])
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            state = pickle.load(f)
        ens = FCMEnsemble(n=len(state["members"]))
        for m, s in zip(ens.members, state["members"]):
            m.xs, m.ys, m.net, m.goal = s["xs"], s["ys"], s["net"], s["goal"]
            m.has_goal = bool(s.get("has_goal", False))
        return ens


def r2_per_feature(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    out = np.full(y.shape[1], np.nan)
    for j in range(y.shape[1]):
        ss = float(((y[:, j] - y[:, j].mean()) ** 2).sum())
        if ss > 1e-12:
            out[j] = 1.0 - float(((y[:, j] - yhat[:, j]) ** 2).sum()) / ss
    return out


# ===========================================================================
# SECTION 4 — Stage 7: screening  (d* = argmin_d ρ_z(φ + FCM(...)))
# ===========================================================================
def feasible_cone(actions, a_, b_, sat_frac=0.5, tol=1e-6):
    A = np.asarray(actions, float)[a_:b_]
    adim = A.shape[1]
    if len(A) == 0:
        return np.zeros(adim, bool), np.zeros(adim, bool)
    hi = (A >= 1.0 - tol).mean(axis=0) > sat_frac
    lo = (A <= -1.0 + tol).mean(axis=0) > sat_frac
    return hi, lo


def project_to_cone(d, hi_blocked, lo_blocked, tol=1e-12):
    d = np.asarray(d, float).copy()
    d[hi_blocked & (d > 0)] = 0.0
    d[lo_blocked & (d < 0)] = 0.0
    n = np.linalg.norm(d)
    return (d / n) if n > tol else None


def dedupe_directions(named, cos_max=0.95):
    kept = []
    for nm, d in named:
        d = np.asarray(d, float)
        if np.linalg.norm(d) < 1e-12:
            continue
        if any(abs(float(np.dot(d, k))) > cos_max for _, k in kept):
            continue
        kept.append((nm, d))
    return kept


def gripper_reverse_direction(actions, a, b, adim):
    seg = np.asarray(actions, float)[a:b, adim - 1]
    if len(seg) == 0:
        return None
    m = float(np.mean(seg))
    d = np.zeros(adim)
    d[adim - 1] = -np.sign(m) if abs(m) > 1e-9 else 1.0
    return d


def subspace_of(d):
    d = np.asarray(d, float)
    pos = float(np.sum(d[:3] ** 2))
    rot = float(np.sum(d[3:6] ** 2)) if len(d) >= 7 else 0.0
    grp = float(d[-1] ** 2)
    tot = pos + rot + grp + 1e-12
    if pos / tot > 0.7:
        return "position"
    if rot / tot > 0.7:
        return "rotation"
    if grp / tot > 0.7:
        return "gripper"
    return "mixed"


class Screener:
    """앙상블 FCM + rho.Subgoal 로 phase 별 열화 방향을 랭킹한다."""

    def __init__(self, ens, sg, lam_probe, horizon_s, beta=1.0):
        self.ens, self.sg = ens, sg
        self.lam, self.h = float(lam_probe), float(horizon_s)
        self.beta = float(beta)

    def drho(self, phase, phi, act, obs, d, goal=None):
        """anchor 배치에 대한 (Δρ̂ 평균, 불확실성). d 는 단위벡터.

        목적함수는 rho_worst (위반마진·조건별 정규화, rho.py 헤더 참조):
        subgoal 은 조건들의 AND 인데 rho 의 RMS·전역 d_start 정규화는 조건 하나의
        파괴를 √(1/J) 로 희석하고 스케일이 큰 조건이 눈금을 독점한다. 실측
        (PickPlaceBread pre phase): 파지 완전 실패 Δρ=−0.148 vs eef 5cm 떼기
        Δρ=−0.139 — 태스크가 죽는 사건과 손이 조금 떨어진 것이 구별되지 않아
        신호 상한이 앙상블 불확실성(0.01~0.02) 수준으로 눌렸고, 그 결과
        steepest 가 3 phase 전부에서 랜덤 방향에 패배했다.
        rho_worst 는 같은 사건에 Δ=−0.84 vs −0.10 을 준다.
        (클립하지 않으므로 '시작보다 더 나쁨'이 음수로 보이는 성질은 유지된다.)
        """
        delta = self.lam * np.asarray(d, float)
        n = len(phi)
        mu, sd = self.ens.predict(phi, act, np.tile(delta, (n, 1)),
                                  self.h, obs, np.full(n, phase, int), g=goal)
        r0 = self.sg.rho_worst(phase, phi)
        drs, dr_sd = [], []
        for i in range(n):
            drs.append(self.sg.rho_worst(phase, phi[i] + mu[i]) - r0[i])
            hi = self.sg.rho_worst(phase, phi[i] + mu[i] + sd[i])
            lo = self.sg.rho_worst(phase, phi[i] + mu[i] - sd[i])
            dr_sd.append(abs(hi - lo) / 2.0)
        return float(np.mean(drs)), float(np.mean(dr_sd)), np.array(drs)

    def score(self, dr, unc):
        """클수록 좋은 screening_score: 예측 열화 − β·불확실성."""
        return -dr - self.beta * unc

    def steepest(self, phase, phi, act, obs, cone, steps=3, fd=0.15,
                 goal=None):
        """FD 로 ∇_d Δρ̂ 를 얻어 ρ_worst 를 가장 깎는 단위 방향을 고정점 반복.

        rho_worst 의 기본 집계가 p=4 인 이유가 여기 있다: 정확한 max(p→∞)는
        비활성 조건에 대한 기울기가 0 이라 이 FD 가 계단 함수를 보게 된다.
        p=4 는 max 에 가까우면서 미분 가능하다."""
        adim = act.shape[1]
        d = np.zeros(adim)
        for _ in range(max(1, steps)):
            g = np.zeros(adim)
            for i in range(adim):
                dp, dm = d.copy(), d.copy()
                dp[i] += fd; dm[i] -= fd
                np_, _, _ = self.drho(phase, phi, act, obs,
                                      dp / max(np.linalg.norm(dp), 1e-9),
                                      goal=goal)
                nm_, _, _ = self.drho(phase, phi, act, obs,
                                      dm / max(np.linalg.norm(dm), 1e-9),
                                      goal=goal)
                g[i] = (np_ - nm_) / (2 * fd)
            d_new = project_to_cone(-g, *cone)
            if d_new is None:
                return None
            if np.dot(d, d_new) > 1 - 1e-4:
                d = d_new
                break
            d = d_new
        return d


def condition_attack_directions(sg, phase):
    """phase 조건별 공격 부호: le→+1(키움), ge→−1(줄임), eq/box→±둘 다."""
    p = sg.spec["phases"][str(phase)]
    out = []
    for n, typ in zip(p["features"], p["type"]):
        idx = fs.index_of(n)
        if typ == "le":
            out.append((n, idx, +1.0))
        elif typ == "ge":
            out.append((n, idx, -1.0))
        else:
            out.append((n, idx, +1.0)); out.append((n, idx, -1.0))
    return out


def screen_phase(scr, phase, phi, act, obs, actions_seg, rng, args, log=print,
                 goal=None):
    """한 phase 의 후보 생성 → 점수화 → 다양성 Top-K."""
    adim = act.shape[1]
    a_, b_ = 0, len(actions_seg)
    cone = feasible_cone(actions_seg, a_, b_, sat_frac=args.sat_frac)
    hi, lo = cone
    if hi.any() or lo.any():
        blk = [AXIS_LABELS[i] + ("+" if hi[i] else "") + ("-" if lo[i] else "")
               for i in range(adim) if hi[i] or lo[i]]
        log(f"    cone blocked: {', '.join(blk)}")

    cand = []
    d_st = scr.steepest(phase, phi, act, obs, cone, steps=args.refine_steps,
                        goal=goal)
    if d_st is not None:
        cand.append(("steepest", d_st))
    # 조건 공격: FCM 야코비안 없이 좌표축이 아닌 "그 feature 를 미는" 방향을
    # steepest 의 FD 부호로 얻기는 비싸다 — 대신 축/랜덤/데모평행/그립 후보를
    # 깔고 점수가 고르게 한다 (steepest 가 조건 결합 방향을 담당).
    for i in range(adim):
        for s in (+1.0, -1.0):
            d = np.zeros(adim); d[i] = s
            p = project_to_cone(d, *cone)
            if p is not None:
                cand.append((f"axis[{AXIS_LABELS[i]}{'+' if s > 0 else '-'}]", p))
    g = gripper_reverse_direction(actions_seg, a_, b_, adim)
    if g is not None:
        p = project_to_cone(g, *cone)
        if p is not None:
            cand.append(("grip_rev", p))
    mean_a = actions_seg.mean(axis=0)
    if np.linalg.norm(mean_a) > 1e-9:
        for s, nm in ((+1, "para"), (-1, "antipara")):
            p = project_to_cone(s * mean_a, *cone)
            if p is not None:
                cand.append((nm, p))
    for i in range(args.rand_dirs):
        p = project_to_cone(rng.normal(size=adim), *cone)
        if p is not None:
            cand.append((f"rand{i}", p))
    cand = dedupe_directions(cand, args.cos_max)

    scored = []
    for nm, d in cand:
        dr, unc, per_anchor = scr.drho(phase, phi, act, obs, d, goal=goal)
        scored.append(dict(name=nm, d=d, drho=dr, unc=unc,
                           score=scr.score(dr, unc), per_anchor=per_anchor))
    scored.sort(key=lambda r: -r["score"])

    # 점수 하한. 이것이 없으면 Top-K 를 채우려고 score ≤ 0 인 후보 — 즉
    # "불확실성을 감안하면 열화시킬 것으로 예측되지 않는 방향" — 까지 실려
    # 나간다 (실측: action_sets.json 12개 중 3개가 음수 점수였고, Stage 8 의
    # 성분 합산이 그것들을 그대로 주입 방향에 섞었다).
    min_score = float(getattr(args, "min_score", 0.0))
    picked, n_below = [], 0
    for r in scored:
        if len(picked) >= args.top_k:
            break
        if r["score"] <= min_score:
            n_below += 1
            continue
        if any(abs(float(np.dot(r["d"], p["d"]))) > args.cos_max
               for p in picked):
            continue
        picked.append(r)
    for r in picked:
        log(f"    {r['name']:<16} drho={r['drho']:+.3f}  unc={r['unc']:.3f}  "
            f"score={r['score']:+.3f}")
    if n_below:
        log(f"    (score <= {min_score:g} 로 제외 {n_below}개)")
    if len(picked) < args.top_k:
        log(f"    [warn] phase {phase}: 후보 {len(picked)}/{args.top_k} 개만 "
            f"기준 통과 — 열화 방향이 부족하다 (FCM 재학습/λ_probe 상향 검토)")
    return picked, scored


def injection_window(per_anchor, anchor_fracs, floor=0.5):
    """anchor 별 예측 열화에서 주입 창(start/duration fraction)을 읽는다:
    |Δρ̂| ≥ floor·max 인 anchor 들의 span. 평평하면 phase 전체."""
    v = np.abs(np.asarray(per_anchor, float))
    if v.max() < 1e-9:
        return 0.0, 1.0
    m = v >= floor * v.max()
    f = np.asarray(anchor_fracs, float)[m]
    start = float(max(0.0, f.min() - 0.05))
    end = float(min(1.0, f.max() + 0.05))
    return start, max(0.05, end - start)


# ===========================================================================
# SECTION 5 — 실행기 (collect / train / screen) + G7
# ===========================================================================
def _load_pipeline(cache_dir, boundaries_path, subgoal_path):
    from extract_features import load_cache
    entries = load_cache(cache_dir)
    with open(boundaries_path) as f:
        boundaries = json.load(f)
    sg = Subgoal.load(subgoal_path)
    return entries, boundaries, sg


def build_rollers(entries, warmup=5, log=print):
    """cache 의 meta.source 로 원본 hdf5 를 찾아 env + DemoRollout 구성."""
    import frame_extract as fx
    rollers, env, key = {}, None, None
    for e in entries:
        src = e["meta"]["source"]
        env_info, demos = fx.read_demo(src)
        name = e["meta"].get("source_demo") or demos[0][0]
        rec = next(d for d in demos if d[0] == name)
        _, states, actions, xml, tq = rec
        k = (env_info["env_name"], str(env_info["robots"]))
        if k != key:
            if env is not None:
                env.close()
            log(f"[env] {k[0]} / {k[1]}")
            env = fx.build_env(env_info)
            key = k
        fx.reset_to_scene(env, xml)
        ex = fx.FrameExtractor(env, env_info.get("object_type"), dt=e["dt"])
        fr = ex.from_states(states, actions, tq, torque_mode="none")
        rollers[e["demo_id"]] = DemoRollout(
            env, states, actions, fr, e["dt"],
            object_type=env_info.get("object_type"), warmup=warmup)
    return rollers, env


def save_dataset(path, data):
    import h5py
    with h5py.File(path, "w") as f:
        for k, v in data.items():
            f.create_dataset(k, data=v)


def load_dataset(path):
    import h5py
    with h5py.File(path, "r") as f:
        return {k: f[k][()] for k in f}


def cmd_collect(args):
    entries, boundaries, sg = _load_pipeline(args.cache_dir, args.boundaries,
                                             args.subgoal)
    rng = np.random.default_rng(args.seed)
    scales = np.concatenate([e["F"] for e in entries], axis=0).std(axis=0)
    rollers, env = build_rollers(entries, warmup=args.warmup)
    try:
        data = collect(rollers, boundaries, sg, args, rng, scales=scales)
    finally:
        if env is not None:
            env.close()
    ok, ratio = gate_g6(data, tol_ratio=args.g6_tol, sg=sg,
                        rho_tol=args.g6_rho_tol)
    os.makedirs(args.out_dir, exist_ok=True)
    save_dataset(os.path.join(args.out_dir, "fcm_dataset.hdf5"), data)
    viz_g6(data, ratio, os.path.join(args.out_dir, "fcm_g6_residual.png"))
    print(f"[out] {os.path.join(args.out_dir, 'fcm_dataset.hdf5')}")
    if not ok and not args.no_gate:
        raise SystemExit("[G6] FAIL — 분기/리플레이 수정 전 학습 금지")
    return data


def cmd_train(args, data=None):
    if data is None:
        data = load_dataset(os.path.join(args.out_dir, "fcm_dataset.hdf5"))
    try:
        sg = Subgoal.load(args.subgoal)
    except Exception:
        sg = None
    ok, _ = gate_g6(data, tol_ratio=args.g6_tol, sg=sg,
                    rho_tol=args.g6_rho_tol)
    if not ok and not args.no_gate:
        raise SystemExit("[G6] FAIL — 학습 중단")
    # holdout: 시간 순 20% (무작위 셔플은 같은 rollout 의 h 행이 양쪽에 새는
    # 낙관 편향을 만든다 — rollout 단위로 나눠야 정직한 R^2)
    key = (np.asarray(data["ep"]).astype(str) if "ep" in data else
           np.zeros(len(data["Y"]), dtype=str))
    uniq = sorted(set(zip(key.tolist(), data["t"].astype(int).tolist(),
                          [tuple(np.round(d, 6)) for d in data["D"]])))
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    n_te = max(1, len(uniq) // 5)
    test_keys = set(uniq[:n_te])
    rid = list(zip(key.tolist(), data["t"].astype(int).tolist(),
                   [tuple(np.round(d, 6)) for d in data["D"]]))
    m_te = np.array([r in test_keys for r in rid])
    # 학습 라벨: λ=0 행은 정의상 0 (Y 에 담긴 드리프트는 G6 진단용이지
    # residual 라벨이 아니다) — f(δ=0)=0 앵커로만 쓴다.
    Yc = data["Y"].copy()
    Yc[data["lam"] == 0.0] = 0.0
    dd = dict(data); dd["Y"] = Yc
    keep = ("C", "O", "A", "D", "h", "Z", "Y", "G")
    tr = {k: v[~m_te] for k, v in dd.items() if k in keep}
    te = {k: v[m_te] for k, v in dd.items() if k in keep}
    if "G" not in tr:
        print("[warn] 데이터셋에 goal 열(G)이 없다 — 구 fcm_dataset.hdf5 다. "
              "goal 입력 3열이 0 으로 학습된다 (재수집 권장).")

    ens = FCMEnsemble(n=args.ensemble, hidden=tuple(args.hidden),
                      max_iter=args.max_iter)
    ens.fit(tr)
    mu, sd = ens.predict(te["C"], te["A"], te["D"], te["h"], te["O"], te["Z"],
                         g=te.get("G"))
    r2 = r2_per_feature(te["Y"], mu)
    print("[train] heldout R^2 (rollout 단위 분할):")
    for n, v in zip(fs.NAMES, r2):
        print(f"    {n:<18} {v:+.3f}" if np.isfinite(v) else
              f"    {n:<18}   n/a")
    ens.save(os.path.join(args.out_dir, "fcm_ensemble.pkl"))
    viz_fit(te["Y"], mu, r2, os.path.join(args.out_dir, "fcm_fit_r2.png"))
    print(f"[out] {os.path.join(args.out_dir, 'fcm_ensemble.pkl')}")
    return ens, (te, mu, sd)


def cmd_screen(args, ens=None):
    entries, boundaries, sg = _load_pipeline(args.cache_dir, args.boundaries,
                                             args.subgoal)
    if ens is None:
        ens = FCMEnsemble.load(os.path.join(args.out_dir, "fcm_ensemble.pkl"))
    rng = np.random.default_rng(args.seed + 1)
    dt = entries[0]["dt"]
    scr = Screener(ens, sg, args.lam_probe, args.horizon * dt, beta=args.beta)

    K = len(sg.labels)
    sets, all_scored = {}, {}
    for phase in range(K):
        log = print
        log(f"  phase {phase} [{sg.labels[phase]}]")
        phi, act, obs, segs, fracs, goals = [], [], [], [], [], []
        for e in entries:
            did = e["demo_id"]
            T = len(e["F"])
            zseg = z_from_bounds(boundaries[did]["bounds"], T)
            anchors = phase_anchors(zseg, phase, args.screen_anchors, T,
                                    args.horizon)
            span = np.where(zseg == phase)[0]
            if not anchors or not len(span):
                continue
            a_, b_ = span[0], span[-1] + 1
            for t in anchors:
                phi.append(e["F"][t])
                act.append(e["actions"][t])
                obs.append(np.concatenate([e["eef_pos"][t], e["obj_pos"][t]])
                           if e.get("eef_pos") is not None
                           else np.zeros(N_OBS))
                fracs.append((t - a_) / max(1, b_ - a_))
                goals.append(np.asarray(e["goal"], float).ravel()[:3])
            segs.append(e["actions"][a_:b_])
        if not phi:
            continue
        phi, act, obs = np.array(phi), np.array(act), np.array(obs)
        # goal 은 앙상블이 goal 로 학습된 경우에만 넘긴다. 구 pkl 은 goal 열이
        # 0 으로 학습됐으므로 실제 값을 주면 학습 분포와 어긋난다.
        g_arg = np.array(goals) if ens.has_goal else None
        actions_seg = np.concatenate(segs, axis=0)
        picked, scored = screen_phase(scr, phase, phi, act, obs, actions_seg,
                                      rng, args, goal=g_arg)
        out = []
        for i, r in enumerate(picked):
            s0, dur = injection_window(r["per_anchor"], fracs)
            out.append(dict(
                candidate_id=f"p{phase}_c{i:03d}",
                name=r["name"],
                direction=[round(float(x), 6) for x in r["d"]],
                subspace=subspace_of(r["d"]),
                start_fraction=round(s0, 3),
                duration_fraction=round(dur, 3),
                predicted_drho=round(r["drho"], 4),
                uncertainty=round(r["unc"], 4),
                screening_score=round(r["score"], 4)))
        sets[f"phase_{phase}"] = out
        all_scored[phase] = scored

    path = os.path.join(args.out_dir, "action_sets.json")
    with open(path, "w") as f:
        json.dump(sets, f, indent=2)
    print(f"[out] {path}")
    viz_sets(sets, os.path.join(args.out_dir, "fcm_sets.png"), sg)
    return sets, all_scored, (entries, boundaries, sg)


def gate_g7(args, sets, all_scored, pipeline, log=print, out_path=None):
    """Top-K recall > random: FCM 랭킹 상위권이 실측(분기 rollout) 상위권과
    겹치는가. 실측 Δρ = 같은 anchor 에서 λ_probe·d 를 실제로 밀어본 결과.

    검증 대상은 action_sets.json 에 실제로 실린 방향(picked)이다. 이전 구현은
    fcm_top = range(top_k), 즉 scored[:top_k] 를 FCM 의 Top-K 로 봤는데, 출하되는
    것은 다양성 필터와 점수 하한을 통과한 picked 이므로 게이트가 파이프라인이
    쓰지 않는 집합을 검증하고 있었다.

    실측 눈금도 screening 과 같은 rho_worst 를 쓴다 — 다른 눈금으로 재면
    recall 이 목적함수 불일치를 재게 된다.
    """
    entries, boundaries, sg = pipeline
    rollers, env = build_rollers(entries, warmup=args.warmup)
    rng = np.random.default_rng(args.seed + 2)
    recalls, randoms, diag = [], [], {}
    try:
        for phase, scored in all_scored.items():
            pool = scored[:args.g7_pool]           # FCM 랭킹 순 상위 pool
            if len(pool) < args.top_k + 2:
                log(f"  phase {phase}: pool {len(pool)} < top_k+2 — SKIP")
                continue
            shipped = {c["name"] for c in sets.get(f"phase_{phase}", [])}
            # Branch A(리플레이) 기준으로 실측 — 학습 라벨과 같은 기준선
            baseA = {}
            for did, roller in list(rollers.items())[:args.g7_demos]:
                T = roller.T
                zseg = z_from_bounds(boundaries[did]["bounds"], T)
                for t in phase_anchors(zseg, phase, args.g7_anchors, T,
                                       args.horizon):
                    outA, _ = roller.branch(t, np.zeros(roller.adim),
                                            args.horizon)
                    if outA is not None:
                        baseA[(did, t)] = outA
            meas = []
            for r in pool:
                drs = []
                for (did, t), outA in baseA.items():
                    roller = rollers[did]
                    out, info = roller.branch(
                        t, args.lam_probe * np.asarray(r["d"]), args.horizon)
                    if out is None:
                        continue
                    phi_t = roller.phi_at(t)
                    r_end = sg.rho_worst(phase, phi_t + (out[-1] - outA[-1]))
                    r_ref = sg.rho_worst(phase, phi_t)
                    drs.append(r_end - r_ref)
                meas.append(float(np.mean(drs)) if drs else np.nan)
            meas = np.asarray(meas)
            okm = np.isfinite(meas)
            if okm.sum() < args.top_k + 2:
                log(f"  phase {phase}: 측정 성공 {int(okm.sum())} < top_k+2 — SKIP")
                continue
            idx_ok = np.where(okm)[0]
            order_true = np.argsort(meas[okm])          # 가장 음수 = 최고 열화
            k_eff = min(args.top_k, int(okm.sum()) - 1)
            true_top = set(idx_ok[order_true[:k_eff]].tolist())
            # 실제 출하된 방향(picked)의 pool 내 위치
            fcm_top = {i for i, r in enumerate(pool)
                       if r["name"] in shipped and okm[i]}
            if not fcm_top:
                log(f"  phase {phase}: 출하 방향이 pool/측정 안에 없음 — SKIP")
                continue
            rec = len(true_top & fcm_top) / max(1, len(fcm_top))
            rnd = k_eff / int(okm.sum())
            recalls.append(rec); randoms.append(rnd)
            diag[f"phase_{phase}"] = dict(
                recall=round(rec, 3), random=round(rnd, 3),
                n_pool=int(okm.sum()), n_shipped=len(fcm_top), k_eff=k_eff,
                measured_drho_worst={pool[i]["name"]: round(float(meas[i]), 4)
                                     for i in idx_ok},
                shipped=sorted(shipped))
            log(f"  phase {phase}: recall = {rec:.2f} "
                f"(random {rnd:.2f}; pool {int(okm.sum())}, "
                f"출하 {len(fcm_top)}개)")
    finally:
        if env is not None:
            env.close()
    if not recalls:
        log("[G7] SKIP: 측정 가능한 phase 없음")
        if out_path:
            with open(out_path, "w") as f:
                json.dump(dict(g7_pass=None, phases=diag), f,
                          ensure_ascii=False, indent=1)
        return None
    ok = float(np.mean(recalls)) > float(np.mean(randoms))
    log(f"[G7] {'PASS' if ok else 'FAIL'}: mean recall {np.mean(recalls):.2f} "
        f"vs random {np.mean(randoms):.2f}")
    if out_path:
        with open(out_path, "w") as f:
            json.dump(dict(g7_pass=bool(ok),
                           mean_recall=round(float(np.mean(recalls)), 3),
                           mean_random=round(float(np.mean(randoms)), 3),
                           phases=diag), f, ensure_ascii=False, indent=1)
        log(f"[out] {out_path}")
    return ok


# ===========================================================================
# SECTION 6 — 시각화
# ===========================================================================
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def viz_g6(data, ratio, out_png):
    plt = _plt()
    m0 = data["lam"] == 0.0
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    r0 = np.abs(data["Y"][m0]).mean(axis=0)
    r1 = np.abs(data["Y"][~m0]).mean(axis=0)
    x = np.arange(fs.N_FEATURES)
    axes[0].bar(x - 0.2, r0, 0.4, label="lam=0 drift (replay vs demo)")
    axes[0].bar(x + 0.2, r1, 0.4, label="lam>0 effect (pert vs replay)")
    axes[0].set_xticks(x); axes[0].set_xticklabels(fs.NAMES, rotation=90,
                                                  fontsize=7)
    axes[0].set_yscale("log"); axes[0].set_ylabel("mean |residual|")
    axes[0].legend(fontsize=8)
    axes[0].set_title("G6: lam=0 must leave ~no residual")
    axes[1].bar(x, ratio)
    axes[1].axhline(0.3, color="r", ls="--", label="tol 0.3")
    axes[1].set_xticks(x); axes[1].set_xticklabels(fs.NAMES, rotation=90,
                                                  fontsize=7)
    axes[1].set_ylabel("drift RMS / effect RMS"); axes[1].legend()
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    print(f"[plot] {out_png}")


def viz_fit(Y, Yhat, r2, out_png):
    plt = _plt()
    fig = plt.figure(figsize=(13, 8))
    show = [n for n in ("eef_object_dist", "object_goal_dist", "contact",
                        "object_height", "eef_speed", "object_slip")
            if n in fs.NAMES]
    for i, n in enumerate(show):
        ax = fig.add_subplot(2, 3, i + 1)
        j = fs.index_of(n)
        ax.scatter(Y[:, j], Yhat[:, j], s=4, alpha=0.4)
        lim = np.percentile(np.abs(Y[:, j]), 99) + 1e-6
        ax.plot([-lim, lim], [-lim, lim], "r--", lw=1)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_title(f"{n}  R²={r2[j]:+.2f}", fontsize=9)
        ax.set_xlabel("measured residual"); ax.set_ylabel("predicted")
    fig.suptitle("FCM heldout fit (rollout-split)")
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    print(f"[plot] {out_png}")


def viz_sets(sets, out_png, sg):
    plt = _plt()
    K = len(sets)
    fig, axes = plt.subplots(1, max(K, 1), figsize=(4.2 * max(K, 1), 4.4),
                             squeeze=False)
    im = None
    for ax, (pk, rows) in zip(axes[0], sorted(sets.items())):
        if not rows:
            continue
        D = np.array([r["direction"] for r in rows])
        im = ax.imshow(D, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(D.shape[1]))
        ax.set_xticklabels(AXIS_LABELS[:D.shape[1]], fontsize=7)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{r['candidate_id']}  dρ̂={r['predicted_drho']:+.2f}"
                            for r in rows], fontsize=7)
        z = int(pk.split("_")[1])
        ax.set_title(f"{pk} [{sg.labels[z]}]", fontsize=9)
    if im is not None:
        fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8)
    fig.suptitle("action_sets: per-phase Top-K anti-subgoal directions")
    fig.savefig(out_png, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] {out_png}")


# ===========================================================================
# SECTION 7 — selftest (numpy 만: mock 물리로 collect→G6→train→screen→G7)
# ===========================================================================
class _MockRoller:
    """선형 mock 물리: residual = h·M δ (+noise). eef_object_dist 는 δ[:3] 의
    +x 에, object_goal_dist 는 −y 에 민감하도록 M 을 짠다 — screening 이
    이 방향을 되찾는지 본다."""

    def __init__(self, T=120, adim=7, seed=0, dt=0.05):
        rng = np.random.default_rng(seed)
        self.T, self.adim, self.dt = T, adim, dt
        self.actions = np.clip(rng.normal(0, 0.3, (T, adim)), -1, 1)
        self.goal = np.array([0.5, 0.2, 0.9])
        self.obj0_z = 0.85
        self.M = np.zeros((fs.N_FEATURES, adim))
        self.M[fs.index_of("eef_object_dist"), 0] = +0.08
        self.M[fs.index_of("object_goal_dist"), 1] = -0.06
        self.M[fs.index_of("contact"), 6] = -0.4
        self.rng = rng
        base = np.linspace(1, 0, T)
        self.F0 = np.tile(base[:, None], (1, fs.N_FEATURES)) * 0.3

    def demo_window_features(self, t, H):
        return np.stack([self.F0[min(t + h + 1, self.T - 1)]
                         for h in range(H)])

    def obs_row(self, t):
        return np.zeros(N_OBS)

    def phi_at(self, t):
        return self.F0[t]

    def branch(self, t, delta, H):
        base = self.demo_window_features(t, H)
        out = base.copy()
        for h in range(H):
            out[h] += (h + 1) * self.dt * (self.M @ delta)
            out[h] += self.rng.normal(0, 1e-4, fs.N_FEATURES)
        # 실제 DemoRollout 과 같은 계약: 분기 시작점의 실측 feature 행.
        # mock 은 리플레이 오차가 없으므로 데모 φ(t) 와 같다.
        return out, dict(clip=0.0, ok=True, phi0=self.F0[t].copy())


def _mock_subgoal():
    return Subgoal(dict(
        meta=dict(canonical_labels=["pre", "move", "post"], alpha=1.0,
                  sd_floor=0.001),
        phases={
            "0": dict(label="pre", features=["eef_object_dist"],
                      mean=[0.0], std=[0.005], global_std=[0.1],
                      type=["le"], thresh=[0.01], persistent=[True],
                      d_start=6.0, v=[0]),
            "1": dict(label="move", features=["object_goal_dist"],
                      mean=[0.0], std=[0.005], global_std=[0.1],
                      type=["le"], thresh=[0.01], persistent=[True],
                      d_start=6.0, v=[0]),
            "2": dict(label="post", features=["contact"],
                      mean=[0.0], std=[0.0], global_std=[0.5],
                      type=["eq"], thresh=[0.0], persistent=[True],
                      d_start=2.0, v=[0]),
        }))


def run_selftest():
    print("=== fcm SELFTEST (mock physics; no robosuite) ===")
    if not _HAS_SK:
        print("[selftest] SKIP: scikit-learn 없음")
        return False
    ok = True
    rng = np.random.default_rng(0)
    sg = _mock_subgoal()
    roller = _MockRoller()
    rollers = {"demo_000": roller}
    boundaries = {"demo_000": dict(bounds=[40, 80],
                                   labels=["pre", "move", "post"])}

    class A:                                       # collect args
        anchors, horizon, levels = 3, 6, 3
        delta_scale, dirs_per_level, max_clip = 0.3, 4, 0.6
        anchor_drift_tol = 0.3
        anchor_rho_tol = 0.15
        min_score = 0.0
    data = collect(rollers, boundaries, sg, A, rng,
                   scales=roller.F0.std(axis=0), log=lambda *a: None)
    print(f"collected {len(data['Y'])} rows "
          f"(λ=0 {int((data['lam'] == 0).sum())})")

    # G6: subgoal 조건 채널 범위로 판정 (sg 전달). mock 의 조건 채널은
    # eef_object_dist / object_goal_dist / contact 셋이다.
    g6, _ = gate_g6(data, sg=sg)
    if not g6:
        ok = False
    gated = subgoal_channels(sg)
    want = {"eef_object_dist", "object_goal_dist", "contact"}
    got = {fs.NAMES[i] for i in np.where(gated)[0]}
    if got != want:
        ok = False; print(f"[FAIL] G6 판정 범위 {got} != {want}")
    else:
        print(f"G6 진단 범위 = subgoal 조건 채널 {sorted(got)} "
              f"(kind 기반 PROGRESS 한정이 아님)")

    # ρ 공간 판정이 데이터에 존재하는 phase 전부를 덮는가 (본판정 경로).
    # mock 의 F0 는 전 구간에서 subgoal 조건 미달이라 phase_of 가 0 만 낸다 —
    # zseg(경계 기반)와 Z(조건 기반)가 다른 것은 의도된 성질이므로, 기준은
    # "Z 에 실제로 있는 phase 를 빠짐없이 판정하는가"다.
    per = rho_drift_effect(data, sg)
    zs = set(np.unique(data["Z"]).astype(int).tolist())
    if set(per) != zs:
        ok = False
        print(f"[FAIL] ρ 공간 판정 phase {sorted(per)} != 데이터의 {sorted(zs)}")
    else:
        rr = {k: per[k][0] / max(per[k][1], 1e-12) for k in per}
        print(f"ρ 공간 판정 phase {sorted(per)} (Z 전부 커버), ratio = "
              + ", ".join(f"p{k}:{v:.3f}" for k, v in sorted(rr.items())))
        # mock 은 리플레이 오차가 없으므로 drift_ρ ≈ 0 이어야 한다
        if max(rr.values()) > 0.3:
            ok = False
            print(f"[FAIL] mock 인데 ρ drift 가 큼: {rr}")
    if "G" not in data:
        ok = False; print("[FAIL] collect 가 goal 열(G)을 담지 않음")
    elif data["G"].shape != (len(data["Y"]), 3):
        ok = False; print(f"[FAIL] G shape {data['G'].shape}")

    ens = FCMEnsemble(n=3, hidden=(48,), max_iter=400)
    ens.fit(data, log=lambda *a: None)
    mu, sd = ens.predict(data["C"], data["A"], data["D"], data["h"],
                         data["O"], data["Z"])
    r2 = r2_per_feature(data["Y"], mu)
    j = fs.index_of("eef_object_dist")
    print(f"train R² eef_object_dist={r2[j]:+.2f}, "
          f"object_goal_dist={r2[fs.index_of('object_goal_dist')]:+.2f}")
    if r2[j] < 0.8:
        ok = False; print("[FAIL] mock 선형 물리를 못 맞춤")

    # screening: phase 0 은 eef_object_dist le → δ[0]+ 가 정답 방향
    scr = Screener(ens, sg, lam_probe=0.5, horizon_s=6 * roller.dt, beta=1.0)
    anchors = phase_anchors(np.array([0] * 40 + [1] * 40 + [2] * 40), 0, 3,
                            120, 6)
    phi = np.stack([roller.F0[t] for t in anchors])
    act = np.stack([roller.actions[t] for t in anchors])
    obs = np.zeros((len(anchors), N_OBS))

    class S:
        sat_frac, cos_max, rand_dirs, top_k, refine_steps = 0.5, 0.95, 4, 3, 2
    picked, scored = screen_phase(scr, 0, phi, act, obs,
                                  roller.actions[:40], rng, S,
                                  log=lambda *a: None)
    best = picked[0]
    cos_x = float(best["d"][0])
    print(f"phase0 best dir={np.round(best['d'], 2)} (want +x heavy), "
          f"drho={best['drho']:+.3f}")
    if cos_x < 0.6 or best["drho"] >= -0.01:
        ok = False; print("[FAIL] screening 이 정답 방향(+x)을 못 찾음")

    # 점수 하한: score <= 0 인 후보는 절대 실리지 않는다
    if any(r["score"] <= 0.0 for r in picked):
        ok = False
        print(f"[FAIL] score<=0 후보가 출하됨: "
              f"{[(r['name'], round(r['score'], 4)) for r in picked]}")
    else:
        print(f"점수 하한: 출하 {len(picked)}개 전부 score>0 "
              f"({[round(r['score'], 3) for r in picked]}); "
              f"pool 내 score<=0 후보 {sum(1 for r in scored if r['score'] <= 0)}개 제외")

    # 하한을 비현실적으로 올리면 아무것도 통과하지 않아야 한다 (필터가 실제로 동작)
    class S2(S):
        min_score = 1e6
    p2, _ = screen_phase(scr, 0, phi, act, obs, roller.actions[:40], rng, S2,
                         log=lambda *a: None)
    if p2:
        ok = False; print(f"[FAIL] min_score=1e6 인데 {len(p2)}개 통과")

    # G7 mock: 실측도 mock branch 로 (screening 과 같은 rho_raw 눈금)
    meas = []
    for r in scored[:6]:
        out, _ = roller.branch(anchors[0], 0.5 * np.asarray(r["d"]), 6)
        base = roller.demo_window_features(anchors[0], 6)
        phi_t = roller.F0[anchors[0]]
        meas.append(sg.rho_raw(0, phi_t + out[-1] - base[-1])
                    - sg.rho_raw(0, phi_t))
    true_top = set(np.argsort(meas)[:3].tolist())
    rec = len(true_top & {0, 1, 2}) / 3
    print(f"G7 mock recall@3 = {rec:.2f} (random {3/len(meas):.2f})")
    if rec <= 3 / len(meas):
        ok = False; print("[FAIL] recall 이 random 이하")

    print(f"\n[selftest] {'PASS' if ok else 'FAIL'}")
    return ok


# ===========================================================================
def add_args(ap):
    ap.add_argument("--cache-dir", default="./cache")
    ap.add_argument("--boundaries", default="./artifacts/boundaries.json")
    ap.add_argument("--subgoal", default="./artifacts/subgoal.json")
    ap.add_argument("--out-dir", default="./artifacts")
    ap.add_argument("--seed", type=int, default=0)
    # collect
    ap.add_argument("--anchors", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=8, help="스텝 (0.4s@20Hz)")
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--delta-scale", type=float, default=0.3)
    ap.add_argument("--dirs-per-level", type=int, default=5)
    ap.add_argument("--max-clip", type=float, default=0.6)
    # warmup 기본 6 (구 3): 리플레이 오차의 주원인이 set_state 로 복원되지 않는
    # controller / gripper 적분기 내부상태이고, 웜업 스텝이 그것을 데모와
    # 수렴시킨다. 3 스텝에서는 gripper_open 드리프트가 채널 폭의 6.5%(RMS
    # 0.085) 남아 phase 0 의 ρ 비율이 0.93 이었다.
    ap.add_argument("--warmup", type=int, default=6,
                    help="분기 전 데모 액션 리플레이 스텝 수 (state 밖 내부상태 정렬)")
    ap.add_argument("--anchor-drift-tol", type=float, default=0.3,
                    help="Branch A 드리프트가 이 비율(데모 스케일 대비)을 "
                         "넘는 anchor 는 버림 (위치 progress 채널 기준)")
    ap.add_argument("--anchor-rho-tol", type=float, default=0.15,
                    help="Branch A 리플레이가 ρ_worst 를 이만큼 넘게 흔드는 "
                         "anchor 는 버림. contact 1프레임 뒤집힘이 ρ 전체 "
                         "스케일 이동이라 위치 기준 필터로는 안 걸러진다")
    ap.add_argument("--g6-tol", type=float, default=0.3,
                    help="[진단] raw feature 단위 drift/effect 허용 비율")
    ap.add_argument("--g6-rho-tol", type=float, default=0.3,
                    help="G6 본판정: phase 별 ρ_worst 공간 drift/effect 허용 비율")
    ap.add_argument("--no-gate", action="store_true")
    # train
    ap.add_argument("--ensemble", type=int, default=5)
    ap.add_argument("--hidden", type=int, nargs="+", default=[96, 96])
    ap.add_argument("--max-iter", type=int, default=1200)
    # screen
    # λ_probe 기본값 1.0: 학습 delta 크기는 graded_levels(0.3, 4) =
    # [0.075, 0.189, 0.476, 1.2] 이므로 1.0 은 분포 상단(외삽 아님)이다.
    # 0.5 에서는 8스텝 안에 subgoal 조건을 실제로 흔들 수 없어 |Δρ̂| 가 0.02~0.08
    # 로 앙상블 불확실성(0.013~0.025) 수준이었고, 방향 랭킹이 잡음 정렬이 됐다.
    # 1.0 으로 올리면 move phase 에서 axis[grip−](= 운반 중 파지 해제)가
    # Δρ_worst=−1.39 로 1위가 되고 steepest 도 상위권에 든다 (실측).
    ap.add_argument("--lam-probe", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--screen-anchors", type=int, default=3)
    ap.add_argument("--rand-dirs", type=int, default=6)
    ap.add_argument("--cos-max", type=float, default=0.95)
    ap.add_argument("--sat-frac", type=float, default=0.5)
    ap.add_argument("--refine-steps", type=int, default=2)
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="Top-K 에 실을 최소 screening_score. 0 이면 '불확실성을 "
                         "감안해도 열화가 예측되는' 방향만 통과")
    # G7
    ap.add_argument("--no-g7", action="store_true",
                    help="screen 후 G7 실측 검증 생략 (robosuite 불필요)")
    ap.add_argument("--g7-pool", type=int, default=10)
    ap.add_argument("--g7-anchors", type=int, default=2)
    ap.add_argument("--g7-demos", type=int, default=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("cmd", nargs="?", default="all",
                    choices=["collect", "train", "screen", "all"])
    ap.add_argument("--selftest", action="store_true")
    add_args(ap)
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(0 if run_selftest() else 1)

    data = ens = None
    if args.cmd in ("collect", "all"):
        data = cmd_collect(args)
    if args.cmd in ("train", "all"):
        ens, _ = cmd_train(args, data)
    if args.cmd in ("screen", "all"):
        sets, scored, pipeline = cmd_screen(args, ens)
        if args.no_g7:
            print("[G7] 건너뜀 (--no-g7). 문서 G7 은 미판정 상태로 남는다.")
        else:
            gate_g7(args, sets, scored, pipeline,
                    out_path=os.path.join(args.out_dir, "g7_diag.json"))


if __name__ == "__main__":
    main()
