#!/usr/bin/env python
"""
degradation.py — Stage 8: 열화 궤적 생성 (SSRR 골격, PIPELINE_v4)
================================================================================

정책은 BC 가 아니라 데모추종 closed-loop 트래커다 (D6, rollout_exec 헤더 참조):
λ=0 이면 데모를 그대로 재현하고, 상태가 밀리면 되돌아온다. 그 정책의 action 에
Stage 7 이 찾아낸 "phase 의 subgoal 방해 방향 집합"을 주입한다.

기본 모드 "burst" (sidetrack 주입 — suboptimal 데모의 모양):

    phase 를 k∈[2,4] 슬롯으로 등분, 슬롯마다 excursion 버스트 1개
    버스트 = 집합에서 랜덤으로 뽑은 방향 1개 × U(0.7,1.3) × half-sine 엔벨로프
    버스트 사이는 깨끗 → 트래커가 데모 경로로 복귀
    a'_t = fit_to_box( a_demo(k) + feedback + λ·seq[k] )

즉 궤적은 데모를 따라가다 랜덤한 시점에 방해 방향으로 이탈했다 돌아오기를
반복한다 — 상수 바이어스("top1")가 만드는 '평행 이동 드리프트'와 이것이
사용자 관찰로 갈린 지점이다. seed 는 family 고정이라 rung 간에는 같은 버스트
패턴에 λ(진폭)만 커진다 (nested ladder). "top1"(후보별 상수 바이어스),
"set"(성분 합산+Bernoulli), "random", "window" 는 ablation 용으로 남겨둔다.

주의(라벨): excursion 은 복귀하므로 phase 끝점이 깨끗하다 — success·goal_err·
ρ_end 로는 우회가 안 보인다. Stage 9 의 damage 가 경로 항(Δpath/path0)을
포함하는 이유다 (preference.damage_of 참조).

λ 램프 (문서 그대로):
    bracketing   λ 를 2배씩 키우며 task 실패가 처음 나오는 구간을 잡고
    binary search 그 경계를 좁혀 λ_max 를 얻는다
    levels     = [0, .25, .50, .75, 1.0] × λ_max
    λ_cap 까지 실패가 없으면 λ_max = λ_cap — 성공은 유지되면서 질만 나빠지는
    순수 효율 family (post 후퇴 열화가 전형).

✗ sigmoid 캘리브레이션 없음 (D7): λ 는 생성 노브일 뿐이고 라벨은 Stage 9 의
simulator 실측이다 (D8).

출력: artifacts/degradation.npz — DegradationFamily 목록 (Sec 3.7).
trajectories 는 (eef, obj, grip, contact, actions, F, k_index) 로 저장한다.
문서의 (states, actions, F) 에서 MuJoCo raw states 대신 관측 채널을 담는 것:
Stage 9 의 실측(성공/오차/경로/슬립/저크)과 Stage 10 의 reward 입력이 필요로
하는 것이 정확히 이것이고, raw state 는 리플레이 외에 소비자가 없다.

    python degradation.py --selftest      # MockStepper; robosuite 불필요
    python degradation.py                 # 실데모 (artifacts/* 필요)
"""

import argparse
import json
import os

import numpy as np

import feature_select as fs
import rollout_exec as rx
from rho import Subgoal


# ===========================================================================
# SECTION 1 — family 실행기
# ===========================================================================
class FamilyRunner:
    """한 데모에 대한 closed-loop 실행 + feature 계산 묶음."""

    def __init__(self, make_stepper, demo_eef, demo_actions, zseg, goal,
                 obj0_z, dt, *, kp=0.8, pos_scale=0.01, max_stretch=1.5,
                 inject_mode="set", p_inject=0.5, n_bursts=(2, 4),
                 burst_frac=0.12):
        self.mk = make_stepper
        self.demo_eef = np.asarray(demo_eef, float)
        self.actions = np.asarray(demo_actions, float)
        self.zseg = np.asarray(zseg, int)
        self.goal = np.asarray(goal, float)
        self.obj0_z, self.dt = float(obj0_z), float(dt)
        self.adim = self.actions.shape[1]
        self.kp, self.pos_scale, self.max_stretch = kp, pos_scale, max_stretch
        # inject_mode — 주입 시퀀스(스텝×성분)를 어떻게 만드는가:
        #   "set"    (기본) phase 의 방해 성분 집합 w 에서 스텝마다 성분별
        #            독립 Bernoulli(p_inject) + U(0.5,1.5) 스케일로 랜덤 조합.
        #   "random" 후보 방향 1개를 phase 전체에서 스텝별 Bernoulli 주입.
        #   "window" Stage 7 의 start/duration 창 안 결정적 주입 (ablation).
        # 셋 다 시퀀스를 미리 굽고(rollout_exec.inject_seq) seed 를 family 에
        # 고정하므로 "같은 패턴, 커지는 λ" nested-ladder 성질은 공통이다.
        self.inject_mode = inject_mode
        self.p_inject = float(p_inject)
        self.n_bursts = tuple(n_bursts)
        self.burst_frac = float(burst_frac)

    def _phase_span(self, phase):
        T = len(self.actions)
        span = np.where(self.zseg[:T] == phase)[0]
        return (int(span[0]), int(span[-1]) + 1) if len(span) else (0, 0)

    def seq_set(self, phase, w, seed):
        """성분별 랜덤 조합: seq[k,j] = 1[U<p]·U(0.5,1.5)·w_j (phase 안)."""
        a, b = self._phase_span(phase)
        seq = np.zeros((len(self.actions), self.adim))
        w = np.asarray(w, float)
        comp = np.nonzero(w)[0]
        if a == b or not len(comp):
            return seq
        rng = np.random.default_rng(seed)
        on = rng.random((b - a, len(comp))) < self.p_inject
        sc = rng.uniform(0.5, 1.5, (b - a, len(comp)))
        seq[a:b, comp] = on * sc * w[comp]
        return seq

    def seq_burst(self, phase, dirs, seed):
        """excursion 버스트: 방해 방향 '집합'에서 버스트마다 하나를 뽑아
        연속 주입하고, 사이 구간은 깨끗하게 둔다 (트래커가 데모로 복귀).

        상수 바이어스(top1)와의 차이가 곧 궤적 모양의 차이다:
          top1  : 매 스텝 같은 방향 → 상수 속도 오프셋 → 데모의 '평행 이동'
                  (실측: 3D 플롯이 드리프트로 보인다는 관찰이 이것)
          burst : 랜덤 시점의 짧은 이탈 + 복귀 → 데모를 따라가다 지그재그로
                  벗어났다 돌아오는 suboptimal 데모의 모양 (sidetrack 주입)

        구조 (seed 고정 → rung 간 같은 패턴, λ 는 진폭만 키움 = nested ladder):
          phase 를 k 슬롯으로 등분해 슬롯마다 버스트 1개 (경로 전체에 분산)
          버스트마다 방향 = dirs 중 랜덤 1개, 진폭 = U(0.7, 1.3)
          엔벨로프 = half-sine (스텝 함수가 아니라 부드러운 이탈-복귀)
        """
        a, b = self._phase_span(phase)
        seq = np.zeros((len(self.actions), self.adim))
        n_ph = b - a
        if n_ph <= 0 or not len(dirs):
            return seq
        rng = np.random.default_rng(seed)
        lo, hi = self.n_bursts
        k = int(rng.integers(lo, hi + 1))
        L = max(4, int(self.burst_frac * n_ph))
        slot = n_ph // max(k, 1)
        if slot <= L + 2:                    # phase 가 짧으면 버스트 수를 줄인다
            k = max(1, n_ph // (L + 3))
            slot = n_ph // k
        for i in range(k):
            s0 = a + i * slot + int(rng.integers(0, max(1, slot - L)))
            d = np.asarray(dirs[int(rng.integers(len(dirs)))], float)
            amp = float(rng.uniform(0.7, 1.3))
            Li = min(L, b - s0)
            if Li < 3:
                continue
            env = np.sin(np.pi * (np.arange(Li) + 0.5) / Li)
            seq[s0:s0 + Li] = amp * env[:, None] * d[None, :]
        return seq

    def seq_random(self, phase, d, seed):
        """방향 1개를 phase 전체에서 스텝별 Bernoulli(p_inject) 주입."""
        a, b = self._phase_span(phase)
        seq = np.zeros((len(self.actions), self.adim))
        if a == b:
            return seq
        rng = np.random.default_rng(seed)
        on = rng.random(b - a) < self.p_inject
        seq[a:b] = on[:, None] * np.asarray(d, float)
        return seq

    def seq_window(self, phase, d, start_fraction, duration_fraction):
        """Stage 7 이 뽑은 창 안에서 방향 1개를 매 스텝 결정적으로 주입."""
        a, b = self._phase_span(phase)
        seq = np.zeros((len(self.actions), self.adim))
        if a == b:
            return seq
        s0 = a + int(start_fraction * (b - a))
        s1 = a + int(min(1.0, start_fraction + duration_fraction) * (b - a))
        seq[s0:max(s1, s0 + 1)] = np.asarray(d, float)
        return seq

    def run(self, seq, lam):
        """미리 구운 주입 시퀀스로 closed-loop 실행. a' = fit(a+fb+λ·seq[k])."""
        fr, A, info = rx.closed_loop_rollout(
            self.mk(), self.demo_eef, self.actions, self.zseg, 0,
            np.zeros((1, self.adim)), float(lam), self.adim,
            kp=self.kp, pos_scale=self.pos_scale,
            amp_base=1.0, stall=False,
            max_stretch=self.max_stretch, inject_seq=seq)
        F = fs.compute_trajectory(
            fr["eef"], fr["obj"], fr["grip"], self.dt,
            fr["eef_quat"], fr["obj_quat"], self.goal, self.obj0_z,
            actions=A, contact=fr["contact"])
        traj = dict(eef=fr["eef"], obj=fr["obj"], grip=fr["grip"],
                    contact=fr["contact"], actions=A, F=F,
                    k_index=info["k_index"],
                    success=info["success"], length=int(info["length"]),
                    realised=float(info["realised"]), lam=float(lam))
        return traj, info


def rho_endpoint(sg, phase, traj, zseg):
    """phase 창에 마지막으로 매핑되는 실행 프레임의 ρ_raw."""
    k = np.asarray(traj["k_index"], int)
    zz = zseg[np.clip(k, 0, len(zseg) - 1)]
    idx = np.where(zz == phase)[0]
    i = int(idx[-1]) if len(idx) else len(traj["F"]) - 1
    i = min(i, len(traj["F"]) - 1)
    # rho_worst: Stage 7 screening 이 방향을 고른 눈금과 같은 눈금으로 잰다.
    # rho_raw(RMS·전역 d_start)로 재면 "방향을 고른 기준"과 "사다리를 재는
    # 기준"이 달라, screening 이 강하다고 본 방향이 사다리에서 평평하게 보이는
    # 인공적 불일치가 생긴다 (조건별 정규화 유무의 차이).
    return float(sg.rho_worst(phase, traj["F"][i]))


# ===========================================================================
# SECTION 2 — λ_max: bracketing → binary search
# ===========================================================================
def find_lambda_max(runner, seq, *, lam0=0.5, lam_cap=4.0, n_bisect=3,
                    log=print):
    """task 실패의 경계를 찾는다. -> (lam_ok, lam_fail | None, kind, probes)

    lam_ok   성공이 확인된 가장 큰 λ   ← 효율 사다리의 상단
    lam_fail 실패가 확인된 가장 작은 λ ← 별도의 cliff rung (없으면 None)

    성공(λ) 이 단조라는 보장은 없다(파지 물리의 cliff) — bracketing 은 처음
    만나는 실패를, bisection 은 그 근방 경계를 잡을 뿐이고, 최종 판정은
    Stage 9 의 G8 이 한다. λ_cap 까지 실패가 없으면 kind='no-cliff'.

    왜 두 값을 나눠 반환하는가 (이전에는 lam_fail 하나만 λ_max 로 썼다):
    levels = [0, .25, .5, .75, 1.0] × lam_fail 로 사다리를 만들면 최상단 rung 이
    정의상 실패다. 실측 결과 cliff family 20/20 이 그랬고, damage 라벨에 실패
    가중치 2.0 이 얹혀 pair 의 54% 가 '성공 vs 실패' 이진 대비가 됐다. 그러면
    R_θ 는 효율이 아니라 성공 판별기를 학습한다 — D-REX/T-REX 골격이 피하려는
    실패 모드다. lam_ok 로 사다리를 만들면 5 rung 전부 성공(순수 효율 랭킹)이고,
    cliff 는 별도 rung 으로 표시해 pair 를 분리 학습할 수 있다.
    """
    lam_ok, lam = 0.0, float(lam0)
    lam_fail = None
    probes = []
    while lam <= lam_cap + 1e-9:
        traj, _ = runner.run(seq, lam)
        probes.append((lam, bool(traj["success"])))
        if traj["success"]:
            lam_ok = lam
            lam *= 2.0
        else:
            lam_fail = lam
            break
    if lam_fail is None:
        return float(lam_cap), None, "no-cliff", probes
    for _ in range(n_bisect):
        mid = 0.5 * (lam_ok + lam_fail)
        traj, _ = runner.run(seq, mid)
        probes.append((mid, bool(traj["success"])))
        if traj["success"]:
            lam_ok = mid
        else:
            lam_fail = mid
    # lam0 자체가 실패면 lam_ok 가 0 근처에 머문다 — 효율 사다리를 만들 여지가
    # 없다는 신호이므로 그대로 반환하고 G8 의 R4(spread) 가 걸러낸다.
    return float(lam_ok), float(lam_fail), "cliff", probes


# ===========================================================================
# SECTION 3 — 전체 조립
# ===========================================================================
LEVELS = [0.0, 0.25, 0.50, 0.75, 1.0]      # × λ_ok — 순수 효율 사다리
CLIFF_LEVEL = 1.25                          # cliff rung 의 x 축 표식 (λ_fail)
ACTION_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz", "grip"]


def phase_component_set(cands, top_k, w_thresh=0.25):
    """후보 방향들 → phase 의 signed 방해 성분 집합 w (adim,).

    top-k 단위방향을 성분별로 합산해 max|·| 로 정규화하고, |w_j| < θ_w 인
    성분은 버린다. 후보들이 공통으로 미는 성분은 살아남고(부호 상쇄 포함),
    한 후보에만 있던 잡성분은 떨어져 나간다. 크기 가중은 두지 않는다 —
    predicted_drho 로 가중하면 물리적으로 치명적인데 FCM 예측치가 작은 성분
    (move 의 grip− 낙하 등)이 소거된다.
    """
    W = np.zeros(len(cands[0]["direction"]))
    for c in cands[:top_k]:
        W = W + np.asarray(c["direction"], float)
    W = W / max(1e-9, np.abs(W).max())
    W[np.abs(W) < w_thresh] = 0.0
    comps = [f"{ACTION_NAMES[j]}{'+' if W[j] > 0 else '-'}{abs(W[j]):.2f}"
             for j in np.nonzero(W)[0]]
    return W, comps


def family_specs(r, phase, cands, args):
    """(candidate-dict, inject_seq) 목록 — inject_mode 별 family 정의.

    "top1" (기본): 후보를 합산하지 않고 각 후보를 독립 family 로 만든다.
    "set" 의 성분 합산은 Stage 7 의 랭킹을 지운다 — 실측 cos(Top-1, 합산 w):
    phase0 0.686 / phase1 0.331 / phase2 0.636. phase 1 은 지배 성분이 dz 에서
    drx 로 바뀌었다. 게다가 합산은 점수가 낮은(또는 음수였던) 후보까지 방향에
    섞으므로 "phase-conditioned FCM 방향" 이라는 주장의 근거가 사라지고,
    Ablation A(등방 Gaussian vs FCM 방향)도 두 조건이 구별되지 않는다 —
    후보 절반이 rand* 였으므로 합산 결과는 이미 절반이 등방 Gaussian 이다.
    """
    if r.inject_mode == "burst":
        # 방해 방향 '집합' 전체를 쓴다: 버스트마다 그 안에서 랜덤으로 하나.
        # family = demo × phase × seed (후보별이 아님) — 사용자가 그린
        # sidetrack 주입 그대로다: "집합들을 찾으면 액션들을 아무거나 랜덤으로".
        dirs = []
        for c in cands[:args.top_k]:
            d = np.asarray(c["direction"], float)
            m = float(np.abs(d).max())
            if m > 1e-9:
                dirs.append(d / m)
        if not dirs:
            return []
        specs = []
        for si in range(args.n_seeds):
            cand = dict(candidate_id=f"p{phase}_burst{si}", mode="burst",
                        sources=[c["candidate_id"] for c in cands[:args.top_k]],
                        names=[c.get("name", "") for c in cands[:args.top_k]],
                        n_dirs=len(dirs), seed=args.seed + si)
            specs.append((cand, r.seq_burst(phase, dirs, args.seed + si)))
        return specs
    if r.inject_mode == "top1":
        specs = []
        for c in cands[:args.top_k]:
            for si in range(args.n_seeds):
                w = np.asarray(c["direction"], float)
                w = w / max(1e-9, np.abs(w).max())     # max|·|=1 로만 정규화
                cand = dict(c, candidate_id=f"{c['candidate_id']}_s{si}",
                            components=[
                                f"{ACTION_NAMES[j]}{'+' if w[j] > 0 else '-'}"
                                f"{abs(w[j]):.2f}" for j in np.nonzero(w)[0]],
                            weights=[float(x) for x in w],
                            sources=[c["candidate_id"]],
                            seed=args.seed + si)
                # 스텝 단위 Bernoulli (seq_random) 를 쓴다. seq_set 의 성분별
                # Bernoulli 는 스텝마다 성분 부분집합을 뽑으므로 그 스텝의 실주입
                # 방향이 후보 방향이 아니게 된다 — top1 의 목적(FCM 이 고른 방향을
                # 그대로 실행)과 정면으로 어긋난다. 스텝 단위면 주입되는 스텝에서
                # 방향이 정확히 보존되고, 간헐성(SSRR 성질)은 유지된다.
                specs.append((cand, r.seq_random(phase, w, args.seed + si)))
        return specs
    if r.inject_mode == "set":
        w, comps = phase_component_set(cands, args.top_k, args.w_thresh)
        specs = []
        for si in range(args.n_seeds):
            cand = dict(candidate_id=f"p{phase}_set{si}",
                        components=comps, weights=[float(x) for x in w],
                        sources=[c["candidate_id"] for c in cands[:args.top_k]],
                        seed=args.seed + si)
            specs.append((cand, r.seq_set(phase, w, args.seed + si)))
        return specs
    if r.inject_mode == "random":
        return [(c, r.seq_random(phase, c["direction"], args.seed))
                for c in cands[:args.top_k]]
    return [(c, r.seq_window(phase, c["direction"], c["start_fraction"],
                             c["duration_fraction"]))
            for c in cands[:args.top_k]]


def build_families(runners, action_sets, sg, args, log=print):
    """runners: {demo_id: FamilyRunner} × action_sets → DegradationFamily 들.

    λ=0 rung 은 데모별로 한 번만 실행해 그 데모의 모든 family 가 공유한다
    (전부 같은 궤적이다). 그 rung 이 데모를 재현하는지(D6)도 여기서 검증.
    """
    zero_runs = {}
    for did, r in runners.items():
        traj, _ = r.run(np.zeros((len(r.actions), r.adim)), 0.0)
        err = float(np.abs(np.asarray(traj["eef"])[:len(r.demo_eef)]
                           - r.demo_eef[:len(traj["eef"])]).max())
        zero_runs[did] = traj
        log(f"  [λ=0] {did}: 데모 재현 max eef err={err:.5f} m, "
            f"success={traj['success']}")
        if err > 0.02:
            log(f"  [warn] {did}: λ=0 rung 이 데모에서 {err*100:.1f}cm 이탈 — "
                f"트래커 게인 확인 필요")

    families = []
    for did, r in runners.items():
        for pk, cands in sorted(action_sets.items()):
            phase = int(pk.split("_")[1])
            for cand, seq in family_specs(r, phase, cands, args):
                if not seq.any():
                    continue
                fid = f"{did}:{cand['candidate_id']}"
                lam_ok, lam_fail, kind, probes = find_lambda_max(
                    r, seq, lam0=args.lam0, lam_cap=args.lam_cap,
                    n_bisect=args.n_bisect)
                # 효율 사다리: λ_ok 기준 5 rung (전부 성공이 기대값)
                trajs = [zero_runs[did]]
                fracs = list(LEVELS)
                lams = [lv * lam_ok for lv in LEVELS]
                kinds = ["eff"] * len(LEVELS)
                for lv in LEVELS[1:]:
                    traj, _ = r.run(seq, lv * lam_ok)
                    trajs.append(traj)
                # cliff rung: 실패가 확인된 λ 를 하나만 따로 붙인다. 라벨에서
                # '효율 pair' 와 '실패 pair' 를 분리할 수 있게 표시해 둔다.
                if lam_fail is not None and args.with_cliff:
                    traj, _ = r.run(seq, lam_fail)
                    trajs.append(traj)
                    fracs.append(CLIFF_LEVEL)
                    lams.append(float(lam_fail))
                    kinds.append("cliff")
                rhos = [rho_endpoint(sg, phase, t, r.zseg) for t in trajs]
                succ = [bool(t["success"]) for t in trajs]
                log(f"  {fid:<22} λ_ok={lam_ok:.2f} "
                    f"λ_fail={'—' if lam_fail is None else f'{lam_fail:.2f}'} "
                    f"({kind}, {len(probes)} probes)  ρ_end="
                    f"[{', '.join(f'{x:+.2f}' for x in rhos)}]  "
                    f"success={''.join('o' if s else 'x' for s in succ)}")
                families.append(dict(
                    family_id=fid, candidate=cand, demo_id=did, phase_id=phase,
                    lambda_max=lam_ok, lambda_ok=lam_ok,
                    lambda_fail=lam_fail, lambda_kind=kind,
                    lambda_levels=lams, level_fracs=fracs, rung_kind=kinds,
                    probes=probes, trajectories=trajs, rho_endpoint=rhos))
    return families


def save_families(path, families, meta):
    np.savez_compressed(path,
                        families=np.array(families, dtype=object),
                        meta=json.dumps(meta))


def load_families(path):
    z = np.load(path, allow_pickle=True)
    return list(z["families"]), json.loads(str(z["meta"]))


# ===========================================================================
# SECTION 4 — 실데모 실행기
# ===========================================================================
def build_runners(entries, boundaries, args, log=print):
    import frame_extract as fx
    from fcm import z_from_bounds
    runners, env, key = {}, None, None
    for e in entries:
        env_info, demos = fx.read_demo(e["meta"]["source"])
        name = e["meta"].get("source_demo") or demos[0][0]
        _, states, actions, xml, _tq = next(d for d in demos if d[0] == name)
        k = (env_info["env_name"], str(env_info["robots"]))
        if k != key:
            if env is not None:
                env.close()
            log(f"[env] {k[0]} / {k[1]}")
            env = fx.build_env(env_info)
            key = k
        fx.reset_to_scene(env, xml)
        ex = fx.FrameExtractor(env, env_info.get("object_type"), dt=e["dt"])
        # SimStepper.reset 은 states[0] 로 되돌리므로 scene xml 이 맞아야 한다.
        env_ref, states_ref, ot = env, states, env_info.get("object_type")

        def mk(env=env_ref, states=states_ref, ot=ot, ex=ex, xml=xml, fx=fx):
            fx.reset_to_scene(env, xml)
            return rx.SimStepper(env, states, ot, ex)

        zseg = z_from_bounds(boundaries[e["demo_id"]]["bounds"], len(e["F"]))
        runners[e["demo_id"]] = FamilyRunner(
            mk, e["eef_pos"], e["actions"][:len(e["F"]) - 1], zseg[:-1],
            e["goal"], float(e["obj_pos"][0, 2]), e["dt"],
            kp=args.kp, pos_scale=args.pos_scale, max_stretch=args.max_stretch,
            inject_mode=args.inject_mode, p_inject=args.p_inject,
            n_bursts=(args.min_bursts, args.max_bursts),
            burst_frac=args.burst_frac)
    return runners, env


def run(args):
    from extract_features import load_cache
    entries = load_cache(args.cache_dir)
    if args.demos:
        entries = entries[:args.demos]
    with open(args.boundaries) as f:
        boundaries = json.load(f)
    with open(args.action_sets) as f:
        action_sets = json.load(f)
    sg = Subgoal.load(args.subgoal)

    if args.inject_mode == "set":
        for pk, cands in sorted(action_sets.items()):
            _, comps = phase_component_set(cands, args.top_k, args.w_thresh)
            print(f"[set] {pk}: {{{', '.join(comps)}}}")

    runners, env = build_runners(entries, boundaries, args)
    try:
        fams = build_families(runners, action_sets, sg, args)
    finally:
        if env is not None:
            env.close()

    meta = dict(n_families=len(fams), levels=LEVELS,
                cliff_level=CLIFF_LEVEL, with_cliff=bool(args.with_cliff),
                lam_cap=args.lam_cap,
                kp=args.kp, pos_scale=args.pos_scale, top_k=args.top_k,
                inject_mode=args.inject_mode, p_inject=args.p_inject,
                n_bursts=[args.min_bursts, args.max_bursts],
                burst_frac=args.burst_frac,
                n_seeds=args.n_seeds, w_thresh=args.w_thresh,
                demos=[e["demo_id"] for e in entries])
    out = os.path.join(args.out_dir, "degradation.npz")
    save_families(out, fams, meta)
    n_rung = [len(f["trajectories"]) for f in fams]
    print(f"\n[out] {out}  ({len(fams)} families, rung "
          f"{min(n_rung)}~{max(n_rung)}개)")

    n_cliff = sum(1 for f in fams if f["lambda_kind"] == "cliff")
    print(f"[stat] cliff {n_cliff} / no-cliff {len(fams) - n_cliff}")
    # 효율 rung 이 전부 성공했는가 = λ_ok 정의가 의도대로 동작하는가.
    # 소수(<2%)의 실패는 파지 물리의 비단조성(성공(λ)이 단조라는 보장이 없다 —
    # find_lambda_max 주석) 때문에 정상적으로 발생하고, 해당 family 는 G8 의
    # R2/R3 가 기각한다. 실측: 48 families × 5 rung 에서 1/240 (0.4%).
    eff_fail = sum(1 for f in fams
                   for t, kd in zip(f["trajectories"], f["rung_kind"])
                   if kd == "eff" and not t["success"])
    n_eff = sum(1 for f in fams for kd in f["rung_kind"] if kd == "eff")
    rate = eff_fail / max(1, n_eff)
    if eff_fail == 0:
        tag = "OK — 순수 효율 사다리"
    elif rate < 0.02:
        tag = ("정상 범위 — 비단조 파지 물리; 해당 family 는 G8 R2/R3 가 "
               "기각한다")
    else:
        tag = ("λ_ok 근방 성공이 불안정 — n_bisect 상향 또는 λ 탐색 재검토 "
               "필요")
    print(f"[stat] 효율 rung 실패 {eff_fail}/{n_eff} ({rate:.1%}) — {tag}")
    if args.plot:
        viz_ladder(fams, sg, os.path.join(args.out_dir,
                                          "degradation_ladder.png"))
        viz_traj3d(fams, os.path.join(args.out_dir,
                                      "degradation_traj3d.png"))
    return fams


# ===========================================================================
# SECTION 5 — 시각화
# ===========================================================================
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def viz_ladder(fams, sg, out_png):
    """phase 별: x = level 분율, y = Δρ_end (λ=0 rung 대비). o=성공 x=실패."""
    plt = _plt()
    phases = sorted(set(f["phase_id"] for f in fams))
    fig, axes = plt.subplots(1, len(phases), figsize=(5 * len(phases), 4.2),
                             squeeze=False)
    cmap = plt.get_cmap("tab10")
    for ax, ph in zip(axes[0], phases):
        sub = [f for f in fams if f["phase_id"] == ph]
        cands = sorted(set(f["candidate"]["candidate_id"] for f in sub))
        for f in sub:
            ci = cands.index(f["candidate"]["candidate_id"])
            dr = np.asarray(f["rho_endpoint"]) - f["rho_endpoint"][0]
            xs = f.get("level_fracs", LEVELS)
            ax.plot(xs, dr, color=cmap(ci % 10), alpha=0.55, lw=1.4)
            for x, y, t, kd in zip(xs, dr, f["trajectories"],
                                   f.get("rung_kind", ["eff"] * len(xs))):
                ax.scatter(x, y, marker="o" if t["success"] else "x",
                           color=cmap(ci % 10), s=44 if kd == "cliff" else 28,
                           edgecolors="k" if kd == "cliff" else "none",
                           linewidths=0.6, zorder=3)
        ax.axvline(1.0, color="0.7", lw=0.6, ls=":")
        for ci, cid in enumerate(cands):
            ax.plot([], [], color=cmap(ci % 10), label=cid)
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_xlabel("level (× λ_ok;  1.25 = cliff rung @ λ_fail)")
        ax.set_ylabel("Δρ_end vs λ=0")
        ax.set_title(f"phase {ph} [{sg.labels[ph]}]  (o=success, x=fail)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("degradation ladders: measured ρ endpoint per rung")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"[plot] {out_png}")


def viz_traj3d(fams, out_png, max_panels=3):
    """phase 마다 사다리 낙차가 가장 큰 family 하나씩: level 별 3D eef 경로."""
    plt = _plt()
    picks = []
    for ph in sorted(set(f["phase_id"] for f in fams)):
        sub = [f for f in fams if f["phase_id"] == ph]
        picks.append(max(sub, key=lambda f: abs(f["rho_endpoint"][-1]
                                                - f["rho_endpoint"][0])))
    picks = picks[:max_panels]
    fig = plt.figure(figsize=(5.6 * len(picks), 5))
    cmap = plt.get_cmap("viridis")
    for i, f in enumerate(picks):
        ax = fig.add_subplot(1, len(picks), i + 1, projection="3d")
        for lv, t in zip(f.get("level_fracs", LEVELS), f["trajectories"]):
            e = np.asarray(t["eef"])
            ax.plot(*e.T, color=cmap(min(lv, 1.0)),
                    lw=1.6 if lv == 0 else 1.1,
                    label=f"λ={t['lam']:.2f} "
                          f"{'o' if t['success'] else 'x'}")
        ax.set_title(f"{f['family_id']}  (phase {f['phase_id']}, "
                     f"λ_ok={f['lambda_max']:.2f} {f['lambda_kind']})",
                     fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    fig.suptitle("degradation ladder trajectories (color = λ level)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_png}")


# ===========================================================================
# SECTION 6 — selftest (MockStepper; robosuite 불필요)
# ===========================================================================
def run_selftest():
    import tempfile
    from fcm import _mock_subgoal

    print("=== degradation SELFTEST (mock plant) ===")
    ok = True
    demo_eef, demo_A, zseg = rx._mock_demo(T=120, adim=7)
    goal = np.array([0.5, 0.0, 0.85])
    mk = lambda: rx.MockStepper(T=len(demo_A), adim=7, goal=goal)
    sg = _mock_subgoal()
    # p_inject=1.0: mock 의 cliff 문턱은 전 스텝 주입 기준으로 잡혀 있다.
    # p=0.5 의 확률성 자체는 아래에서 주입 횟수 통계로 따로 검증.
    runner = FamilyRunner(mk, demo_eef[:-1], demo_A, zseg, goal, 0.85, 0.05,
                          kp=0.8, pos_scale=0.01, max_stretch=1.5,
                          inject_mode="set", p_inject=1.0)

    class A:
        top_k, lam0, lam_cap, n_bisect, seed = 2, 0.5, 4.0, 3, 0
        kp, pos_scale, max_stretch = 0.8, 0.01, 1.5
        n_seeds, w_thresh = 1, 0.25
        with_cliff = True

    # mock 후보: phase 1(뒤 절반, 물체 파지·운반 중) 에 +x 주입 → 과도하면
    # goal 을 지나쳐 실패해야 한다 (cliff 존재)
    action_sets = {"phase_1": [dict(
        candidate_id="p1_c000", direction=[1, 0, 0, 0, 0, 0, 0],
        subspace="position", start_fraction=0.0, duration_fraction=1.0,
        predicted_drho=-0.1, uncertainty=0.01, screening_score=0.09)]}

    fams = build_families({"demo_000": runner}, action_sets, sg, A,
                          log=lambda *a: None)
    if len(fams) != 1:
        print(f"[FAIL] expected 1 family, got {len(fams)}")
        return False
    f = fams[0]

    z0 = f["trajectories"][0]
    if not z0["success"] or z0["lam"] != 0.0:
        ok = False; print(f"[FAIL] λ=0 rung: success={z0['success']}")
    err = float(np.abs(np.asarray(z0["eef"])[:len(demo_eef)]
                       - demo_eef[:len(z0['eef'])]).max())
    print(f"λ=0 rung 데모 재현 err={err:.6f}")
    if err > 1e-6:
        ok = False; print("[FAIL] λ=0 이 데모를 재현하지 못함")

    print(f"λ_ok={f['lambda_ok']:.3f}  λ_fail={f['lambda_fail']:.3f} "
          f"({f['lambda_kind']}), "
          f"probes={[(round(l, 2), s) for l, s in f['probes']]}")
    if f["lambda_kind"] != "cliff" or not (0 < f["lambda_fail"] <= A.lam_cap):
        ok = False; print("[FAIL] cliff 를 찾아야 한다 (+x 과주입 = 과녁 초과)")
    if not (f["lambda_ok"] < f["lambda_fail"]):
        ok = False; print(f"[FAIL] λ_ok({f['lambda_ok']}) >= "
                          f"λ_fail({f['lambda_fail']})")

    # 핵심: 효율 rung(λ_ok 기준 5개)은 전부 성공, cliff rung 만 실패여야 한다.
    succ = [t["success"] for t in f["trajectories"]]
    kinds = f["rung_kind"]
    print(f"rung kind    = {kinds}")
    print(f"rung success = {succ}")
    eff_bad = [s for s, k in zip(succ, kinds) if k == "eff" and not s]
    if eff_bad:
        ok = False
        print(f"[FAIL] 효율 rung 에 실패가 있다 — λ_ok 정의가 안 지켜졌다")
    if kinds[-1] != "cliff" or succ[-1]:
        ok = False; print("[FAIL] 마지막 rung 은 cliff(실패)여야 한다")
    if len(succ) != len(LEVELS) + 1:
        ok = False; print(f"[FAIL] rung 수 {len(succ)} != {len(LEVELS)}+1")

    # 열화 단조성(대체로): 최종 goal 오차가 level 에 따라 비감소
    errs = [float(np.linalg.norm(np.asarray(t["obj"])[-1] - goal))
            for t in f["trajectories"]]
    n_viol = sum(1 for a, b in zip(errs, errs[1:]) if b < a - 1e-6)
    print(f"final goal err per rung = {[round(e, 3) for e in errs]} "
          f"(violations {n_viol})")
    if n_viol > 1:
        ok = False; print("[FAIL] 사다리가 goal 오차 기준으로 뒤죽박죽")

    # --no-cliff 경로: 효율 rung 만
    class A2(A):
        with_cliff = False
    f2 = build_families({"demo_000": runner}, action_sets, sg, A2,
                        log=lambda *a: None)[0]
    if len(f2["trajectories"]) != len(LEVELS) or not all(
            t["success"] for t in f2["trajectories"]):
        ok = False
        print(f"[FAIL] --no-cliff: rung {len(f2['trajectories'])}, "
              f"success {[t['success'] for t in f2['trajectories']]}")
    else:
        print(f"--no-cliff -> {len(LEVELS)} rung 전부 성공 (순수 효율 사다리)")

    # top1 모드: 후보를 합산하지 않고 후보마다 독립 family
    r_t1 = FamilyRunner(mk, demo_eef[:-1], demo_A, zseg, goal, 0.85, 0.05,
                        kp=0.8, pos_scale=0.01, inject_mode="top1",
                        p_inject=1.0)
    specs = family_specs(r_t1, 1, action_sets["phase_1"], A)
    d_req = np.asarray(action_sets["phase_1"][0]["direction"], float)
    w_used = np.asarray(specs[0][0]["weights"], float)
    cos = float(np.dot(d_req, w_used)
                / (np.linalg.norm(d_req) * np.linalg.norm(w_used)))
    print(f"top1 모드: family {len(specs)}개, cos(Top-1 요청, 실주입) = {cos:+.3f}")
    if abs(cos - 1.0) > 1e-6:
        ok = False; print("[FAIL] top1 은 후보 방향을 그대로 써야 한다")
    # 주입되는 모든 스텝의 방향이 후보 방향과 정확히 평행해야 한다 (성분별
    # Bernoulli 를 쓰면 여기서 깨진다)
    seq_t1 = specs[0][1]
    nz = seq_t1[np.abs(seq_t1).sum(axis=1) > 0]
    coss = [float(np.dot(rw, w_used) / (np.linalg.norm(rw)
                                        * np.linalg.norm(w_used)))
            for rw in nz]
    print(f"    주입 스텝 {len(nz)}개, 스텝별 cos = "
          f"[{min(coss):.3f}, {max(coss):.3f}]" if coss else "    주입 스텝 없음")
    if not coss or min(coss) < 1.0 - 1e-9:
        ok = False
        print("[FAIL] top1 의 스텝별 주입 방향이 후보 방향과 평행하지 않다")

    # set 모드 확률성: p=0.5, 성분 2개(w=[1,0,1,...]) — 스텝별 조합이 실제로
    # 랜덤 부분집합인지 (comp0 만 / comp1 만 / 둘 다 / 없음 이 섞여야 한다)
    r05 = FamilyRunner(mk, demo_eef[:-1], demo_A, zseg, goal, 0.85, 0.05,
                       kp=0.8, pos_scale=0.01, inject_mode="set",
                       p_inject=0.5)
    w2 = np.zeros(7); w2[0], w2[2] = 1.0, -1.0
    seq = r05.seq_set(1, w2, seed=1)
    rows = seq[np.abs(seq).sum(axis=1) > 0]
    n_phase = int((r05.zseg == 1).sum())
    patt = set((bool(r[0]), bool(r[2])) for r in rows)
    _, info05 = r05.run(seq, 1.0)
    print(f"set p=0.5: 주입 {info05['n_noise']}/{n_phase} 스텝, "
          f"성분조합 {len(patt)}종 (기대: on/off 섞임)")
    if not (0.35 * n_phase <= info05["n_noise"] <= 0.95 * n_phase):
        ok = False; print("[FAIL] 주입률이 성분별 p=0.5 와 안 맞음")
    if len(patt) < 2:
        ok = False; print("[FAIL] 스텝별 성분 조합이 랜덤하지 않음")
    sc = np.abs(rows[:, [0, 2]][rows[:, [0, 2]] != 0])
    if not (0.5 - 1e-9 <= sc.min() and sc.max() <= 1.5 + 1e-9):
        ok = False; print(f"[FAIL] 성분 스케일 U(0.5,1.5) 위반: "
                          f"[{sc.min():.2f},{sc.max():.2f}]")
    if not np.allclose(seq, r05.seq_set(1, w2, seed=1)):
        ok = False; print("[FAIL] seed 고정 재현성 위반 (nested ladder 깨짐)")

    # burst 모드 (기본): excursion 구조 검사
    r_bu = FamilyRunner(mk, demo_eef[:-1], demo_A, zseg, goal, 0.85, 0.05,
                        kp=0.8, pos_scale=0.01, inject_mode="burst",
                        n_bursts=(2, 4), burst_frac=0.12)
    dirs2 = [np.eye(7)[0], np.eye(7)[2]]           # +x, +z 두 방향의 집합
    seqb = r_bu.seq_burst(1, dirs2, seed=5)
    a_, b_ = r_bu._phase_span(1)
    nz = np.abs(seqb).sum(axis=1) > 0
    # (a) phase 밖 주입 없음
    if nz[:a_].any() or nz[b_:].any():
        ok = False; print("[FAIL] burst 가 phase 밖에 주입")
    # (b) 연속 run 으로 나뉘고 (2~4개), 사이에 깨끗한 구간이 있다
    runs, s = [], None
    for i in range(len(nz)):
        if nz[i] and s is None: s = i
        if not nz[i] and s is not None: runs.append((s, i)); s = None
    if s is not None: runs.append((s, len(nz)))
    gaps = (b_ - a_) - sum(e - st for st, e in runs)
    if not (1 <= len(runs) <= 4 and gaps > 0):
        ok = False; print(f"[FAIL] burst 구조: runs={len(runs)}, 깨끗한 스텝={gaps}")
    # (c) 각 버스트는 집합 중 '한' 방향과 정확히 평행 + half-sine 엔벨로프
    #     (양끝보다 중간이 큼)
    for st_, e_ in runs:
        seg = seqb[st_:e_]
        coss = [max(abs(float(np.dot(row, d))) / (np.linalg.norm(row) + 1e-12)
                    for d in dirs2) for row in seg]
        if min(coss) < 1.0 - 1e-9:
            ok = False; print("[FAIL] burst 방향이 집합 원소와 평행하지 않음")
        mag = np.linalg.norm(seg, axis=1)
        if not (mag[len(mag)//2] > mag[0] and mag[len(mag)//2] > mag[-1]):
            ok = False; print("[FAIL] burst 엔벨로프가 half-sine 이 아님")
    # (d) seed 재현성 (nested ladder 전제)
    if not np.allclose(seqb, r_bu.seq_burst(1, dirs2, seed=5)):
        ok = False; print("[FAIL] burst seed 재현성 위반")
    # (e) 두 방향이 실제로 섞여 쓰이는가 (여러 seed 에 걸쳐)
    used = set()
    for sd in range(6):
        sq = r_bu.seq_burst(1, dirs2, seed=sd)
        for row in sq[np.abs(sq).sum(axis=1) > 0]:
            used.add(int(np.argmax(np.abs(row))))
    if used != {0, 2}:
        ok = False; print(f"[FAIL] burst 가 집합의 방향들을 안 섞음: {used}")
    print(f"burst 모드: {len(runs)}개 excursion, 깨끗한 스텝 {gaps}, "
          f"방향 사용 축={sorted(used)}, half-sine·seed 재현 OK")
    # (f) family_specs 경로: family = seed 당 1개 (후보별이 아님)
    class AB(A):
        n_seeds = 2
    specs_b = family_specs(r_bu, 1, [
        dict(candidate_id="p1_c000", direction=[1, 0, 0, 0, 0, 0, 0]),
        dict(candidate_id="p1_c001", direction=[0, 0, 1, 0, 0, 0, 0])], AB)
    if len(specs_b) != 2 or specs_b[0][0]["n_dirs"] != 2:
        ok = False; print(f"[FAIL] burst family_specs: {len(specs_b)}개")
    else:
        print(f"burst family_specs: seed 당 1 family × {AB.n_seeds} "
              f"(방향 {specs_b[0][0]['n_dirs']}개 공유)")

    # phase_component_set: 상쇄·소수성분 제거 확인
    cs = [dict(direction=[1, 0, 0.6, 0, 0, 0, 0]),
          dict(direction=[-1, 0, 0.6, 0, 0.1, 0, 0])]
    wagg, comps = phase_component_set(cs, 2, 0.25)
    print(f"phase_component_set: {comps} (dx 상쇄, dry 잡성분 제거 기대)")
    if abs(wagg[0]) > 1e-9 or wagg[2] != 1.0 or abs(wagg[4]) > 1e-9:
        ok = False; print(f"[FAIL] 집합 집계가 틀림: {wagg}")

    # npz 왕복
    tmp = tempfile.mktemp(suffix=".npz")
    save_families(tmp, fams, dict(test=True))
    fams2, meta2 = load_families(tmp)
    if len(fams2) != 1 or meta2.get("test") is not True:
        ok = False; print("[FAIL] npz round-trip")
    else:
        print("npz round-trip OK")
    os.remove(tmp)

    print(f"\n[selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cache-dir", default="./cache")
    ap.add_argument("--boundaries", default="./artifacts/boundaries.json")
    ap.add_argument("--subgoal", default="./artifacts/subgoal.json")
    ap.add_argument("--action-sets", default="./artifacts/action_sets.json")
    ap.add_argument("--out-dir", default="./artifacts")
    ap.add_argument("--top-k", type=int, default=4,
                    help="phase 당 사용할 후보 수 (action_sets 순위순)")
    ap.add_argument("--demos", type=int, default=0,
                    help="앞에서 N개 데모만 (0 = 전부)")
    ap.add_argument("--lam0", type=float, default=0.5)
    ap.add_argument("--lam-cap", type=float, default=4.0)
    ap.add_argument("--n-bisect", type=int, default=3)
    # kp=0.1 (게인 10): kp=0.8 (게인 80)은 위치 주입을 평형 오프셋 ~1.25cm 로
    # 뭉개서 λ 를 아무리 키워도 cliff 가 나오지 않는다 (실측: λ=4 에서 ρ_end
    # 변화 0.01). 게인 10이면 오프셋 ~10cm·λ_exec — 태스크를 실제로 흔들면서
    # λ=0 재현(오차 0.8mm)은 게인과 무관하게 유지된다.
    ap.add_argument("--kp", type=float, default=0.1)
    ap.add_argument("--pos-scale", type=float, default=0.01)
    ap.add_argument("--max-stretch", type=float, default=1.5)
    ap.add_argument("--inject-mode", default="burst",
                    choices=["burst", "top1", "set", "random", "window"],
                    help="burst (기본): 방해 방향 '집합'에서 버스트마다 랜덤 "
                         "방향을 뽑아 이탈-복귀 excursion 주입 — suboptimal "
                         "데모 모양 (sidetrack). "
                         "top1: 후보별 상수 바이어스 (드리프트형; ablation). "
                         "set/random/window: ablation")
    ap.add_argument("--min-bursts", type=int, default=2,
                    help="burst: phase 당 excursion 최소 개수")
    ap.add_argument("--max-bursts", type=int, default=4,
                    help="burst: phase 당 excursion 최대 개수")
    ap.add_argument("--burst-frac", type=float, default=0.12,
                    help="burst: excursion 하나의 길이 = frac × phase 길이")
    ap.add_argument("--with-cliff", action="store_true", default=True,
                    help="λ_fail 에 cliff rung 을 하나 추가 (기본 on)")
    ap.add_argument("--no-cliff", dest="with_cliff", action="store_false",
                    help="효율 사다리만 (cliff rung 없음)")
    ap.add_argument("--p-inject", type=float, default=0.5,
                    help="set: 성분별 / random: 스텝별 주입 확률")
    ap.add_argument("--n-seeds", type=int, default=2,
                    help="set 모드: demo×phase 당 랜덤 패턴(family) 수")
    ap.add_argument("--w-thresh", type=float, default=0.25,
                    help="set 모드: 정규화 후 살아남을 성분의 최소 |w|")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(0 if run_selftest() else 1)
    run(args)


if __name__ == "__main__":
    main()
