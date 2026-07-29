#!/usr/bin/env python
"""
degradation.py — Stage 8: 열화 궤적 생성 (SSRR 골격, PIPELINE_v4)
================================================================================

정책은 BC 가 아니라 데모추종 closed-loop 트래커다 (D6, rollout_exec 헤더 참조):
λ=0 이면 데모를 그대로 재현하고, 상태가 밀리면 되돌아온다. 그 정책의 action 에
Stage 7 이 찾아낸 "phase 의 subgoal 방해 성분 집합" 을 주입한다 (inject_mode
="set", 기본):

    W(z)   = top-k 후보 방향의 성분별 합 → max|·| 정규화 → |w_j|<θ_w 제거
    seq[k] = Σ_j 1[U<p]·U(0.5,1.5)·w_j·e_j     (phase 스텝마다 독립 샘플)
    a'_t   = fit_to_box( a_demo(k) + feedback + λ·seq[k] )

즉 FCM 은 "어느 성분이 subgoal 을 방해하는가"(집합+부호)만 주고, 스텝마다
어떤 성분이 얼마나 들어갈지는 랜덤이다 — step1 (+dx,0,+dz,0), step2
(0,0,+dz,2·drx) 처럼. seed 는 family 고정이라 rung 간에는 같은 패턴에 λ 만
커진다 (nested ladder). "random"(방향 1개 + 스텝 Bernoulli), "window"(Stage 7
창 결정적)는 ablation 용으로 남겨둔다.

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
                 inject_mode="set", p_inject=0.5):
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
    return float(sg.rho_raw(phase, traj["F"][i]))


# ===========================================================================
# SECTION 2 — λ_max: bracketing → binary search
# ===========================================================================
def find_lambda_max(runner, seq, *, lam0=0.5, lam_cap=4.0, n_bisect=3,
                    log=print):
    """task 실패의 경계를 찾는다.

    성공(λ) 이 단조라는 보장은 없다(파지 물리의 cliff) — bracketing 은 처음
    만나는 실패를, bisection 은 그 근방 경계를 잡을 뿐이고, 최종 판정은
    Stage 9 의 G8 (단조성·계단성) 이 한다. λ_cap 까지 실패가 없으면
    (λ_cap, 'no-cliff') 를 반환한다 — 순수 효율 family.
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
        return float(lam_cap), "no-cliff", probes
    for _ in range(n_bisect):
        mid = 0.5 * (lam_ok + lam_fail)
        traj, _ = runner.run(seq, mid)
        probes.append((mid, bool(traj["success"])))
        if traj["success"]:
            lam_ok = mid
        else:
            lam_fail = mid
    return float(lam_fail), "cliff", probes


# ===========================================================================
# SECTION 3 — 전체 조립
# ===========================================================================
LEVELS = [0.0, 0.25, 0.50, 0.75, 1.0]
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
    """(candidate-dict, inject_seq) 목록 — inject_mode 별 family 정의."""
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
                lam_max, kind, probes = find_lambda_max(
                    r, seq, lam0=args.lam0, lam_cap=args.lam_cap,
                    n_bisect=args.n_bisect)
                trajs = [zero_runs[did]]
                for lv in LEVELS[1:]:
                    traj, _ = r.run(seq, lv * lam_max)
                    trajs.append(traj)
                rhos = [rho_endpoint(sg, phase, t, r.zseg) for t in trajs]
                succ = [bool(t["success"]) for t in trajs]
                log(f"  {fid:<22} λ_max={lam_max:.2f} ({kind}, "
                    f"{len(probes)} probes)  ρ_end="
                    f"[{', '.join(f'{x:+.2f}' for x in rhos)}]  "
                    f"success={''.join('o' if s else 'x' for s in succ)}")
                families.append(dict(
                    family_id=fid, candidate=cand, demo_id=did, phase_id=phase,
                    lambda_max=lam_max, lambda_kind=kind,
                    lambda_levels=[lv * lam_max for lv in LEVELS],
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
            inject_mode=args.inject_mode, p_inject=args.p_inject)
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

    meta = dict(n_families=len(fams), levels=LEVELS, lam_cap=args.lam_cap,
                kp=args.kp, pos_scale=args.pos_scale, top_k=args.top_k,
                inject_mode=args.inject_mode, p_inject=args.p_inject,
                n_seeds=args.n_seeds, w_thresh=args.w_thresh,
                demos=[e["demo_id"] for e in entries])
    out = os.path.join(args.out_dir, "degradation.npz")
    save_families(out, fams, meta)
    print(f"\n[out] {out}  ({len(fams)} families × {len(LEVELS)} levels)")

    n_cliff = sum(1 for f in fams if f["lambda_kind"] == "cliff")
    print(f"[stat] cliff {n_cliff} / no-cliff {len(fams) - n_cliff}")
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
            ax.plot(LEVELS, dr, color=cmap(ci % 10), alpha=0.55, lw=1.4)
            for x, y, t in zip(LEVELS, dr, f["trajectories"]):
                ax.scatter(x, y, marker="o" if t["success"] else "x",
                           color=cmap(ci % 10), s=28,
                           zorder=3)
        for ci, cid in enumerate(cands):
            ax.plot([], [], color=cmap(ci % 10), label=cid)
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_xlabel("level (× λ_max)")
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
        for lv, t in zip(LEVELS, f["trajectories"]):
            e = np.asarray(t["eef"])
            ax.plot(*e.T, color=cmap(lv), lw=1.6 if lv == 0 else 1.1,
                    label=f"λ={t['lam']:.2f} "
                          f"{'o' if t['success'] else 'x'}")
        ax.set_title(f"{f['family_id']}  (phase {f['phase_id']}, "
                     f"λ_max={f['lambda_max']:.2f} {f['lambda_kind']})",
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

    print(f"λ_max={f['lambda_max']:.3f} ({f['lambda_kind']}), "
          f"probes={[(round(l, 2), s) for l, s in f['probes']]}")
    if f["lambda_kind"] != "cliff" or not (0 < f["lambda_max"] <= A.lam_cap):
        ok = False; print("[FAIL] cliff 를 찾아야 한다 (+x 과주입 = 과녁 초과)")

    succ = [t["success"] for t in f["trajectories"]]
    if not (succ[0] and not succ[-1]):
        ok = False; print(f"[FAIL] rung 성공 패턴 {succ}: 0 성공/최상단 실패여야")
    print(f"rung success = {succ}")

    # 열화 단조성(대체로): 최종 goal 오차가 level 에 따라 비감소
    errs = [float(np.linalg.norm(np.asarray(t["obj"])[-1] - goal))
            for t in f["trajectories"]]
    n_viol = sum(1 for a, b in zip(errs, errs[1:]) if b < a - 1e-6)
    print(f"final goal err per rung = {[round(e, 3) for e in errs]} "
          f"(violations {n_viol})")
    if n_viol > 1:
        ok = False; print("[FAIL] 사다리가 goal 오차 기준으로 뒤죽박죽")

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
    ap.add_argument("--inject-mode", default="set",
                    choices=["set", "random", "window"],
                    help="set: phase 방해 성분 집합을 스텝별 랜덤 조합 (기본). "
                         "random: 후보 방향 1개를 스텝별 Bernoulli (ablation). "
                         "window: Stage 7 창에서 결정적 (ablation)")
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
