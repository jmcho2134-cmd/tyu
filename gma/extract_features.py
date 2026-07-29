#!/usr/bin/env python
"""
extract_features.py — Stage 3: demo.hdf5 -> cache/demo_XXX.npz
================================================================================

★ robosuite가 필요한 유일한 지점. 이 스테이지가 성공하면 Stage 4 이후는 전부
순수 numpy로 돈다 (PIPELINE_v4.md Stage 3).

    read_demo -> build_env -> reset_to_scene -> FrameExtractor.from_states
              -> feature_select.compute_from_frames -> cache/demo_XXX.npz

Cache layout (PIPELINE_v4.md Sec 3.2) — 이후 스테이지가 읽는 유일한 계약:

    F        : (T, N_FEATURES) float64   # feature_select.NAMES 순서 고정
    names    : (N_FEATURES,)   str
    actions  : (T, adim)       float64
    goal     : (3,)            float64
    dt       : float                      # 1 / control_freq
    demo_id  : str                        # demo_000, demo_001, ... 발견 순서
    meta     : json str                   # 진단용 (goal_source, replay_err, ...)

Ψ는 저장하지 않는다: F[:, [names.index(n) for n in fs.PSI_COLUMNS]] 로 복원되고
역방향은 불가능하기 때문 (Sec 3.2).

GATE G2 — state 복원 오차 <= tol (기본 1e-6)
--------------------------------------------
set_state_from_flattened + forward 후 sim.get_state().flatten() 을 저장된 state와
비교한다 (time 원소 제외). 이 왕복이 어긋나면 리플레이로 재생성한 관측 전체를
믿을 수 없으므로, 문서대로 "이후 전부 중단" — 기본값은 SystemExit.
(주의: forward() 는 qpos/qvel 을 바꾸지 않으므로 이 오차는 정상일 때 ~0.
 0이 아니라면 scene XML 불일치나 차원 불일치가 원인이다.)

Entry points
------------
    extract(demo_root, cache_dir)   실데모 추출 (robosuite 필요)
    synth(cache_dir, n_demos)       구조가 같은 합성 데모 캐시 생성 (numpy만.
                                    Stage 4/5 를 실데모 없이 개발/검증하는 용도)
    load_cache(cache_dir)           캐시 로드 -> [entry, ...]  ★ Stage 4 의 입력

CLI::

    python extract_features.py --demo-root ./demos --cache-dir ./cache   # 추출
    python extract_features.py --synth --n-demos 12                      # 합성
    python extract_features.py --selftest                                # numpy만
"""

import argparse
import json
import os
from glob import glob

import numpy as np

import feature_select as fs

CACHE_PATTERN = "demo_*.npz"


# ===========================================================================
# SECTION 1 — cache I/O  (여기 두 함수가 Stage 3 <-> Stage 4 의 계약 전부)
# ===========================================================================
def save_cache(cache_dir, demo_id, F, actions, goal, dt, meta=None,
               eef_pos=None, obj_pos=None):
    """One demo -> cache/<demo_id>.npz. Returns the path.

    eef_pos/obj_pos 는 계약(Sec 3.2) 밖의 선택 채널: 3D 궤적 시각화용 raw
    좌표. 없어도 이후 스테이지는 전부 동작한다."""
    F = np.asarray(F, float)
    if F.shape[1] != fs.N_FEATURES:
        raise ValueError(f"F has {F.shape[1]} columns, schema says {fs.N_FEATURES}")
    if not np.isfinite(F).all():
        bad = [fs.NAMES[j] for j in np.unique(np.argwhere(~np.isfinite(F))[:, 1])]
        raise ValueError(f"{demo_id}: non-finite feature column(s): {bad}")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{demo_id}.npz")
    extra = {}
    if eef_pos is not None:
        extra["eef_pos"] = np.asarray(eef_pos, float)
    if obj_pos is not None:
        extra["obj_pos"] = np.asarray(obj_pos, float)
    np.savez(
        path,
        F=F,
        names=np.array(fs.NAMES),
        actions=np.asarray(actions, float),
        goal=np.asarray(goal, float).ravel(),
        dt=float(dt),
        demo_id=demo_id,
        meta=json.dumps(meta or {}),
        **extra,
    )
    return path


def load_cache(cache_dir):
    """cache/demo_*.npz -> list of dict entries, demo_id 순 정렬.

    entry = {demo_id, F, names, actions, goal, dt, meta}
    names 가 현재 스키마(fs.NAMES)와 다르면 즉시 실패한다: 스키마가 바뀐 뒤
    남은 낡은 캐시를 조용히 섞어 쓰는 것이 최악의 실패 모드이기 때문.
    """
    paths = sorted(glob(os.path.join(cache_dir, CACHE_PATTERN)))
    if not paths:
        raise FileNotFoundError(f"no {CACHE_PATTERN} under {cache_dir}")
    entries = []
    for p in paths:
        z = np.load(p, allow_pickle=False)
        names = [str(n) for n in z["names"]]
        if names != fs.NAMES:
            raise ValueError(
                f"{p}: cached feature names differ from the current schema.\n"
                f"  cached : {names}\n  schema : {fs.NAMES}\n"
                f"  -> re-run extraction (stale cache)")
        entries.append(dict(
            demo_id=str(z["demo_id"]),
            F=z["F"],
            names=names,
            actions=z["actions"],
            goal=z["goal"],
            dt=float(z["dt"]),
            meta=json.loads(str(z["meta"])),
            eef_pos=z["eef_pos"] if "eef_pos" in z else None,
            obj_pos=z["obj_pos"] if "obj_pos" in z else None,
        ))
    return entries


def psi_of(entry):
    """entry -> Ψ (T, len(PSI_COLUMNS)). 저장하지 않고 열 부분집합으로 복원."""
    idx = [entry["names"].index(n) for n in fs.PSI_COLUMNS]
    return entry["F"][:, idx]


# ===========================================================================
# SECTION 2 — GATE G2: state 복원 오차
# ===========================================================================
def replay_error(env, states, n_probe=16):
    """Max |round-trip state - saved state| over n_probe evenly spaced frames
    (time 원소는 제외). 정상이면 float 정밀도 수준."""
    states = np.asarray(states, float)
    idx = np.unique(np.linspace(0, len(states) - 1,
                                min(n_probe, len(states))).astype(int))
    worst = 0.0
    for t in idx:
        env.sim.set_state_from_flattened(states[t])
        env.sim.forward()
        back = np.asarray(env.sim.get_state().flatten(), float)
        if back.shape != states[t].shape:
            raise RuntimeError(
                f"state dim mismatch: saved {states[t].shape} vs sim "
                f"{back.shape} — scene XML and demo do not match")
        worst = max(worst, float(np.max(np.abs(back[1:] - states[t, 1:]))))
    return worst


# ===========================================================================
# SECTION 3 — extract(): 실데모 (robosuite 필요한 유일한 경로)
# ===========================================================================
def discover_demos(demo_root, pattern="demo.hdf5"):
    return sorted(glob(os.path.join(demo_root, "**", pattern), recursive=True))


def extract(demo_root, cache_dir, *, tol=1e-6, torque_mode="auto",
            enforce_gate=True, pattern="demo.hdf5"):
    """demos/**/demo.hdf5 전부 -> cache/demo_XXX.npz.

    demo_id 는 발견(정렬) 순서로 demo_000 부터 부여한다. Stage 4 의
    boundaries.json 이 이 id 를 그대로 키로 쓴다.
    """
    from frame_extract import read_demo, build_env, reset_to_scene, FrameExtractor

    paths = discover_demos(demo_root, pattern)
    if not paths:
        raise SystemExit(f"no '{pattern}' under {demo_root}")

    env, env_key, extractor = None, None, None
    results, gate_fail = [], []
    counter = 0
    try:
        for hp in paths:
            env_info, demos = read_demo(hp)
            robots = env_info["robots"]
            key = (env_info["env_name"],
                   tuple(robots) if isinstance(robots, list) else robots,
                   env_info.get("control_freq", 20))
            if key != env_key:
                if env is not None:
                    env.close()
                print(f"[env] building {key[0]} / {key[1]} @ {key[2]} Hz")
                env = build_env(env_info)
                env_key = key
                extractor = None
            dt = 1.0 / env_info.get("control_freq", 20)

            for name, states, actions, xml, torques in demos:
                demo_id = f"demo_{counter:03d}"
                counter += 1
                reset_to_scene(env, xml)
                # reset_from_xml_string 이 sim 을 재구성하므로 extractor 는
                # scene 로드 후에 만들어야 obj_model/obj_body 참조가 유효하다.
                extractor = FrameExtractor(env, env_info.get("object_type"),
                                           dt=dt)

                err = replay_error(env, states)
                if err > tol:
                    gate_fail.append((demo_id, err))
                    print(f"[G2 FAIL] {demo_id}: replay error {err:.3e} > tol {tol:.1e}")

                fr = extractor.from_states(states, actions, torques,
                                           torque_mode=torque_mode)
                F = fs.compute_from_frames(fr)

                if actions is not None and len(actions) != len(states):
                    print(f"[warn] {demo_id}: {len(actions)} actions vs "
                          f"{len(states)} states")

                meta = dict(
                    source=hp, source_demo=name,
                    env_name=env_info["env_name"],
                    robots=robots if isinstance(robots, list) else [robots],
                    object_type=env_info.get("object_type"),
                    action_dim=int(np.asarray(actions).shape[1]) if actions is not None else None,
                    goal_source=fr.goal_source,
                    torque_source=fr.torque_source,
                    replay_err=err,
                )
                p = save_cache(cache_dir, demo_id, F, actions, fr.goal, dt, meta,
                               eef_pos=fr.eef_pos, obj_pos=fr.obj_pos)
                results.append(p)
                print(f"[ok] {demo_id}: T={len(F)}  replay_err={err:.2e}  "
                      f"goal[{fr.goal_source}]  torque[{fr.torque_source}]  -> {p}")
    finally:
        if env is not None:
            env.close()

    # ---- action layout 일관성 (G1 의 절반: layout 부분) ----------------------
    dims = set()
    for p in results:
        z = np.load(p, allow_pickle=False)
        dims.add(int(z["actions"].shape[1]))
    if len(dims) > 1:
        print(f"[warn] inconsistent action dims across demos: {sorted(dims)}")

    print(f"\n[extract] {len(results)} demos cached in {cache_dir}")
    if gate_fail:
        msg = ", ".join(f"{d}({e:.1e})" for d, e in gate_fail)
        if enforce_gate:
            raise SystemExit(f"[G2] state replay gate FAILED: {msg} — "
                             f"이후 스테이지 진행 금지 (PIPELINE_v4.md Sec 5)")
        print(f"[G2] FAILED (not enforced): {msg}")
    else:
        print(f"[G2] PASS: all replay errors <= {tol:.1e}")
    return results


# ===========================================================================
# SECTION 4 — synth(): 합성 데모 (numpy만; Stage 4/5 개발·검증용)
# ===========================================================================
def synth_frames(rng, T=None, dt=0.05):
    """구조가 실데모와 같은 합성 pick-place 한 편.

    approach -> grasp -> transport -> place -> retreat 5-phase 구조를
    경계 시점·시작 위치·goal·속도 프로파일 모두 데모마다 지터해서 만든다.
    Stage 5 의 cross-demo 분산비(Var_i/Var_{i,t})가 의미를 가지려면 데모 간
    변주가 실제로 있어야 하기 때문.
    """
    from frame_extract import Frames

    T = int(T if T is not None else rng.integers(90, 150))
    # phase 경계 (fraction of T, 지터 포함)
    fr_ = np.array([0.30, 0.42, 0.70, 0.82]) + rng.normal(0, 0.02, 4)
    b1, b2, b3, b4 = (np.clip(fr_, 0.1, 0.95) * T).astype(int)

    p0 = np.array([0.25, 0.00, 0.85]) + rng.normal(0, 0.01, 3)   # 객체 시작
    g = np.array([0.50, 0.20, 0.90]) + rng.normal(0, 0.003, 3)   # goal (빈 중심)
    e0 = np.array([0.05, -0.05, 1.05]) + rng.normal(0, 0.02, 3)  # eef 시작

    eef = np.zeros((T, 3))
    obj = np.zeros((T, 3))

    def ease(n):
        s = np.linspace(0, 1, n)
        return 3 * s**2 - 2 * s**3          # smoothstep: 끝점 속도 0

    # approach: eef -> 객체
    eef[:b1] = e0 + ease(b1)[:, None] * (p0 - e0)
    # grasp: 객체 위에 정지 (미세 노이즈)
    eef[b1:b2] = p0
    obj[:b2] = p0
    # transport: 들어 올려 goal 로 (포물선 lift)
    n = b3 - b2
    s = ease(n)
    arc = np.zeros((n, 3))
    arc[:, :2] = p0[None, :2] + s[:, None] * (g[None, :2] - p0[None, :2])
    lift = 0.12 + rng.uniform(0, 0.05)
    arc[:, 2] = p0[2] + s * (g[2] - p0[2]) + lift * np.sin(np.pi * s)
    eef[b2:b3] = arc
    obj[b2:b3] = arc                          # 파지 중: 함께 이동
    # place: goal 에 정지, 객체 내려놓음
    eef[b3:b4] = g
    obj[b3:] = g
    # retreat: eef 위로 후퇴
    n = T - b4
    away = g + np.array([0.0, 0.0, 0.15])
    eef[b4:] = g + ease(n)[:, None] * (away - g)

    eef += rng.normal(0, 0.0015, eef.shape)   # 텔레옵 손떨림
    obj[:b2] = p0                             # 놓인 물체는 흔들리지 않음
    obj[b3:] = g

    # gripper aperture proxy: open ~0.08 / closed ~0.02 (2-means 로 갈라짐)
    grip = np.full(T, 0.08) + rng.normal(0, 0.002, T)
    grip[b1 + (b2 - b1) // 2: b3 + (b4 - b3) // 2] = \
        0.02 + rng.normal(0, 0.002, (b3 + (b4 - b3) // 2) - (b1 + (b2 - b1) // 2))
    contact = np.zeros(T)
    contact[b1 + (b2 - b1) // 2: b3 + (b4 - b3) // 2] = 1.0

    # quats: approach 동안 wrist 가 정렬 (grasp_align 하강 후 plateau)
    ang0 = 0.6 + rng.normal(0, 0.05)
    ang = np.full(T, 0.05)
    ang[:b1] = 0.05 + (ang0 - 0.05) * (1 - ease(b1))
    eef_quat = np.stack([np.sin(ang / 2), np.zeros(T), np.zeros(T),
                         np.cos(ang / 2)], axis=1)
    obj_quat = np.tile([0.0, 0.0, 0.0, 1.0], (T, 1))

    # actions: 위치 델타 스케일 + gripper 채널 (adim=7)
    acts = np.zeros((T, 7))
    acts[1:, :3] = np.diff(eef, axis=0) / dt * 0.5
    acts[:, 6] = np.where(contact > 0, 1.0, -1.0)
    acts = np.clip(acts, -1, 1)

    return Frames(eef_pos=eef, obj_pos=obj, grip=grip, goal=g, dt=dt,
                  eef_quat=eef_quat, obj_quat=obj_quat, contact=contact,
                  actions=acts, goal_source="synthetic",
                  meta=dict(bounds=[int(b1), int(b2), int(b3), int(b4)]))


def synth(cache_dir, n_demos=12, seed=0, dt=0.05):
    """합성 데모 n_demos 편 -> cache. 실데모와 같은 계약으로 저장되므로
    Stage 4/5 를 이 캐시로 먼저 개발/검증할 수 있다."""
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(n_demos):
        fr = synth_frames(rng, dt=dt)
        F = fs.compute_from_frames(fr)
        demo_id = f"demo_{i:03d}"
        meta = dict(source="synthetic", seed=int(seed),
                    true_bounds=fr.meta["bounds"],
                    goal_source=fr.goal_source, torque_source=fr.torque_source)
        paths.append(save_cache(cache_dir, demo_id, F, fr.actions, fr.goal,
                                fr.dt, meta, eef_pos=fr.eef_pos,
                                obj_pos=fr.obj_pos))
    print(f"[synth] {n_demos} synthetic demos -> {cache_dir}")
    return paths


# ===========================================================================
# SECTION 5 — selftest (numpy만; robosuite 불필요)
# ===========================================================================
def run_selftest():
    import tempfile
    print("=== extract_features SELFTEST ===")
    ok = True
    tmp = tempfile.mkdtemp(prefix="extract_features_selftest_")

    # 1. synth -> load_cache 왕복
    n = 5
    synth(tmp, n_demos=n, seed=7)
    entries = load_cache(tmp)
    if len(entries) != n:
        ok = False; print(f"[FAIL] wrote {n}, loaded {len(entries)}")
    ids = [e["demo_id"] for e in entries]
    if ids != sorted(ids) or ids[0] != "demo_000":
        ok = False; print(f"[FAIL] demo_id ordering: {ids}")

    for e in entries:
        T = len(e["F"])
        if e["F"].shape != (T, fs.N_FEATURES):
            ok = False; print(f"[FAIL] {e['demo_id']} F shape {e['F'].shape}")
        if e["actions"].shape != (T, 7):
            ok = False; print(f"[FAIL] {e['demo_id']} actions {e['actions'].shape}")
        if e["goal"].shape != (3,):
            ok = False; print(f"[FAIL] {e['demo_id']} goal {e['goal'].shape}")
        if not np.isfinite(e["F"]).all():
            ok = False; print(f"[FAIL] {e['demo_id']} non-finite F")
    print(f"cache round-trip: {n} entries, F=(T,{fs.N_FEATURES}), "
          f"actions=(T,7), goal=(3,)")

    # 2. Ψ 복원 = boundary 열 부분집합
    e = entries[0]
    psi = psi_of(e)
    if psi.shape != (len(e["F"]), len(fs.PSI_COLUMNS)):
        ok = False; print(f"[FAIL] psi shape {psi.shape}")
    j = [e["names"].index(nm) for nm in fs.PSI_COLUMNS]
    if not np.allclose(psi, e["F"][:, j]):
        ok = False; print("[FAIL] psi_of != column subset")
    else:
        print(f"psi_of(entry) == F[:, PSI_COLUMNS]  {psi.shape}")

    # 3. 합성 데모가 태스크 구조를 갖는가 (Stage 4 가 볼 신호)
    d_og = e["F"][:, fs.index_of("object_goal_dist")]
    d_eo = e["F"][:, fs.index_of("eef_object_dist")]
    if not (d_og[0] > 0.15 and d_og[-1] < 0.02):
        ok = False; print(f"[FAIL] object_goal_dist not resolved: "
                          f"{d_og[0]:.3f} -> {d_og[-1]:.3f}")
    if not (d_eo[0] > 0.15 and d_eo.min() < 0.02):
        ok = False; print(f"[FAIL] eef never reached object")
    gr = e["F"][:, fs.index_of("gripper_open")]
    if np.std(gr) < 1e-3:
        ok = False; print("[FAIL] gripper signal has no transition")
    print(f"structure: obj->goal {d_og[0]:.3f}->{d_og[-1]:.3f}, "
          f"eef->obj min {d_eo.min():.4f}, grip 2-mode std {np.std(gr):.4f}")

    # 4. cross-demo 변주 (Stage 5 분산비의 전제)
    bounds = np.array([en["meta"]["true_bounds"] for en in entries], float)
    Ts = np.array([len(en["F"]) for en in entries])
    if np.all(np.std(bounds / Ts[:, None], axis=0) < 1e-4):
        ok = False; print("[FAIL] no cross-demo boundary variation")
    else:
        print(f"cross-demo variation: T in [{Ts.min()},{Ts.max()}], "
              f"bound-frac std {np.round(np.std(bounds / Ts[:, None], axis=0), 3)}")

    # 5. 낡은 스키마 캐시는 거부되는가
    p = os.path.join(tmp, "demo_099.npz")
    z = dict(np.load(os.path.join(tmp, "demo_000.npz"), allow_pickle=False))
    z["names"] = np.array(["bogus"] * fs.N_FEATURES)
    np.savez(p, **z)
    try:
        load_cache(tmp)
        ok = False; print("[FAIL] stale-schema cache was accepted")
    except ValueError:
        print("stale-schema cache -> ValueError (거부됨)")
    os.remove(p)

    print(f"\n[selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--demo-root", default="./demos")
    ap.add_argument("--cache-dir", default="./cache")
    ap.add_argument("--pattern", default="demo.hdf5")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="GATE G2: max allowed state replay error")
    ap.add_argument("--torque-mode", default="auto",
                    choices=["auto", "recorded", "inverse", "none"])
    ap.add_argument("--no-gate", action="store_true",
                    help="report G2 but do not abort on failure")
    ap.add_argument("--synth", action="store_true",
                    help="write synthetic demo caches instead of extracting")
    ap.add_argument("--n-demos", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(0 if run_selftest() else 1)
    if args.synth:
        synth(args.cache_dir, n_demos=args.n_demos, seed=args.seed)
        return
    extract(args.demo_root, args.cache_dir, tol=args.tol,
            torque_mode=args.torque_mode, enforce_gate=not args.no_gate,
            pattern=args.pattern)


if __name__ == "__main__":
    main()
