#!/usr/bin/env python
"""
rollout_exec.py — THE single definition of "what gets executed".
================================================================================

M2 (fcm.py) confirms candidate directions and M3 (degradation.py) builds the
lambda ladder. Both must execute the perturbation the SAME way, or M2 validates
something M3 never does. This module owns that execution so the two cannot
drift apart. It is the same argument that moved replay into frame_extract.py.

Why closed loop
---------------
Adding noise to the demo's recorded actions and replaying them open-loop does
not produce "the demo, but worse" -- it produces an incoherent trajectory. Once
the state drifts, every later demo action was computed for a state that no
longer holds, so "close the gripper" fires where there is no object. The ranking
label then separates coherent from incoherent rather than efficient from
inefficient, which is not what proposal Sec 8.1 is asking for.

So the perturbation is injected into a POLICY, exactly as D-REX/SSRR do. The
policy here is not a BC network -- it is a demo-tracking controller:

    a_t = a_demo[k]  +  (kp / pos_scale) * (eef_demo[k] - eef_now)
          \\_________/    \\_______________________________________/
           feedforward                 state feedback

That is state-conditioned, which is the property that matters. With one demo a
BC net would be a lookup table with error; this tracker has none and reproduces
the demo exactly at lambda = 0.

The feedback is on a[:3] ONLY
-----------------------------
Measured on PickPlaceCan/Panda, injecting delta=0.5 along each action axis and
comparing final eef deviation open- vs closed-loop:

    dx  0.461 -> 0.006   (98.6% cancelled)      drx 0.355 -> 0.378  (not cancelled)
    dy  0.307 -> 0.006   (98.1%)                dry 0.171 -> 0.161  (5.8%)
    dz  0.596 -> 0.007   (98.9%)                drz 0.155 -> 0.011  (92.6%)
                                                grip 0.031 -> 0.025 (20.2%)

The tracker corrects position with gain kp/pos_scale = 80 and does nothing about
wrist rotation, so position-dominant directions are ~98% neutralised while
rotation survives. A screening step that ranks directions by open-loop effect
will therefore put the cancelled ones on top. measure_attenuation() below turns
that into a per-axis weight so the search can account for it.

    python rollout_exec.py --selftest        # mock plant; no robosuite
"""

import argparse

import numpy as np

import feature_select as fs


# ===========================================================================
# SECTION 1 — steppers
# ===========================================================================
class SimStepper:
    """robosuite env as reset()/step(a) -> frame dict.

    Closed loop: every step actually advances the simulator and returns the TRUE
    resulting state, so the controller sees its own error.
    """

    def __init__(self, env, states, object_type=None, extractor=None):
        self.env, self.states, self.ot = env, states, object_type
        if extractor is None:
            import frame_extract as fx
            extractor = fx.FrameExtractor(env, object_type)
        self.ex = extractor

    def _read(self):
        f = self.ex._frame(with_dynamics=True)
        # keys kept in the legacy shape fcm/degradation already pass around
        return dict(eef=f["eef_pos"], obj=f["obj_pos"], grip=f["grip"],
                    eef_quat=f["eef_quat"], obj_quat=f["obj_quat"],
                    contact=f["contact"], qvel=f.get("qvel"),
                    torque=f.get("torque"))

    def reset(self):
        self.env.sim.set_state_from_flattened(self.states[0])
        self.env.sim.forward()
        return self._read()

    def step(self, a):
        self.env.step(a)
        return self._read()

    def success(self):
        try:
            return bool(self.env._check_success())
        except Exception:
            return None


class MockStepper:
    """Toy integrator for the selftest: eef += 0.01*a[:3]; the object follows the
    eef while grasped. Rotation dims move nothing, which is deliberate -- it
    makes the selftest's attenuation profile trivially checkable."""

    def __init__(self, T=200, adim=7, goal=np.array([0.5, 0.0, 0.85])):
        self.T, self.adim, self.goal = T, adim, np.asarray(goal, float)

    def reset(self):
        self.k = 0
        self.eef = np.array([0.0, 0.0, 1.0])
        self.obj = np.array([0.25, 0.0, 0.85])
        return self._read()

    def _grasped(self):
        return (self.k > self.T * 0.3) and (self.k < self.T * 0.85)

    def _read(self):
        g = 1.0 if self._grasped() else 0.0
        q = np.array([0.0, 0.0, 0.0, 1.0])
        return dict(eef=self.eef.copy(), obj=self.obj.copy(), grip=g,
                    eef_quat=q, obj_quat=q, contact=g, qvel=None, torque=None)

    def step(self, a):
        prev = self.eef.copy()
        self.eef = self.eef + 0.01 * np.asarray(a, float)[:3]
        if self._grasped():
            self.obj = self.obj + (self.eef - prev)
        self.k += 1
        return self._read()

    def success(self):
        return bool(np.linalg.norm(self.obj - self.goal) < 0.1)


# ===========================================================================
# SECTION 2 — the execution
# ===========================================================================
def in_phase(z_demo, k, phase, family="single"):
    if family == "all":
        return True
    return int(z_demo[min(k, len(z_demo) - 1)]) == phase


def fit_to_box(a, delta, tol=1e-12):
    """Largest s in [0, 1] with a + s*delta inside [-1, 1] on every axis.

    The alternative, clip(a + delta), does not shrink a perturbation -- it
    ROTATES it. Measured: with an action sitting at +-1 on two axes, a request
    for delta along [0.50, 0.50, 0.70, 0, 0, 0, 0] executes as a pure +z push,
    45 degrees off; across many draws the cosine between requested and executed
    falls from 0.999 at clip < 0.1 to 0.41 at clip > 0.6. Worse, how far it
    rotates depends on |delta|, so a lambda ladder meant to scale ONE direction
    quietly turns it, and proposal Sec 8.2's "same family, increasing magnitude"
    stops being true.

    Scaling instead keeps the direction EXACT at every lambda and every step;
    only the magnitude adapts to whatever room the action box leaves. The
    returned s is also the honest version of the "lower bound" reading: the
    executed perturbation really is s/1.0 of the one requested, along the very
    same vector.

    Note this cannot be pre-computed from the demo's actions: the perturbation
    lands on demo_action PLUS the tracker's feedback term, which at gain 10 adds
    up to 1.00 for a 100 mm error. That is why the feasibility has to be decided
    here, per step, on the action actually about to be sent.
    """
    a = np.asarray(a, float)
    delta = np.asarray(delta, float)
    s = 1.0
    for i in range(len(delta)):
        if delta[i] > tol:
            s = min(s, (1.0 - a[i]) / delta[i])
        elif delta[i] < -tol:
            s = min(s, (-1.0 - a[i]) / delta[i])
    return float(max(s, 0.0))


def closed_loop_rollout(stepper, demo_eef, demo_actions, z_demo, phase, D_z,
                        lam, adim, *, kp=0.8, pos_scale=0.01, p_base=0.15,
                        amp_base=0.5, max_stretch=2.5, family="single",
                        seed=0, stall=True, perturb_mode="scale",
                        inject_mask=None, p_cap=0.9, inject_seq=None):
    """Run the demo-tracking policy closed-loop, injecting the phase's
    anti-subgoal directions. lambda = 0 reproduces the demo exactly.

    Both the injection RATE and the injection SIZE scale with lambda:

        p(lam)   = min(p_cap, p_base * lam)      how often
        eps(lam) = amp_base * lam                how hard

    and a perturbed step STALLS -- the demo waypoint index k does not advance --
    so a degraded run also takes longer, which is itself an inefficiency the
    reward can see. max_stretch caps that so a badly damaged run terminates.

    PIPELINE_v4 Stage 8 mode (degradation.py 가 쓴다):
      inject_mask : (Tdemo,) bool — 데모 인덱스 k 가 이 마스크 안일 때만 주입.
                    action_sets 의 start/duration_fraction 창을 그대로 표현하며,
                    주어지면 z_demo/phase/family 대신 이것이 판정한다.
      p_base 를 크게 + p_cap=1.0 + stall=False + amp_base=1.0 으로 부르면
      v4 의 결정적 주입  a'_t = fit( a_demo + λ·mask(k)·d )  이 된다.

      inject_seq : (Tdemo, adim) — 스텝별 주입 델타를 미리 계산해 넘기는 모드.
                   주어지면 D_z/inject_mask/p 게이트를 전부 무시하고 데모 인덱스
                   k 에서  delta = amp_base·λ·inject_seq[k]  를 결정적으로 더한다
                   (0 행이면 주입 없음). 랜덤성(성분 포함 여부·스케일)은 호출자가
                   시퀀스에 미리 구워 넣는다 — 시드를 family 에 고정하면 λ 만
                   커지는 nested-ladder 성질이 그대로 유지된다.

    perturb_mode:
      "scale" (default) shrink the step to fit the action box, keeping the
              direction exact -- see fit_to_box for why this matters.
      "clip"  the old behaviour: clip(a + delta), which rotates the direction by
              a lambda-dependent amount.

    Returns (frames, A_exec, info). `frames` uses the legacy key names
    (eef/obj/grip/eef_quat/obj_quat/contact) that fcm.py and degradation.py
    already pass around.
    """
    rng = np.random.default_rng(seed)
    Tdemo = len(demo_actions)
    Dz = np.atleast_2d(np.asarray(D_z, float))
    p = min(p_cap, p_base * lam)
    eps = amp_base * lam
    max_steps = max(Tdemo + 1, int(Tdemo * max_stretch))

    fr = stepper.reset()
    eef, obj, grip = [fr["eef"]], [fr["obj"]], [fr["grip"]]
    eq, oq, con = [fr["eef_quat"]], [fr["obj_quat"]], [fr["contact"]]
    A_exec = []
    k_index = [0]              # which DEMO waypoint each executed frame sits on
    k, steps, n_noise = 0, 0, 0
    cur = fr["eef"]
    clip_amt = 0.0
    realised = []              # s per injection: how much of eps actually ran

    while k < Tdemo and steps < max_steps:
        # feedforward demo action + position feedback (a[:3] only -- see header)
        a = np.asarray(demo_actions[k], float).copy()
        a[:3] = a[:3] + kp * (np.asarray(demo_eef[k], float) - cur) / pos_scale

        stalled = False
        delta = None
        if inject_seq is not None:
            row = np.asarray(inject_seq[min(k, len(inject_seq) - 1)], float)
            if lam > 0.0 and np.any(row):
                delta = amp_base * lam * row
        else:
            in_win = (bool(inject_mask[min(k, len(inject_mask) - 1)])
                      if inject_mask is not None
                      else in_phase(z_demo, k, phase, family))
            if lam > 0.0 and in_win and rng.random() < p:
                delta = eps * Dz[rng.integers(len(Dz))]
        if delta is not None:
            clip_amt = max(clip_amt, fs.clip_fraction(a, delta, adim))
            if perturb_mode == "scale":
                # Shrink to fit; the direction is preserved exactly.
                s = fit_to_box(a, delta)
                realised.append(s)
                a = np.clip(a + s * delta, -1.0, 1.0)   # clip is a no-op guard
            else:
                realised.append(1.0)
                a = fs.perturb_action(a, delta, adim, proper_rotation=True)
            n_noise += 1
            stalled = stall
        else:
            a = np.clip(a, -1.0, 1.0)

        fr = stepper.step(a)
        cur = fr["eef"]
        A_exec.append(a)
        eef.append(fr["eef"]); obj.append(fr["obj"]); grip.append(fr["grip"])
        eq.append(fr["eef_quat"]); oq.append(fr["obj_quat"]); con.append(fr["contact"])
        steps += 1
        if not stalled:
            k += 1
        k_index.append(min(k, Tdemo - 1))

    frames = dict(eef=np.array(eef), obj=np.array(obj), grip=np.array(grip),
                  eef_quat=np.array(eq), obj_quat=np.array(oq),
                  contact=np.array(con))
    info = dict(ok=True, success=stepper.success(), length=steps + 1,
                n_noise=n_noise, reached_end=(k >= Tdemo), clip=float(clip_amt),
                lam=float(lam), phase=int(phase),
                # How much of the requested eps actually executed, along the
                # EXACT requested direction. Under perturb_mode="scale" this is
                # the number to read: `clip` only says how far outside the box
                # the request fell, `realised` says how much of it ran. 1.0 =
                # nothing lost. Callers must NOT default this to 1.0 when it is
                # absent -- that silently reports success for a run that was
                # never measured.
                realised=float(np.mean(realised)) if realised else 1.0,
                realised_min=float(np.min(realised)) if realised else 1.0,
                perturb_mode=perturb_mode,
                # k_index lets a caller recover "which demo phase was I in" for
                # every executed frame. Necessary because stalls stretch time:
                # frame t of a degraded run and frame t of the demo are no longer
                # the same moment in the task, so comparing them index-by-index
                # measures the time shift instead of the damage. Re-segmenting
                # the degraded run is not an option either -- a badly damaged run
                # can lose the phase entirely, and then the metric silently
                # changes what it refers to exactly when the damage is largest.
                k_index=np.array(k_index, int))
    return frames, (np.array(A_exec) if A_exec else np.zeros((0, adim))), info


def open_loop_rollout(stepper, demo_actions, z_demo, phase, delta, adim, *,
                      family="single"):
    """Demo actions + a FIXED delta on the phase's own steps, no feedback.

    This is what fcm's H-step counterfactual uses: for a short horizon the delta
    is exactly identified because nothing else moves the action. It is NOT what
    M3 executes, so it must never be used to confirm a direction -- that is the
    mistake this module exists to prevent. Kept because measure_attenuation
    needs an un-corrected baseline to divide by.
    """
    fr = stepper.reset()
    eef = [fr["eef"]]
    clip_amt = 0.0
    for k in range(len(demo_actions)):
        a = np.asarray(demo_actions[k], float).copy()
        if in_phase(z_demo, k, phase, family):
            clip_amt = max(clip_amt, fs.clip_fraction(a, delta, adim))
            a = fs.perturb_action(a, delta, adim, proper_rotation=True)
        else:
            a = np.clip(a, -1.0, 1.0)
        fr = stepper.step(a)
        eef.append(fr["eef"])
    return np.array(eef), dict(clip=float(clip_amt))


# ===========================================================================
# SECTION 3 — attenuation profile
# ===========================================================================
def measure_attenuation(make_stepper, demo_eef, demo_actions, z_demo, phase,
                        adim, *, mag=0.5, kp=0.8, pos_scale=0.01, floor=0.02):
    """Per-action-axis: how much of an open-loop effect survives the tracker.

    Returns w in [floor, ~1] of length adim. w[i] ~ 0 means "the controller
    undoes this axis"; w[i] ~ 1 means "this axis goes through".

    Costs adim + 2 rollouts, ONCE per phase. Multiply a candidate direction by w
    before scoring it and the search stops nominating directions that M3 will
    neutralise. The floor keeps a fully-cancelled axis from being zeroed out
    entirely -- position perturbations still knock the object about during the
    perturbed step itself, which the tracker cannot undo after the fact.
    """
    zero = np.zeros(adim)

    def run_ol(d):
        e, _ = open_loop_rollout(make_stepper(), demo_actions, z_demo, phase, d, adim)
        return e[-1]

    def run_cl(d):
        Dz = np.atleast_2d(d if np.linalg.norm(d) > 0 else zero)
        fr, _, _ = closed_loop_rollout(
            make_stepper(), demo_eef, demo_actions, z_demo, phase, Dz,
            lam=(0.0 if np.linalg.norm(d) == 0 else 1.0), adim=adim,
            kp=kp, pos_scale=pos_scale, p_base=1.0,
            amp_base=float(np.linalg.norm(d)), stall=False, seed=0)
        return fr["eef"][-1]

    base_ol, base_cl = run_ol(zero), run_cl(zero)
    w = np.ones(adim)
    for i in range(adim):
        d = np.zeros(adim); d[i] = mag
        ol = float(np.linalg.norm(run_ol(d) - base_ol))
        cl = float(np.linalg.norm(run_cl(d / mag * mag) - base_cl))
        w[i] = 1.0 if ol < 1e-9 else np.clip(cl / ol, floor, 1.5)
    return w


# ===========================================================================
# SECTION 4 — selftest
# ===========================================================================
def _mock_demo(T=120, adim=7):
    """A straight-line reach-and-carry demo for the mock plant.

    The feedforward action is 0.5, not 1.0, on purpose: perturb_action clips to
    [-1, 1], so a demo that already commands the limit absorbs every positive
    perturbation and the axis looks (wrongly) unaffected. Leaving headroom is
    what makes the attenuation measurement mean anything.
    """
    A = np.zeros((T, adim))
    A[:, 0] = 0.5                      # push +x with headroom for perturbation
    A[:, 6] = np.where(np.arange(T) > T * 0.3, 1.0, -1.0)
    st = MockStepper(T=T, adim=adim)
    fr = st.reset()
    eef = [fr["eef"]]
    for t in range(T):
        fr = st.step(A[t]); eef.append(fr["eef"])
    z = np.zeros(T, dtype=int); z[T // 2:] = 1
    return np.array(eef), A, z


def run_selftest():
    print("=== rollout_exec SELFTEST (mock plant; no robosuite) ===")
    ok = True
    adim = 7
    demo_eef, demo_A, z = _mock_demo(adim=adim)
    mk = lambda: MockStepper(T=len(demo_A), adim=adim)
    D_z = np.eye(adim)[[0]]                     # push +x harder

    # 1. lambda = 0 must reproduce the demo. If not, every rung above is sand.
    fr0, A0, i0 = closed_loop_rollout(mk(), demo_eef, demo_A, z, 0, D_z, 0.0, adim)
    err = float(np.abs(fr0["eef"][:len(demo_eef)] - demo_eef[:len(fr0["eef"])]).max())
    print(f"[1] lam=0 reproduces demo: max eef err = {err:.6f}  "
          f"len={i0['length']} (demo {len(demo_A)+1})")
    if err > 1e-6:
        ok = False; print("    [FAIL] lam=0 must be the demo exactly")

    # 2. length grows with lambda (the stall mechanism), bounded by max_stretch.
    lens = []
    for lam in [0.0, 0.5, 1.0, 2.0, 4.0]:
        _, _, inf = closed_loop_rollout(mk(), demo_eef, demo_A, z, 0, D_z, lam,
                                        adim, seed=1)
        lens.append(inf["length"])
    grows = all(b >= a for a, b in zip(lens, lens[1:]))
    cap = int(len(demo_A) * 2.5) + 1
    print(f"[2] lengths {lens} monotone={grows} bounded<={cap}: "
          f"{'OK' if grows and max(lens) <= cap else 'X'}")
    ok &= grows and max(lens) <= cap

    # 3. closed loop must curb drift that open loop lets run away.
    d = np.zeros(adim); d[0] = 0.5
    e_ol, _ = open_loop_rollout(mk(), demo_A, z, 0, d, adim)
    fr_cl, _, _ = closed_loop_rollout(mk(), demo_eef, demo_A, z, 0,
                                      np.atleast_2d(d), 1.0, adim,
                                      p_base=1.0, amp_base=1.0, stall=False)
    dev_ol = float(np.linalg.norm(e_ol[-1] - demo_eef[-1]))
    dev_cl = float(np.linalg.norm(fr_cl["eef"][-1] - demo_eef[-1]))
    print(f"[3] final drift  open-loop={dev_ol:.4f}  closed-loop={dev_cl:.4f}  "
          f"-> feedback curbs it: {'OK' if dev_cl < dev_ol else 'X'}")
    ok &= dev_cl < dev_ol

    # 4. the attenuation profile: on this plant only a[:3] moves the eef, and
    #    the tracker feeds back on exactly those, so they must be attenuated
    #    while the rotation axes (which move nothing here) stay at 1.
    w = measure_attenuation(mk, demo_eef, demo_A, z, 0, adim)
    lab = ["dx", "dy", "dz", "drx", "dry", "drz", "grip"]
    print("[4] attenuation w = " + "  ".join(f"{l}:{v:.3f}" for l, v in zip(lab, w)))
    if not (w[0] < 0.5):
        ok = False; print("    [FAIL] dx should be attenuated by the tracker")
    if not np.allclose(w[3:6], 1.0):
        ok = False; print("    [FAIL] axes the plant ignores should stay at 1.0")

    # 5. an all-zero direction must not move the trajectory. It still triggers
    #    stalls, so the run gets LONGER -- comparing frame-by-frame against the
    #    demo would fail on the time shift alone, so compare where it ENDS.
    fr_z, _, inf_z = closed_loop_rollout(mk(), demo_eef, demo_A, z, 0,
                                         np.zeros((1, adim)), 4.0, adim, seed=2)
    end_err = float(np.linalg.norm(fr_z["eef"][-1] - demo_eef[-1]))
    print(f"[5] zero direction @lam=4 -> final eef err {end_err:.6f}, "
          f"len {inf_z['length']} vs demo {len(demo_A)+1} "
          f"({inf_z['n_noise']} stalls)")
    if end_err > 1e-3:
        ok = False
        print("    [FAIL] a zero direction changed the trajectory; the tracker "
              "should absorb a stall")

    # 6. realised must be REPORTED, not silently defaulted. A caller reading
    #    info.get("realised", 1.0) on a dict that lacks the key gets 1.0 and
    #    reports a perfect run for something never measured -- which is exactly
    #    what happened once, hiding every shrunken step behind a fake 1.00.
    if "realised" not in i0:
        ok = False
        print("[6] [FAIL] info has no 'realised' key -- callers will default it "
              "to 1.0 and report a perfect run for something unmeasured")
    else:
        # fit_to_box directly: the rollout's own action is demo + feedback, and
        # the feedback pulls it back off the limit, so a demo pinned at +1 does
        # NOT reach the box edge once the controller has had its say. Test the
        # primitive on the actual saturated inputs instead.
        a_sat = np.ones(adim)
        s_block = fit_to_box(a_sat, np.eye(adim)[0] * 1.0)      # +x into +1
        s_free = fit_to_box(a_sat, -np.eye(adim)[0] * 1.0)      # -x, room to go
        s_half = fit_to_box(np.zeros(adim), np.eye(adim)[0] * 2.0)  # wants 2, fits 1
        print(f"[6] fit_to_box: into a saturated axis -> {s_block:.3f} (expect 0), "
              f"away from it -> {s_free:.3f} (expect 1), "
              f"2x oversized -> {s_half:.3f} (expect 0.5)")
        bad = (abs(s_block) > 1e-9 or abs(s_free - 1.0) > 1e-9
               or abs(s_half - 0.5) > 1e-9)
        if bad:
            ok = False; print("    [FAIL] fit_to_box is wrong")
        # and the rollout must report it
        _, _, i_r = closed_loop_rollout(mk(), demo_eef, demo_A, z, 0,
                                        np.eye(adim)[[2]], 1.0, adim,
                                        p_base=1.0, seed=3)
        print(f"    rollout reports realised={i_r['realised']:.3f} "
              f"(free +z axis, expect 1.0)")
        if i_r["realised"] < 0.99:
            ok = False; print("    [FAIL] a free axis should execute in full")

    print(f"\n[selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--selftest", action="store_true")
    ap.parse_args()
    run_selftest()


if __name__ == "__main__":
    main()