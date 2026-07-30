PS C:\Users\Administrator\Desktop\tyu-main\gma> python .\fcm.py train --hidden 256 256 --max-iter 2000
[G6] PASS: ρ_worst 공간 drift/effect 최대 0.075 (phase 0 [pre]) <= 0.3, anchor drop 5% <= 50%
     phase 0 [pre  ] drift_ρ=0.0072 effect_ρ=0.0955 ratio=0.075 (λ=0 736행 / λ>0 12752행)
           drift 분포: 중앙 0.00000  95% 0.0005  최대 0.0535
           > 0.15 인 행 0개 (0.00%) 제외 시 drift_ρ=0.00718 ratio=0.075
     phase 1 [move ] drift_ρ=0.0025 effect_ρ=1.0484 ratio=0.002 (λ=0 504행 / λ>0 8648행)
           drift 분포: 중앙 0.00026  95% 0.0015  최대 0.0235
           > 0.15 인 행 0개 (0.00%) 제외 시 drift_ρ=0.00255 ratio=0.002
     phase 2 [post ] drift_ρ=0.0023 effect_ρ=0.7415 ratio=0.003 (λ=0 576행 / λ>0 10072행)
           drift 분포: 중앙 0.00078  95% 0.0041  최대 0.0110
           > 0.15 인 행 0개 (0.00%) 제외 시 drift_ρ=0.00231 ratio=0.003
     [진단] raw feature 비율 (분모가 δ 무관 방향까지 포함해 희석됨): eef_object_dist=0.080, object_goal_dist=0.035, gripper_open=0.489, contact=0.000
  member 0: fitted (val loss 0.8330)
  member 1: fitted (val loss 0.8257)
  member 2: fitted (val loss 0.8517)
  member 3: fitted (val loss 0.8288)
  member 4: fitted (val loss 0.8475)
[train] heldout R^2 (rollout 단위 분할):
    eef_object_dist    +0.383
    object_goal_dist   +0.323
    gripper_open       +0.603
    contact            +0.346
    grasp_align        +0.539
    object_height      +0.405
    eef_speed          +0.820
    object_speed       +0.461
    action_magnitude   +0.898
    eef_accel          +0.773
    eef_jerk           +0.708
    object_slip        +0.687
    eef_ang_speed      +0.978
    object_ang_speed   +0.722
    energy               n/a
[plot] ./artifacts\fcm_fit_r2.png
[out] ./artifacts\fcm_ensemble.pkl
PS C:\Users\Administrator\Desktop\tyu-main\gma> python .\fcm.py screen
  phase 0 [pre]
    cone blocked: grip-
    rand2            drho=-0.044  unc=0.035  score=+0.009
    axis[dz+]        drho=-0.035  unc=0.027  score=+0.009
    (score <= 0 로 제외 13개)
    [warn] phase 0: 후보 2/4 개만 기준 통과 — 열화 방향이 부족하다 (FCM 재학습/λ_probe 상향 검토)
  phase 1 [move]
    cone blocked: grip+
    axis[grip-]      drho=-1.290  unc=0.294  score=+0.996
    rand2            drho=-0.601  unc=0.401  score=+0.200
    steepest         drho=-0.506  unc=0.308  score=+0.198
    axis[dz+]        drho=-0.420  unc=0.229  score=+0.190
  phase 2 [post]
    cone blocked: grip-
    rand1            drho=-1.133  unc=0.305  score=+0.829
    axis[dz+]        drho=-0.662  unc=0.118  score=+0.544
    rand4            drho=-0.721  unc=0.197  score=+0.524
    antipara         drho=-0.222  unc=0.054  score=+0.169
[out] ./artifacts\action_sets.json
[plot] ./artifacts\fcm_sets.png
[env] PickPlaceBread / ['UR5e']




PS C:\Users\Administrator\Desktop\tyu-main\gma> python -c "import json;d=json.load(open('artifacts/g7_diag.json'));print(json.dumps(d,indent=1,ensure_ascii=False))"
{
 "g7_pass": true,
 "mean_recall": 0.583,
 "mean_random": 0.4,
 "phases": {
  "phase_0": {
   "recall": 0.5,
   "random": 0.4,
   "n_pool": 10,
   "n_shipped": 2,
   "k_eff": 4,
   "measured_drho_worst": {
    "rand2": -0.0279,
    "axis[dz+]": -0.0537,
    "rand3": -0.0207,
    "rand1": -0.0321,
    "axis[dy+]": -0.0464,
    "axis[dry+]": -0.0682,
    "rand5": 0.0274,
    "rand4": -0.027,
    "axis[drz+]": -0.0023,
    "axis[dx+]": 0.0214
   },
   "shipped": [
    "axis[dz+]",
    "rand2"
   ]
  },
  "phase_1": {
   "recall": 0.5,
   "random": 0.4,
   "n_pool": 10,
   "n_shipped": 4,
   "k_eff": 4,
   "measured_drho_worst": {
    "axis[grip-]": 0.0002,
    "rand2": 0.0131,
    "steepest": -0.1105,
    "axis[dz+]": -0.1128,
    "rand0": -0.1168,
    "rand1": 0.027,
    "rand3": -0.026,
    "rand5": -0.0315,
    "axis[drx+]": 0.0148,
    "axis[dry+]": -0.0177
   },
   "shipped": [
    "axis[dz+]",
    "axis[grip-]",
    "rand2",
    "steepest"
   ]
  },
  "phase_2": {
   "recall": 0.75,
   "random": 0.4,
   "n_pool": 10,
   "n_shipped": 4,
   "k_eff": 4,
   "measured_drho_worst": {
    "rand1": -2.0897,
    "axis[dz+]": -1.3876,
    "rand4": -1.433,
    "antipara": -0.0814,
    "axis[grip+]": -0.4276,
    "rand0": -0.156,
    "rand5": -0.0836,
    "axis[drx+]": -0.1972,
    "axis[drz+]": 0.0412,
    "axis[dry+]": -0.3186
   },
   "shipped": [
    "antipara",
    "axis[dz+]",
    "rand1",
    "rand4"
   ]
  }
 }
}
PS C:\Users\Administrator\Desktop\tyu-main\gma> 

[warn] feature_select: torque/qvel missing -> energy column is identically 0. Torque is NOT recoverable by state replay; record it during collection or fall back to frame_extract.inverse_dynamics_torque().
  phase 0: recall = 0.50 (random 0.40; pool 10, 출하 2개)
  phase 1: recall = 0.50 (random 0.40; pool 10, 출하 4개)
  phase 2: recall = 0.75 (random 0.40; pool 10, 출하 4개)
[G7] PASS: mean recall 0.58 vs random 0.40
[out] ./artifacts\g7_diag.json
