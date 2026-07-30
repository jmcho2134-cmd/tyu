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
