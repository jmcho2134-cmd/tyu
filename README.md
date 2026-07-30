PS C:\Users\Administrator\Desktop\tyu-main\gma> python .\fcm.py screen --g7-reuse
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
[out] ./artifacts\action_sets.json  (예측 기준 사본: ./artifacts\action_sets_predicted.json)
[plot] ./artifacts\fcm_sets.png
[G7] 저장된 실측 재사용: ./artifacts\g7_diag.json
[G7] 실측 재사용 모드: 시뮬레이터를 돌리지 않고 3 phase 의 저장된 측정값을 쓴다
  phase 0: recall = 0.50 (random 0.40; pool 10, 출하 2개)
     실측 재랭킹 → ['axis[dry+]', 'axis[dz+]', 'axis[dy+]', 'rand1']
  phase 1: recall = 0.50 (random 0.40; pool 10, 출하 4개)
     [유령] 예측 강함 / 실측 0 근방 — 제외: axis[grip-](예측 -1.290 → 실측 +0.0002), rand2(예측 -0.601 → 실측 +0.0131), rand1(예측 -0.384 → 실측 +0.0270)
     실측 재랭킹 → ['rand0', 'axis[dz+]', 'steepest', 'rand5']
  phase 2: recall = 0.75 (random 0.40; pool 10, 출하 4개)
     실측 재랭킹 → ['rand1', 'rand4', 'axis[dz+]', 'axis[grip+]']
[G7] PASS: mean recall 0.58 vs random 0.40
[G7] action_sets.json 을 실측 기준으로 재작성: {'phase_0': 4, 'phase_1': 4, 'phase_2': 4}
     (예측 기준 집합은 action_sets_predicted.json 에 보존)
[out] ./artifacts\g7_diag.json
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
   "shipped_predicted": [
    "axis[dz+]",
    "rand2"
   ],
   "shipped_measured": [
    "axis[dry+]",
    "axis[dz+]",
    "axis[dy+]",
    "rand1"
   ],
   "ghosts": [],
   "filter_applied": true
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
   "shipped_predicted": [
    "axis[dz+]",
    "axis[grip-]",
    "rand2",
    "steepest"
   ],
   "shipped_measured": [
    "rand0",
    "axis[dz+]",
    "steepest",
    "rand5"
   ],
   "ghosts": [
    "axis[grip-]",
    "rand2",
    "rand1"
   ],
   "filter_applied": true
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
   "shipped_predicted": [
    "antipara",
    "axis[dz+]",
    "rand1",
    "rand4"
   ],
   "shipped_measured": [
    "rand1",
    "rand4",
    "axis[dz+]",
    "axis[grip+]"
   ],
   "ghosts": [],
   "filter_applied": true
  }
 }
}
PS C:\Users\Administrator\Desktop\tyu-main\gma> 
