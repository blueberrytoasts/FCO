# Densitometer Stability Investigation — Next Steps

**Date:** 2026-02-22

## Background

Observed a consistent upward OD drift (~0.0012 OD, ~0.5%) over 3 hours when repeatedly measuring an **unirradiated (0 cGy)** film with the Dektronics UV/VIS densitometer (transmitted visible mode, Seoul Semiconductor SunLike 3000K white LED source). This was observed across multiple time interval experiments (10 sec, 3 min, 5 min, 10 min, hourly).

Since the film is unirradiated and chemically inert, the drift is **entirely instrument-sourced** — most likely slow thermal stabilization of the detector/electronics chain. The built-in temperature correction is insufficient to fully account for it.

Key observations:
- Drift is not linear — has perturbations when AC kicks on (external temp drops)
- OD continues rising even when temperature drops (correction partially overcorrects or undercorrects)
- Drift appears to slow after ~60–90 min but does not fully plateau within 3 hours
- LED warm-up is not the primary cause (LEDs stabilize in minutes); electronics chain drift is more likely

## Planned Experiments

### Experiment 1 — Repeatability Check (0 cGy, cold start)
**Purpose:** Confirm the drift is reproducible and not a one-night fluke.

- Same protocol as original: 0 cGy film, cold start, measure every 3 min for 3 hours
- 10 readings per interval, record temp each interval
- Expected outcome: similar upward drift (~0.0012 OD over 3 hrs)
- If result is wildly different, something else was going on in the original run

### Experiment 2 — Warm-Up Period Test (0 cGy, 1-hour warm-up)
**Purpose:** Test whether a mandatory warm-up period eliminates or significantly reduces drift.

- Turn device on, let it run for 1 hour before taking any measurements
- Then measure same 0 cGy film every 3 min for 60–90 min
- Does NOT need to be 3 hours — 90 min post-warmup is sufficient to assess residual drift
- Expected outcome: flatter OD curve if warm-up is the fix

### Experiment 3 — Irradiated Film (500 cGy + 0 cGy in parallel)
**Purpose:** Characterize drift in the clinically relevant scenario and separate instrument drift from post-irradiation darkening (PID).

- Measure **both** a 500 cGy film AND a 0 cGy film, **alternating at each time point**
- Same 3-min interval, 10 readings each film per interval
- Net OD = OD_500cGy(t) − OD_0cGy(t) at each time point
- This isolates true film development from instrument drift
- Without the parallel 0 cGy measurement, you cannot separate PID from drift

**Why this matters:** An irradiated Gafchromic film undergoes post-irradiation darkening — OD increases after exposure even without instrument drift. Measuring both films simultaneously is the only way to distinguish the two effects.

## Key Questions These Experiments Will Answer

1. Is the drift reproducible? (Exp 1)
2. Does 1-hour warm-up solve it? (Exp 2)
3. Does the drift matter for actual dose measurements, or does Net OD cancel it out? (Exp 3)
4. How much of the OD increase on an irradiated film is real PID vs instrument? (Exp 3)

## Relevance to Protocol

In the IR-FCO protocol, the 0 cGy film is already used as a reference (Net OD = OD_irradiated − OD_unirradiated). This cancels instrument drift **only if** the reference and dose films are measured at the same time or back-to-back. If they are measured at different points in a session, residual drift introduces error.

A warm-up period + alternating reference measurements is likely the practical fix for clinical use.
