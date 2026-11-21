# Experiment 010: a 2D drone with analog sticks (design)

## Intent

A drone seen from the side, flown with two analog sticks like a real hobby controller, shown on
the page with the sticks moving as the network flies. First course: take off from one pillar and
land on another. The reader can watch the trained network, compare it with a hand-written
autopilot, and fly it themselves. The user also asked whether this needs SAC (it doesn't: PPO
handles continuous actions; SAC is the comparison).

## Rules (as built)

- Side view, 24 m wide, water below, two pillars: start (2 m wide, 0.5–4 m tall) and target
  (1.6–2.6 m wide, 0.5–7 m tall), 8–19.5 m apart, mirrored half the time.
- Sticks (Mode 2 reduced to 2D, "angle mode"): left stick throttle (centre = half power = a
  level drone's weight), right stick tilt (the flight controller holds up to 30 degrees).
- Physics at 60 Hz: motor lag (80 ms), the flight controller's torque limit (weak at low
  throttle), gravity, linear + quadratic drag, optional wind/gusts/payload. Sticks update at 20 Hz.
- Landing: a foot touches the target deck at under 1.5 m/s down, under 1.5 m/s sideways, under 20
  degrees of tilt, centre over the deck. Side or rotor hits, hard landings, the water, leaving the
  area: failure. 20 s limit.
- The network sees 14 numbers: offsets to the landing spot and the start deck, velocity, tilt,
  spin, motor, pad width, height, height over what's below, its last sticks. No wind, no mass.
- Reward: +1 per metre closer to the landing spot, +10 for landing, -5 for failing.

## Calls made during the build (no check-ins, per the user's standing preference)

- **PPO first, SAC as the comparison**, both with 64x64 MLPs so they ship at the same size.
- **Angle mode, not raw rotor control**: the reader can fly it too, and it's how beginners fly.
- **A grounded state**: the first contact model let tilt on the ground press a foot into the deck,
  which re-snapped the drone every physics step; with a payload it stayed glued to the pad at full
  throttle. The legs now hold it level until thrust beats weight.
- **Altitude hold for human pilots only** (JS, on by default): at centre throttle a climbing drone
  keeps climbing, which is realistic but hard with a keyboard. The networks fly the raw sticks.
- **Smoother sticks**: tried a penalty on stick changes (0.1 and 1.0) and a stick speed limit
  (0.25 and 0.1 per update); reported as findings rather than tuned until it looked good.
- **Timing fairness**: PPO seed 1 trained alone; the SAC seeds ran three at a time on separate
  cores with nothing else running.

## Experiment plan

1. PPO, calm air, 3 seeds x 5M steps.
2. SAC, calm air, 3 seeds x 1M steps (about 30x slower per step on this CPU).
3. PPO trained in random wind + payload, 3 seeds x 5M.
4. Single runs: stick-change penalty 0.1 and 1.0, stick speed limit 0.25 and 0.1.
5. 200 held-out courses per checkpoint; final pilots in calm, wind, payload, both.

## Left to the builder

Course generator ranges, colours, camera, which network ships first in the game.
