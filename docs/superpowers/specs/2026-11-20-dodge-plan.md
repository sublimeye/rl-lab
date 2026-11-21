# Experiments 012 and 013: dodging, then watching

Agreed with the user on 2026-11-20 ("two experiments ... keep it autonomous end to end").
Both build on post 11's vectorised fleet (experiments/011-drone2/fleet_env.py).

## 012 "dodge school": threats you react to

Threats, each a crash on contact:
- **Birds**: as post 11 (slow 1-3 m/s, fast 4-7 m/s, bobbing), up to 4.
- **Falling rocks**: dropped from above the screen at a steady beat, aimed near the drone with a
  deterministic spread; fall under gravity with a little drift. The camera can't see straight up,
  so the camera pilot gets each rock's **shadow** on the ground (sun overhead).
- **A missile**: launched from a side, constant speed faster than the drone, steers by
  proportional navigation (turn rate N x line-of-sight rate) with a turn-rate limit and finite
  fuel. A "missile launched" warning bit (like a real missile-warning sensor).

Plus a **dodge-school arena**: no landing, hold a point in the air for 12 s under dense threats.
Training mix "school" = dense threats on normal courses + 50% arenas, mixed (not sequential, to
avoid forgetting). Baseline = trained on the plain test mix (post 11's way).

Pilots: exact positions (4 most urgent threats: position, velocity, kind; urgency = distance /
closing speed) and the camera (rays with a threat channel and a shadow channel).

Measurements: landed of 200 test courses; arena survival per threat; blindfold per threat kind;
an **escape test**: for each hit, rewind to L seconds before it and try 9 full-stick escapes; if
any survives, the hit was avoidable with L seconds' notice.

## 013 "watch, then go": threats you predict

**Air-frogs**: drift, then jump on a rhythm (period 2-3 s per frog, small jitter), guarding the
gap the drone must cross. Two kinds: with a visible crouch before the jump, and without (rhythm
only). The pilot sees positions and velocities, never the frog's timer.

Memory: a 3.2 s history of the frog slots (16 snapshots at 0.2 s) for the MLP, compared with an
LSTM (sb3-contrib RecurrentPPO) and with no memory. gamma 0.995 so waiting a few seconds isn't
discounted away; hovering drains the battery, so waiting has a real cost.

Proof it predicts (not just waits): when it goes vs the frog's phase; a rhythm switch mid-flight
(does it get fooled?); and an auxiliary head trained to predict each frog's position 0.5 s ahead,
drawn on the page as a ghost.

## Known flaws checked before starting

- Post 11's birds re-entered 2 m beyond the screen edge (wrap-around): near an edge a bird
  appeared out of nowhere. 012 birds re-enter 6 m out.
- Rare threats give PPO little signal (post 11: most courses had no bird in the way): dense
  school courses; arenas mixed in rather than run first (fine-tuning afterwards forgets).
- Unwinnable threats teach nothing: the escape test checks every threat type is avoidable with
  reasonable notice; a missile with endless fuel would always win, so it has 4 s of fuel.
- Slot order: threats sorted by urgency, not distance, so a fast missile isn't hidden behind a
  slow bird that happens to be nearer.
- Memory must be needed for real: the frog's timer is never in the input (or the MLP needs none).
- Waiting must be worth it: gamma 0.99 at 20 updates/s halves a reward 3.5 s away; 013 uses 0.995.

## Making learning faster (measured first)

Profile (M1 Pro, 4 threads, post 11's state task): PPO's update takes ~70% of wall-clock, the
simulator ~30% (rollouts alone run at 126k steps/s; with 10 epochs total is 38k steps/s; 4
epochs: 68k; 1024 drones and 4 epochs: 91k). Candidates, judged by wall-clock to a good pilot
(not steps/s, which misled in post 11):
1. fewer PPO epochs / more drones per update;
2. warm start from post 11's pilot (input surgery: old inputs keep their weights, new inputs start
   at zero, so the starting pilot flies exactly like post 11's);
3. dense threats (more lessons per minute).
No JAX (needs the user's approval); 5-hour cap per run.

## Outcome, 012 (2026-11-21)

- Faster: 512 drones and 4 PPO epochs reach a better pilot ~8x sooner than post 11's settings;
  warm start from post 11 decays; gSDE no help.
- Pilots trained on the test mix don't dodge (blindfold); the escape test says nearly every hit
  was avoidable. A missile-speed ramp (6 -> 11 m/s) teaches missile dodging (167/200 arenas, 1
  blindfolded). Birds only teach dodging when aimed (180/200, 3 blindfolded); that pilot then
  over-reacts to random birds. School 2 (courses + aimed birds + missile arenas) dodges aimed
  birds without over-reacting, half-learns the missile in one seed of two, lands less.
- Post: site/src/content/posts/012-dodge.mdx; numbers from experiments/012-dodge/summarize.py.
