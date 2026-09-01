# Enduro VoC-v7 200k preregistered acceptance

This specification was frozen after the independent fresh-100k mechanism pass
and before any fresh-200k run using the added transition telemetry.  It must not
be weakened or redefined after inspecting the 200k data.

## Population and aggregation

- Full continuation window: `100000 < real_step <= 200000`.
- Late window: `175000 < real_step <= 200000`.
- Stability windows: `(170000, 180000]`, `(180000, 190000]`, and
  `(190000, 200000]`.
- Exclude overshoot rows above 200000.  Do not interpolate or substitute a
  nearest row.  Use only complete CSV rows whose required fields are finite.
- Pool counts and sufficient statistics over events.  Do not average row-level
  rates without their corresponding support.
- Define `delta_q = Q_continue_ema - Q_stop_ema`, with tie tolerance `1e-6`.
  Positive and negative mean strictly above and below that tolerance.
- Deep means pre-decision depth at least 8.  Exclude WAIT, forced-only, and
  otherwise invalid control rows unless a criterion explicitly audits them.

## Mechanism criteria

The full and late windows must both satisfy:

- teacher conditional gap at least `0.075`;
- student conditional gap `E[p_continue | delta_q > 0] -
  E[p_continue | delta_q < 0]` at least `0.05`;
- student-gap / teacher-gap retention at least `0.50`;
- pooled signed margin greater than zero;
- online-vs-EMA non-tie sign agreement at least `0.60`;
- train and holdout CONTINUE and STOP support each greater than `5%`;
- each wrong-side saturation rate below `1%`;
- held-out EMA selected-action TD RMSE at most `0.5`;
- zero actor, online-Q, and gate AMP skips and zero non-finite events.

At least two of the three stability windows must have both a positive student
gap and positive signed margin.  Across the ordered full-window learner rows,
the maximum consecutive run of negative trailing-five-row pooled student gaps
must be at most three.

## Four required learned behaviours

Each behaviour must pass in both the full and late windows.

### 1. Easy negative-Q states stop

- Each Q sign has at least `5%` of non-tie support.
- `E[p_continue | delta_q < 0] <= 0.475`.
- Sampled `STOP | delta_q < 0 >= 0.525`.
- Conditional argmax STOP accuracy is at least `0.60`.

### 2. Hard positive-Q states continue searching

- `E[p_continue | delta_q > 0] >= 0.525`.
- Sampled `CONTINUE | delta_q > 0 >= 0.525`.
- Conditional argmax CONTINUE accuracy is at least `0.60`.

### 3. Deep negative-Q states stop

For `depth >= 8 and delta_q < -1e-6`:

- support is at least 256 in the full window and 64 in the late window;
- mean continue probability is at most `0.475`;
- sampled STOP rate is at least `0.525`;
- argmax STOP accuracy is at least `0.60`.

Forced stops do not count as successes, and the overall forced-stop rate must
remain below `1%`.

### 4. Useful computation is accepted and then re-evaluated

An eligible transition is an immediate adjacent pair in the same environment
stream for which both decisions are valid, the prior EMA `delta_q > 1e-6`, the
prior sampled control is PROCEED or RESET, the current
`predecision_last_control` equals that prior control, and current pre-decision
depth equals prior depth plus one.

- Observable eligible-pair coverage is at least `95%`.
- PROCEED and RESET each have support at least 256 in the full window and 64 in
  the late window.
- At the next decision, each positive and negative Q branch has support at
  least 128 in the full window and 32 in the late window.
- At the next decision, `p_continue | positive >= 0.525` and
  `p_continue | negative <= 0.475`.
- The next sampled action is sign-correct at least `0.525` for each sign.
- The next conditional argmax action is sign-correct at least `0.60` for each
  sign.
- The PROCEED and RESET transition slices each have positive signed margin.

In each of the three stability windows all four behaviours must retain the
correct direction, and at least two windows must meet their `0.525` strength
conditions.

## Fixed-checkpoint confirmation

Training telemetry establishes control relative to the learned EMA Q.  Before
claiming final Enduro success, evaluate the fixed 200k checkpoint with training
disabled, epsilon zero, stochastic gate sampling, and fixed seeds.  Apply the
same four behavioural definitions and report selected-action heldout
calibration.  Pong and Space Invaders remain out of scope until this Enduro
confirmation passes.
