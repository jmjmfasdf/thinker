# Evidence for learned adaptive planning and useful planning budgets

Checked 2026-09-25 against primary paper PDFs. The research target is a policy that learns, through environment interaction, when and how much to imagine in each situation. Fixed-budget performance curves are supporting diagnostics, not the target itself. This note also distinguishes a fixed trained agent evaluated at different budgets from agents trained with different budgets. It does not claim a comprehensive literature census.

## Closest reference: learned sequential planning control

### Learning model-based planning from scratch (Imagination-based Planner, IBP) — Pascanu, Li et al. (2017)

Source: https://arxiv.org/abs/1707.06170 ; Sections 2, 3.2–3.3 and Figure 4.

- Before each real action, a learned manager chooses whether to act or perform another imagination step, conditioned on the environment state and recurrent plan context. Planning length is an outcome of these internal choices.
- The restricted tree implementation chooses among {act, imagine from the actual root, imagine from the latest imagined state}. This closely resembles STOP/RESET/PROCEED, allowing both variable length and learned allocation between extending a path and restarting from the root.
- In the continuous experiment, a learned interaction-network world model is trained on transitions collected through real actions. The manager learns with REINFORCE and entropy regularization, receiving negative external task loss plus negative internal resource cost. The controller/memory learn the external objective through stochastic value gradients. These components train simultaneously on-policy with separate objectives/gradients.
- Figure 4b–c: higher imagination costs reduce the number of imagined steps chosen, eventually to zero, with a corresponding task-performance tradeoff. Figure 4d–e: n-step planning improves over one-step planning, and learned tree planning improves further in the tested settings. Costs increasing after execution encourage planning earlier.
- Scope: the continuous Spaceship Task 2.0 has nonlinear gravitational dynamics, noisy continuous thrust and fuel penalties, but the reported experiments allow only 1–3 real actions, each followed by an 11-step physical trajectory. This is sequential control, not evidence at long-horizon Atari scale.
- Distinguish learned architecture and reported cost/strategy adaptation from proof of calibrated per-state VOC. Cost sweeps do not by themselves establish that one frozen model adapts to previously unseen costs. Some discrete-maze experiments restrict the manager or use fixed imagination caps and should not be cited as adaptive-stopping tests.
- Among the references checked here, this is the closest architectural and learning-objective match to the user's goal. It does not validate the current Thinker VOC critic or guarantee successful transfer to Pong/Enduro.

## Direct test-time budget interventions

### Thinker: Learning to Plan and Act — Chung, Anokhin & Krueger (2023)

Source: https://arxiv.org/abs/2307.14993 ; PDF Figure 6, Section 5.1.

- A Sokoban agent trained at stage length K=20 is evaluated with different K values up to 20. The agent benefits substantially from planning and, at its original budget, exceeds the distilled base policy.
- The plotted relation is not strictly monotonic. Small K performs poorly, and changing a fixed trained agent's stage length also changes its input/time distribution. This is evidence of dependence on available planning, not proof of optimal per-state stopping.
- K includes the real action: K=1 means no imagined interaction; K=20 provides up to 19 imaginary steps. A stage step is not identical to a fresh model query, because previously expanded nodes can be traversed.
- Figure 7 is a separate experiment: agents are trained AND evaluated at K in {1,2,5,10,20}. Around K=10 is sufficient there. Do not transfer that saturation claim unqualified to the fixed-checkpoint Figure 6 curve.
- Appendix H.1/Figure 21 freezes a weak base policy and uses true dynamics, with only tree representations provided to a learned improvement operator. More K improves return, including at K=40. These are different trained operators and a different model assumption, not the same protocol as Figure 6.
- The direct budget figure is Sokoban evidence; it does not establish the effect in Pong/Enduro or in human-imitation checkpoints.

### MuZero — Schrittwieser et al. (2020)

Source: https://arxiv.org/abs/1911.08265 ; Figure 3A–B and Supplementary Figure S3.

- Fully trained networks are frozen while the MCTS simulation count is varied at evaluation.
- Atari: trained at 50 simulations per search; aggregate performance across 57 games improves up to approximately 100 simulations, then largely plateaus/slightly decreases.
- Go: trained at 800 simulations per search; the learned model supports substantially longer searches, up to two orders of magnitude more search time.
- Figure 3D is a different experiment that varies TRAINING simulations and evaluates all agents at 50. Do not mix the two claims.
- Actions are selected through the search policy, and the network learns the search-improved policy. More search is therefore connected to the action decision, although model error still limits scaling.

### TD-MPC — Hansen, Wang & Su (2022)

Source: https://arxiv.org/abs/2203.04955 ; Figure 6, Appendix C/Figure 8.

- Continuous-action control with a learned latent dynamics/reward/value model and trajectory optimization.
- Evaluate fully trained agents while changing the planning horizon or optimizer iteration count, holding the other setting fixed. Training defaults: horizon 5, six iterations. Five runs.
- Quadruped Walk: larger horizons/iteration budgets improve return at low budgets, then saturate. The paper reports that iterations can be reduced enough to save 50% planning cost without reducing performance.
- Appendix C includes Quadruped Run, Fish Swim, Reacher Hard and Cartpole Swingup Sparse. Budget sensitivity is greater for difficult/high-dimensional dynamics and smaller for simple tasks.
- This is a directly documented continuous-control example. The relevant budget ablation is in TD-MPC; do not cite TD-MPC2 merely by association as if it were the same figure.

## Adaptive stopping and cost-sensitive allocation

### Metacontrol for Adaptive Imagination-Based Optimization — Hamrick et al. (ICLR 2017)

Source: https://arxiv.org/abs/1705.02670 ; Sections 3.1–3.2, Figures 2–4 and Appendix Figure 7.

- A metacontroller chooses whether to perform another internal optimization/imagination step or execute the proposed action; it can also select among experts with different reliability and cost.
- The spaceship task has nonlinear gravitational dynamics and a continuous thrust action, but is a ONE-SHOT control decision followed by an 11-step trajectory. It is not an ongoing Atari-style control benchmark.
- Iterative planning has diminishing returns: simple scenes need about 2–3 steps, harder scenes about 5–8. The learned controller spends more computation on harder scenes and less when computation costs more.
- With the learned interaction-network expert, median total cost (task loss + computation cost) is 11% below the best corresponding fixed-iteration agent across tested cost settings. With the true simulator, the analogous figure is 15%.
- This supports adaptive computation utility, but does not independently validate a learned Q(CONTINUE)-Q(STOP) critic as in our experiment.
- The MLP expert condition performed poorly: a successful stopping mechanism requires a useful underlying optimizer/model. Appendix B.4 also discusses premature stopping during joint learning and the entropy regularization used to discourage that collapse.

## Useful comparators with different interpretations

### DRC: An Investigation of Model-free Planning — Guez et al. (2019)

Source: https://arxiv.org/abs/1901.03559 ; Section 4.3/Figure 6.

- After training, impose up to ten no-op steps at the start of Sokoban episodes to give the recurrent network additional computation on unchanged observations.
- Solving performance improves; five networks are each tested on the same 5,000 levels.
- There is no explicit learned forward world model/tree search. This is a valuable control: a compute–performance curve alone does not identify explicit model-based imagination.

### I2A: Imagination-Augmented Agents for Deep Reinforcement Learning — Weber et al. (2017)

Source: https://arxiv.org/abs/1707.06203 ; Section 4.1/Figure 4 (right).

- Sokoban training curves compare imagination depths 1,3,5,15. Longer rollouts improve learning/final solving performance, with diminishing returns; depth 15 exceeds 90% solved.
- These are agents trained with different rollout depths, not a clean frozen-checkpoint budget intervention. They support the value of imagination capacity but answer a different question from test-time scaling.

## Implications for the current VOC pilot

1. The central test is learned situation-dependent allocation: at a fixed cost/cap, does the expected computation budget depend reproducibly on the situation and on the evolving search results, and does extra computation benefit the states receiving it? Variability caused only by stochastic sampling is insufficient.
2. Test whole-policy adaptive planning against fixed/random allocation at comparable average compute, using actual environment reward and the reward–compute tradeoff. Human action likelihood is a separate behavioral-fit outcome, not the definition of useful planning.
3. Separately test resource sensitivity. A single model expected to adapt to changing available budgets/costs should receive the resource condition and train across relevant conditions. Separate models trained at different costs do not establish within-model adaptation.
4. Record total model calls, search depth, and branch/reset allocation separately. More imagination does not necessarily mean deeper planning. Use forced-budget return curves as supporting diagnostics rather than as the main scientific objective; strict monotonicity is not required.
5. Keep additional recurrent processing distinct from additional world-model information. DRC shows why greater compute alone cannot identify explicit prospective simulation.
6. The completed 100k pilot repaired teacher-to-gate transmission but did not establish useful allocation. Its paired STOP/CONTINUE tests are local estimates, not whole-policy matched-compute comparisons.
7. Prefer IBP for learned sequential imagination control, Hamrick et al. for adaptive cost-sensitive stopping and a fixed-computation comparison, original Thinker for a related architecture and fixed-budget positive control, MuZero for Atari search scaling, and TD-MPC for continuous-action planning.
