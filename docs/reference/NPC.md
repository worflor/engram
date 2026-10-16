# Experience-driven NPCs

*“a dynamic creature of learned oscillators”*

An Engram NPC turns an ordered stream of low-dimensional observations into remembered experience. It fits temporal dynamics, groups similar current situations and histories into wells, learns action consequences, and can rehearse observed consequences before responding. The host supplies the environment's meaning: witness channels, available actions and feedback.

The runtime is `engram_npc.py`. It depends on NumPy and the canonical Engram reference, with no game API, server or language model requirement. The Minecraft adapter is an optional real-game laboratory; it is not required to use the NPC in another game, simulation or software system.

## Run the example

```sh
python examples/npc_experience.py --training 120 --planning 5
```

The example is a delayed-cue navigation task. An NPC observes a red/blue cue near the start of a corridor. At the later junction, its current witness no longer contains that cue. The appropriate branch depends on the earlier observation. There is no policy field storing the cue, and the fitted features contain no raw-history mean or saved cue token. The history enters through Engram's fitted K/G/residual geometry.

The host offers physically available action IDs, assigns a small movement cost, and reports success or failure at an endpoint. It never gives the controller the correct action. The same reference controller runs in the [Minecraft lane](MINECRAFT-NPC.md), where native collision checks, movement and actual endpoint blocks provide those observations and outcomes.

## Host contract

```python
from engram_npc import EngramNPC, WitnessSchema

schema = WitnessSchema(
    ('distance', 'threat', 'health', 'light'),
    scales=(20, 1, 100, 15),
    representation='my-game/senses-v1',
)
npc = EngramNPC(schema, seed=7)

# Supply ordered observations before requesting a response. A temporal fit
# becomes available after three real observations; no missing history is invented.
for witness in initial_observations:
    npc.observe(witness)

decision = npc.decide(available_action_ids)
next_witness, feedback, terminal = environment.apply(decision.action)
npc.feedback(decision, next_witness, feedback,
             terminal=terminal, next_actions=next_available_action_ids)
npc.dream(updates=20)
npc.save('guard.engram')
```

Action IDs are arbitrary host strings. They can identify movement, interaction, dialogue intents, tool calls or parameterized behaviors owned by the host. Changing the action set is supported. The library does not contain a fixed game-action vocabulary, world grid, goal catalogue, maximum number of wells or maximum episode length.

`WitnessSchema` names channels, units/scales, offsets and a representation version. Odd-dimensional hosts receive an explicit zero companion coordinate for complex pairing. Values are normalized using that fixed schema. Sampling cadence and channel meanings must remain consistent; the library cannot infer them from unlabeled numbers. Save a new representation identity when their meaning changes.

For measured streams, `WitnessSchema.calibrate(channels, training_observations, representation=..., minimum_scales=...)` learns fixed channel means and population deviations in O(D) storage. Constant channels use one host unit unless a positive floor is supplied. Use training observations only and freeze the result before evaluation or memory formation. The [data and stress study](../briefs/npc-data-and-stress.md) tests unit conversion, real inertial recordings and the limits of this geometry. Calibrating again does not transform accumulated wells into the new coordinates.

`begin_episode()` resets the live temporal context and eligibility traces while retaining learned experience. Use it when entering an independent encounter. Continuous hosts can keep observing without episode resets and choose `history_decay<1` for exponential recency. There is no truncated observation buffer in this history. The caller controls whether old evidence retains full weight or decays.

`history_decays=(1.0,0.9)` declares several temporal scales in the native context representation. The single cumulative scale remains the default; `npc.history` is the primary scale and `npc.histories` exposes the bank. Checkpoint payload version 2 preserves all scales and still reads version-1 single-scale payloads. The [history contract](HISTORY.md) gives the exact aliasing counterexample, concatenation algebra, migration boundary and a learned recency task. Changing scales changes the geometry of experience; do not silently attach an old well centroid to new features.

## What the memory learns

The temporal fit uses constant storage proportional to witness dimension. It keeps a running mean and joint lag covariance for `(z[n], z[n−1], z[n−2])`, with stable centered updates. These sufficient statistics reproduce the reference ridge fit on the same centered data. No full trajectory needs to remain resident. Means are used to fit the recurrence around an appropriate origin; a mean history is not separately supplied to the response policy.

Each context combines the current normalized witness with K, G and residual RMS. Current levels preserve distinctions that a homogeneous dynamics signature alone would erase. Temporal coefficients preserve distinctions that identical current witnesses would erase. `novelty_radius` controls online well growth in this joint feature geometry. It is an explicit modeling parameter, not a maximum well count.

Wells contain actual Engram sufficient statistics, plus learned action values, visits, feedback totals and action-conditioned transition outcomes. The existing `EngramFile` observation graph is retained alongside this conditional model. Its older bounded recent-dream buffers are separate from the NPC's cumulative streaming fit and response model; they do not limit how many observations, states or actions the NPC can learn.

Responses use temporal-difference learning with replacing eligibility traces, plus explicit exploration of untried or underexplored actions. `learning_rate`, `discount`, `trace_decay`, `exploration` and `epsilon` configure that policy. The structure can collect experience with zero feedback too; useful task-directed behavior requires the host to define what outcomes matter. A numeric witness alone cannot specify a goal.

`responses()` exposes action values, visits and mean observed feedback for the current well. Values are learned estimates under the supplied task, not calibrated probabilities or language understanding.

`Decision.exploratory` marks random/untried selection. The separate underexplored-action bonus can also affect a decision; the flag is not an exhaustive classification of all exploration or a confidence estimate.

## Dreaming as rehearsal

`dream(updates=N)` performs Dyna-style planning over the memory's **observed** state/action outcome model. It updates response values without incrementing observed visits, feedback counts or fitted history. N is caller-selected and has no library maximum. This gives experience a reusable planning role without accepting generated positions as real experience.

This differs from `Engram.dream` parameter-history synthesis and `WellNetwork.dream` oscillator/graph continuation, which remain available. The NPC policy does not depend on unrestricted coefficient averaging or switched oscillator trajectories being physically stable. Hosts can still use generated curves as candidate animation or motion, evaluated under their own constraints.

## Blank, tailored and general seeds

**Blank:** construct `EngramNPC(schema)`. Observation wells and action consequences grow from the environment. There are no preloaded action preferences.

**Environment-tailored:** load a saved NPC with `EngramNPC.load('guard.engram', schema=schema)`. This restores actual learned response values, action-conditioned transitions, fitting moments, RNG state and live eligibility history. The same witness schema is required. A translated game arena can retain its representation when the host explicitly uses relative coordinates; the lane tests that transfer.

**General dynamics prior, including Alexandria:** use `DynamicsPrior.load(path, representation=...)`. Feed it vectors in its actual original embedding/preprocessing coordinates, then use its geometric affinities as explicitly named host observations if useful. It does not invent action values for a new environment.

```python
from engram_npc import DynamicsPrior

prior = DynamicsPrior.load('alexandria.engram',
                          representation='my-matching-glove300-pipeline')
for vector in matching_embedding_observations:
    profile = prior.observe(vector,
                            representation='my-matching-glove300-pipeline')
```

Alexandria has 300 dimensions, a particular pairing and a historical vocabulary pipeline. Eight physical sensor values do not automatically inhabit that coordinate system. Its v4 per-well data supplies K; absent G statistics are excluded and reported as such. The representation name is a caller declaration, not a proof that an embedding matches. The host must supply the actual vocabulary/adapter and consistent smoothing.

`examples/npc_seeded.py --vocabulary <matching-glove300.bin>` runs the included Alexandria artifact against caller-supplied GLV1 vectors without downloading or copying the vocabulary. That example checks finite dynamics affinities; it does not claim Alexandria knows Minecraft actions. Generic affinity priors and task-specific skill memories carry different information.

## Checkpoints and evaluation

NPC state is stored in `npc.json`, an extension section inside the reference ZIP v8 `.engram` memory. Existing reference import/export preserves that section. Native Rust ENBR has no equivalent NPC policy payload. Loading a plain dynamics artifact through `EngramNPC.load` is rejected rather than fabricating learned responses.

Checkpoints atomically replace a file. Save after an action's feedback has been acknowledged; saving during a pending decision is refused. Decision tokens belong to one agent and can be consumed once. Independent NPC instances own independent contexts and RNGs. A saved agent is not a saved game world; the host owns correspondence between agent state, world state and outstanding external actions.

Evaluate with `decide(..., explore=False)`, `observe(..., learn=False)` and `feedback(..., learn=False)`. The live observation context still advances; learned well statistics and response values remain fixed. The runnable study tests learned behavior, an empty-memory baseline, disabled temporal features, save/load transfer and reversal of the environment's outcome mapping.

The system deliberately has no fixed limits on wells, actions or episode length. Memory still grows with learned contexts and transitions; numeric ranges, the existing bounded snapshot format and available compute/storage remain finite. Planning work, exploration/recency choices and experimental deadlines are caller-owned. No data is silently pruned to satisfy a hidden NPC state cap.

Loading also reconciles the NPC/dynamics well inventories, action visits against transition counts, and total real updates against recorded visits. These checks reject internally inconsistent snapshots. They do not authenticate external files or equate floating feedback totals across different summation orders.

Determinism means the same ordered observations, action IDs, feedback, configuration and seed under the same numerical runtime. The [stress campaign](../../research/npc/stress.json) checks full continuation across a checkpoint and independent Python hash seeds. It does not imply invariance to changing the order of experience: an online learner's changing centers, values and exploration decisions depend on that order. Archive bytes include incidental file metadata; the campaign compares canonical experience state and decision sequences.

Routing currently examines every learned well, so its work grows with the number of contexts. Streaming temporal fitting uses O(D) storage; that does not make the growing response/transition model constant-sized or promise constant-time lookup.

The controlled task establishes temporal experience use and adaptation on that task. It does not establish optimal navigation, general intelligence, arbitrary transfer or a stable autonomous control policy for every environment. Run your host's task outcomes and held-out cases; inspect both successes and path costs.
