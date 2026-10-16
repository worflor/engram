# Ordered witness histories and temporal scales

`engram_history.WitnessHistory` owns the streaming fit summary. `engram_npc.WitnessHistory` re-exports that same class for existing callers. Affine history snapshots use this root implementation too. There is no second fitting engine behind the NPC.

## A fit summary is not a complete decision state

The cumulative histories `000100` and `001000`, embedded in one complex pair with a zero companion, have the same length, current sample, first/last two samples and multiset of consecutive triples: `000`, `100`, `010`, `001`. Their exact cumulative moments and K/G/residual fit therefore agree. The goal “was there a pulse among the last three observations?” differs.

The fit is doing its job correctly. That goal needs a temporal distinction the cumulative statistics discard. More training of the same aliased context cannot reconstruct the missing distinction. Exponential recency separates this pair, and the NPC supports several explicitly chosen temporal scales in its native context representation.

```python
from engram_npc import EngramNPC, WitnessSchema

npc = EngramNPC(
    WitnessSchema(('signal', 'companion'), representation='senses-v2'),
    history_decays=(1.0, 0.9), seed=7,
)
```

Every scale fits the same observations with a different explicit decay. Context geometry concatenates each scale's K/G/log-residual block with the current witness. The primary scale remains `npc.history`; `npc.histories` exposes the whole bank. Each context's ordinary Engram well statistics retain that primary scale's measured dynamics; its experience centroid uses all configured scales. Neither raw pulse age nor a task-specific cue field is added.

The default is one cumulative scale. The `history_decay=` single-scale argument is supported; a nondefault scalar and an explicit bank must agree on the primary scale. Changing the bank changes feature geometry and requires fresh compatible experience or an explicit migration study. Do not attach old centroids to new feature coordinates. Novelty radius must be chosen for the selected representation; extra scales are not automatically useful for every task.

`begin_episode()` resets every scale together. Checkpoints write NPC payload version 2 inside the unchanged ZIP v8 container, preserving the bank and config. Version-1 single-scale NPC payloads remain readable. Loading checks the number of scales, matching sample counts and common final samples. Version-1 NPC readers do not support version-2 payloads. Existing packets and the Rust memory profile have no temporal-bank payload.

Storage/fit work is O(SD) for S declared scales and D witness channels. There is no fixed bank-size cap, but choosing more scales increases actual cost. Well routing/storage still grow with experience. A finite bank is not a universally sufficient or lossless observer.

Run `python examples/temporal_witness.py`. In the controlled two-history task, 240 training encounters followed by 100 frozen evaluations give 50% for `(1.0,)` and 100% for `(1.0,0.9)` at seeds 7, 19 and 81. The radius is explicitly 0.01. This establishes learned use of the recency distinction in that task, not general intelligence or a universal choice of decays.

## Compose ordered summaries

```python
from engram_history import WitnessHistory

left, right = WitnessHistory(4, decay=0.9), WitnessHistory(4, decay=0.9)
for row in first_clip:
    left.push(row)
for row in following_clip:
    right.push(row)
whole = left.concat(right)
fitted = whole.signature()
```

`concat` summarizes the **ordered concatenation** without replaying either interior and without mutating its operands. Histories must share dimension, decay and the host's actual channel/unit meanings. Dimension/decay are checked; semantic coordinate compatibility remains the caller's responsibility. This is only appropriate when the clips really are consecutive. Separate encounters should keep their declared boundaries.

The summary retains:

- Total samples, recency-weighted sample mean and weight.
- Weighted mean/scatter of `(target, previous, older)` triples.
- The first two and last two complex samples.

When A is followed by a B of length m, age A's weights/scatter by `decay**m`, combine the internal moments, and insert the missing triples `(b0, a_last, a_previous)` and `(b1, b0, a_last)` when those samples exist. Their weights are respectively `decay**(m−1)` and `decay**(m−2)`. Empty and short clips need fewer triples.

For two weighted joint populations with weights a/b, means μA/μB and centered scatters CA/CB, write δ=μB−μA. Their merged scatter is:

```text
C = CA + CB + (a*b/(a+b)) δ δ*
μ = μA + (b/(a+b)) δ
```

The implementation evaluates the weight product without a needless overflowing intermediate. The centered merge is established parallel-statistics mathematics; [Pébay and colleagues' weighted-moment report](https://www.sandia.gov/research/publications/details/formulas-for-robust-parallel-computation-of-arbitrary-order-arbitrary-varia-2015-02-01/) provides the general background. The chronological seam triples are essential to this Engram application.

The result equals the direct summary of `A || B`. Concatenation is associative in exact arithmetic and generally noncommutative. Floating association is tolerance-based, not bitwise identical. The independent campaign checked 2,815 binary-history partitions at each of three exact decays, plus 96 floating associations up to 1,000 samples; maximum measured signature difference was `3.41e-12`. Tests also include large coordinate offsets and decay underflow.

This algebra composes **fit evidence**, not arbitrary brains, well assignments, Q tables, eligibility traces or every intermediate learning decision. Splitting online NPC learning into workers and concatenating their final fit summaries does not reproduce a sequentially trained NPC.

## Older saved histories

History snapshots write version 2 with explicit prefix endpoints. Legacy snapshots did not retain those endpoints. For zero/one/two samples they can be recovered from the complete saved suffix; for longer histories they remain explicitly unknown. Such a history can continue fitting and can precede a new clip, but cannot be appended after a nonempty preceding clip when its missing prefix would be needed. `concat` raises rather than inventing seam samples. Unknown prefixes survive a subsequent save/load.

## A bounded observer check

The [finite observer requests](../../research/exact/witness-observer-requests.json) describe all 64 six-bit sliding windows. Allowed actions append 0 or 1 and discard the oldest bit. The goal is recent-pulse presence. Exact observations use current value plus rational K/G/mean-squared residual, with the same ridge and degeneracy rule; equality of squared residual is equivalent to equality of nonnegative RMS here.

Deployed Scryer **1.0.28** found the cumulative candidate insufficient and returned the pair `000100`/`001000` with an empty distinguishing action word: their goals already differ. The two-scale candidate preserves this declared goal under all finite action words in this finite transition system. Both returned proof DAGs were separately submitted to deployed `wolf.scryer.verify`, which returned `kind: valid`. [Studies and replay receipts](../../research/exact/witness-observer.json).

This certifies the supplied finite table, including its candidate-equivalence labels. The Python generator's binding from rational fits to those labels is independently checked in the repository; it is not itself certified by that DAG. The two-scale table distinguishes all 64 windows, while the goal/action quotient has only four classes. It is therefore sufficient here, not minimal. The source uses a six-sample sliding window, whereas the production NPC's temporal fits are cumulative/decayed; this is not a proof about its unbounded stream, floating distances or learned policy.

Representation evaluation should declare a task-relevant distinction, demonstrate its loss, add a representation that retains it, then test actual decisions. The [integration study](../briefs/history-and-composition.md) records the other accepted and deferred research directions.
