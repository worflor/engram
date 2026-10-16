# Context-indexed relational evidence

`engram_relations.ContextRelations` retains real channel means and centered
cross-channel scatter for each observed context. Covariance and correlation are
derived read-only views. A zero-variance channel keeps its measured level; its
correlation row/column is zero and is not evidence of an absent relationship.

This complements oscillator signatures. It records simultaneous relationships,
not temporal order, recurrence coefficients, an action policy or semantic names.
The same context may be a well ID, a role ID, or a host-defined composite key.
The host must keep those meanings consistent. The class does not infer that two
different identifiers mean the same thing or that identical names across agents
denote compatible experience.

## Use and attach to a well

```python
from engram_relations import ContextRelations

relations = ContextRelations(('threat', 'comfort'), representation='encounter-v1')
relations.push('protective:owner', [0.2, 0.8])
relations.push('protective:owner', [0.8, 0.2])
view = relations.moments('protective:owner')
assert view.count == 2
assert view.correlation[0, 1] < 0
assert relations.moments('unfamiliar-role') is None
```

When using `WellNetwork`, a host can call `relations.push(well_id, witness)`
after routing its observation. Each call contributes one witness, irrespective
of how many blocks the codec fitted. Observe that weighting explicitly: do not
confuse witness counts with whole-trajectory well counts. The host owns the
fixed units, channel order, context definition and sampling convention.

Run `python examples/relational_wells.py` for measured conditional correlations,
JSON checkpoint/restore, unknown-context behavior and a constant-level control.
The [adversarial study](../briefs/relational-memory.md#streaming-parity-and-adversarial-binding)
uses this exact root implementation under noise, shifted units and bad bindings.

## State and algebra

For each context, the state is count `n`, mean `mu`, and centered scatter
`S = sum((x-mu)(x-mu)^T)`. A push applies a centered Welford update. Population
covariance is `S/n`, rather than the unbiased sample estimate `S/(n-1)`.
For two summaries with counts `a,b`, means `mu_a,mu_b` and `delta=mu_b-mu_a`:

```text
n  = a + b
mu = mu_a + delta * b/n
S  = S_a + S_b + delta delta^T * a*b/n
```

This is the classical
[parallel centered-moment construction](https://digital.library.unt.edu/ark:/67531/metadc837537/).
Merge preserves the sufficient statistics of the multiset union in exact
arithmetic. It is not temporal concatenation; unlike `WitnessHistory.concat`,
it needs no boundary triples because it retains no lagged observations.
Floating results depend on accumulation/grouping and are checked by tolerance.
It is not a merge of well-routing decisions, Q values or actor memories.

`merge(other)` requires identical channel names/order and representation identity
and returns an independent object. It never mutates either operand. Contexts
present on one side are copied; matching keys combine. Independent episodes
may share a moment context only when the host intends this pooled interpretation.

For C contexts and D selected real channels, persistent numeric storage is
O(C D²), and an observation update costs O(D²). D can be any positive integer;
complex pairing and the codec's even-dimensional requirement do not apply here.
There is no fixed context/sample cap or retained observation log. Counters and
numeric operations still have finite implementation resources. Full covariance
is suitable for a small selected witness schema, not automatically cheap for
thousands of raw sensor dimensions.

## API and persistence

- `ContextRelations(channels: tuple[str,...], representation=...)`: unique,
  nonempty channel names and a nonempty representation identity.
- `push(context, row)`: a nonempty string key and a finite real row of the declared
  dimension. Invalid input or finite-range update failure leaves prior state intact.
- `contexts`: sorted observed keys. Looking up a key does not create a context.
- `moments(context)`: `None` for an unknown key; otherwise an independent
  `ContextMoments` snapshot containing count, mean, scatter, covariance,
  variances and correlation. Its arrays have immutable bytes owners.
- `merge(other)`: independent schema-checked union of moment evidence.
- `to_dict()` and `from_dict(data)`: version-1, JSON-compatible sufficient state.
  The caller owns file writes; there is no implicit filesystem operation.

Version-1 state contains `channels`, `representation`, and a `contexts` map with
`count`, `mean` and `scatter`. Loading requires these statistics; missing fields
do not become fabricated zero evidence. It validates positive integer counts,
finite real dimensions, symmetry, positive semidefiniteness and centered rank at
most `count-1`, within the declared numerical checks. A one-sample centered
scatter must be exactly zero, as produced by the streaming writer. For larger
populations, symmetry, PSD and rank checks operate after division by the largest
absolute scatter entry, using tolerance 1e-10. This makes validation independent
of overall scale and avoids overflowing the eigensolver on large finite state.
The retained scatter remains in its original units. Invalid state rejects.
This is validation of supplied statistics, not proof that their reported
observations occurred or a hostile-input security audit.

This dictionary is separate from ZIP v8, V4, ENBR and the NPC checkpoint. Save it
alongside those states under the same host schema if needed. Automatic attachment,
joint transactional saving and consumer migration are not provided. A host that
changes channel units or context identity must create a new compatible memory.

Correlation removes positive affine channel units within a context, at the cost
of levels and amplitudes. Preserve the full state when those matter. Neither
correlation magnitude nor sample count supplies calibrated confidence by itself.
Wrong role bindings can reverse a well-measured relationship; no projection
recovers information the host assigned to the wrong entity.
