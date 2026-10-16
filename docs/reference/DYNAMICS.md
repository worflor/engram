# Dynamics and conditioning

## The recurrence as a state transition

For a single pair, set `s[n] = [z[n], z[n−1]]`. Then:

```text
s[n] = A s[n−1],       A = [[K, −G], [1, 0]]
det(λI − A) = λ² − Kλ + G
```

The minus sign on G is part of the recurrence. The product of the characteristic roots is G and their sum is K. Distinct roots give combinations of their powers; repeated roots can introduce a polynomial factor in n. Complex K/G allow richer coupled real-coordinate motion than the real examples below.

The numerical extractor scales by a representable power of two, evaluates the
larger quadratic numerator, and recovers its partner through the root product G.
This avoids losing a small root when the direct subtraction nearly cancels and
avoids forming an overflowing K squared. Scalar and vector paths share that
implementation. Nonfinite inputs and roots outside representable range reject;
near-repeated roots remain sensitive to coefficient perturbations.

For observations with an independent coordinate offset, the optional [affine model](AFFINE.md) adds `b` and keeps the same K/G characteristic roots. Its anchored regression and forecast preserve translation while retaining the origin explicitly. It is an analysis extension with separate state, not an alternate interpretation of existing packet bytes.

## Four exact examples

| K | G | Characteristic roots | Consequence in exact arithmetic |
|---|---|---|---|
| `5/4` | `3/8` | `1/2`, `3/4` | Both modes decay. |
| `2` | `1` | `1` with multiplicity 2 | Constant-velocity trajectories can grow linearly; root magnitude one does not imply bounded motion. |
| `1` | `0` | `0`, `1` | The recurrence holds the latest sample. |
| `5/2` | `1` | `1/2`, `2` | A component in the root-2 mode grows exponentially. |

These companion matrices were studied through the deployed Stithy `wolf.scryer.ask`. The returned characteristic polynomials, roots, multiplicities and diagonalisability distinguish the cases. Eight determinant/discriminant claims were separately `CONFIRMED` by `wolf.scryer.check`. The discriminants for the rational-coefficient cases were checked on the integer-scaled polynomials `8λ²−10λ+3` and `2λ²−5λ+2`.

The table's last column interprets those roots using the recurrence; it is not a returned engine finding about Engram's implementation.

Raw findings, declines, requests, invocation IDs and invocation-reported version **1.0.28** are retained in [oscillator-cases.json](../../research/exact/oscillator-cases.json). The rational-matrix studies explicitly decline integer-lattice readings such as Smith normal form; those are outside these matrices' intended role. No proof DAG was returned by these studies or replayed, and these checks are not certification of the floating-point implementation.

## Strict stable generation coordinates

For `z²−Kz+G`, set `g=G` and `kappa=(K−G*conj(K))/(1−|G|²)`. The two roots are strictly inside the unit disk exactly when `|g|<1` and `|kappa|<1`. The inverse is `K=kappa+g*conj(kappa)`, `G=g`. These are reflection/Schur coordinates. Such stability coordinates are classical; [Smith's filter treatment](https://www.dsprelated.com/freebooks/filters/Stability_Revisited.html) discusses the real-coefficient step-down test. The complex quadratic specialization used here can also be checked directly:

Let the roots be a/b, with magnitudes u/v. Then `K−G*conj(K)=a(1−v²)+b(1−u²)`. If u,v<1, triangle inequality and `(1−u)(1−v)(1−uv)>0` give its magnitude strictly below `1−u²v²`. Conversely, if uv<1 but u≥1, reverse triangle inequality gives a magnitude at least `(u+v)(1−uv)≥(1+uv)(1−uv)`, contradicting the required strict bound. Interchange a/b if needed. This proves the criterion. Substitution verifies the inverse.

The admissible coordinates form two open disks, so convex interpolation there preserves strict **frozen-oscillator** stability. `engram_stability.blend_schur` implements that explicit generation view; `WellNetwork.blend` and `WellNetwork.dream(..., blend={'well_a': 1, 'well_b': 1})` apply it to existing full-statistic wells. Counts, observed coefficient sums, routing and graph edges are untouched. Weighted residual-power RMS supplies the chosen generated-noise scale; it is not newly observed evidence or calibrated confidence.

`Dream.mixture` records the normalized source-well weights. Its itinerary is empty for a frozen mixture: no actual graph nodes were visited in a fabricated chronological order. Ordinary/graph dreams retain their existing itinerary and an empty mixture.

For a requested positive root radius ρ, use `schur_coordinates(K,G,radius=ρ)`: it applies the unit-disk construction to `k=K/ρ` and `g=G/ρ²`. The inverse returns `K=ρ(kappa+g*conj(kappa))`, `G=ρ²g`. This follows by substituting `z=ρu` in the characteristic polynomial: `p(ρu)=ρ²(u²−ku+g)`. Retain the radius with the dimensionless coordinates. The radius is part of the interpolation rule: changing it can change the generated coefficients even when the same sources remain admissible.

`network.blend(weights,radius=ρ)` and `network.dream(context,blend=weights,blend_radius=ρ)` use that admissibility contract. Every participating law must already be inside the requested disk; the operation does not force an inadmissible source into it. The root test and reflection-margin test run on rounded coefficients relative to ρ. `Dream.mixture_radius` records the requested disk; ordinary dreams leave it `None`. This is separate from the existing explicit `max_radius` projection. Radii above one permit growing modes and do not promise unit-circle stability. Root-rescaling, finite-range and float32-boundary regressions cover this scaled construction.

The default `margin=1e-8` is a reflection-disk margin, not a promised spectral-radius margin. Inputs outside it are rejected, never clipped. Static/ramp boundary modes remain usable through their existing APIs but are outside the default strict unit-disk blend. Both Schur conditions and numerical roots are checked after optional complex64 rounding. One regression starts with a strictly stable double-precision oscillator that rounds to a unit root in float32 and verifies rejection. Consolidation still merges measured statistics; it does not secretly replace them with a reflection-coordinate centroid.

The audited nearby stable pair has an ordinary coefficient mean with radius 1.005, while its reflection mean has radius about 0.99495. In a separately seeded 10,000-case campaign, 522 ordinary means were unstable; every reflection blend and its float32-rounded output passed the strict checks. This describes the declared synthetic distribution, not a failure rate for real fitted wells. [Numerical scope](../../research/history-frontiers.json).

Frozen stability does not prove bounded transients or safe switching. The dream API refuses to combine a frozen blend with graph walking. It does not install an arbitrary dwell-time restriction on unrelated learned motions.

### A specific switching certificate

For the exact homogeneous companions with `(K,G)=(±9/5,81/100)`, let r=9/10. Their repeated-root powers have the same absolute row sums. For n≥10, their infinity norm is `r^(n−1) * (19n/10−9/10)`. This decreases with n there. At n=43 it is about 0.967379, below one; at n=42 it is about 1.049590, above one.

Thus residence blocks of at least 43 updates contract in one common infinity norm for **this exact pair**. Products of permitted blocks contract at block boundaries; intermediate amplification is still possible. This does not certify arbitrary wells, affine forcing, changing coefficients, resets, noise or rounded implementations. It is a research result, not a production `dwell=43` switch for every graph.

Scryer 1.0.28 confirmed the algebraic inverse/factor identities, bound values, decreasing-ratio implication and quintic recurrence in [component receipts](../../research/exact/history-composition-cases.json). The initial linear/rational checks routed to a different engine door; the retained route-pinned retries settle them. The common-norm argument above composes those components by hand. No proof DAG for this full switching argument was replayed.

## Why reconstruction can be sensitive

V4 full packets retain fitted coefficients at float64 precision and each block's actual initial conditions. Compact coefficients use float32 and cannot reconstruct. Full decoding still uses recurrent micro prediction and quantized residuals, so growing modes, repeated roots, long horizons or poorly conditioned estimates can amplify small changes. Historical full formats also rounded coefficients and inferred later initial conditions.

The macro/micro hierarchy adds another dependency: later micro initial conditions are reconstructed from earlier sub-block output. Block/sub-block partitioning and partial tails therefore matter. A one-step fit error or residual quantization step alone is insufficient to bound the final reconstruction.

Predictive capture concerns the predictor's residual energy. It does not include all of these serial reconstruction effects. Cross-language decoder agreement checks whether the same packet is interpreted consistently; both decoders can agree on a poor reconstruction.

The measured gallery and `tools/parity.py` report source error separately. The optional `max_error` / `encode_with_tolerance` policy checks actual full-wire replay and uses exact raw float32 storage if its absolute budget is exceeded. Default prediction has no universal accuracy or stability theorem, and no compression-ratio guarantee follows from this policy. Current limits belong on [STATUS](../STATUS.md).
