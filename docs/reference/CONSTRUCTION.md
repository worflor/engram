# Mathematical construction

Engram represents motion by fitted recurrences and experience by retained statistics of those recurrences and their observations. Packets, histories, wells and relational summaries retain different information. They share a coordinate contract and moment-based construction; they are not interchangeable encodings of a complete observation stream.

## Coordinates and observations

A trajectory is a finite real matrix `w[T,D]`. The codec requires positive even D. A permutation selects coordinate pairs, and pair j becomes `z_j = w[pairing[2j]] + i*w[pairing[2j+1]]`. Each pair is one real plane on which a complex recurrence is fitted. An odd-dimensional NPC witness receives an explicit zero companion from its schema. The real relational summary does not require complex pairing and accepts any positive number of channels.

Channel meanings, units, preprocessing, pairing and sampling convention define the representation. Parameter coordinates are comparable only within that representation. Adaptive per-trajectory pairing is useful for encoding an individual signal; a memory must retain a common pairing across observations. Labels identify host-defined categories and do not change the coordinate algebra.

Public real-sample boundaries reject complex arrays. A complex sample must be represented explicitly by its real and imaginary coordinates. Codec samples use float32; fitting accumulates float64/complex128. Affine analysis and relational statistics retain float64 independently of packet precision.

## The retained lag moments

For one complex pair, form the consecutive triples `q_t = (y_t,u_t,v_t) = (z_t,z_(t-1),z_(t-2))`. With nonnegative weights a_t, the centered moment state is

```text
W  = sum a_t
mu = sum a_t q_t / W
C  = sum a_t (q_t-mu)(q_t-mu)*
```

Here `*` denotes conjugate transpose. The raw Hermitian second moment is `H = C + W mu mu*`. If every lag column is translated by a common origin c, the corresponding moment is `H_c = C + W (mu-c*1)(mu-c*1)*`.

`WitnessHistory` retains this state independently for each complex pair, together with sample-level moments and boundary samples. Cumulative weights summarize the episode; exponential weights declare a recency policy. Neither retains every chronological distinction. [History](HISTORY.md) defines the ordered-state and compatibility contracts.

## Extracting a recurrence

The homogeneous predictor is

```text
y_t = K u_t - G v_t + error_t
```

For each pair it minimizes `sum a_t |y_t-K*u_t+G*v_t|^2 + eta*(|K|^2+|G|^2)`. Write

```text
A = sum a_t |u_t|^2        B = sum a_t |v_t|^2
Q = sum a_t u_t conj(v_t)
U = sum a_t y_t conj(u_t)  V = sum a_t y_t conj(v_t)
```

The complex normal equations and their solution are

```text
[[A+eta, -conj(Q)], [ -Q, B+eta ]] [K,G]^T = [U,-V]^T

Delta = (A+eta)*(B+eta) - |Q|^2
K = (U*(B+eta) - conj(Q)*V) / Delta
G = (Q*U - (A+eta)*V) / Delta
```

The reference coefficient penalty is `eta = max(float32_epsilon, 1e-4*(A+B)/2)`. It is a declared regularization policy, not a stability constraint. Its absolute floor also limits exact scale invariance near numerical silence. Short or numerically degenerate inputs use the entry point's documented fallback. Residual RMS measures fit error in the same coordinates; it is not a calibrated likelihood or a future-error bound.

The batch signature fits the supplied coordinates. `WitnessHistory.signature` uses the episode's weighted sample mean as a common origin by default; its `center=False` view uses the original coordinates. These choices answer different regression questions.

## An affine view of the same lag evidence

An explicit intercept gives `y_t = K*u_t-G*v_t+b+error_t`. Minimizing over b first yields `b = mu_y-K*mu_u+G*mu_v`. Fitting K/G therefore uses **each lag column's own mean**, through C, rather than subtracting one common origin. The intercept is unpenalized.

`AffineOscillator.from_history` reads the retained lag evidence and endpoints; it does not need a second raw history. Its anchored forecast is `mu_y + K*(previous-mu_u) - G*(older-mu_v)`. Translating coordinates by c preserves K/G and changes b to `b+c*(1-K+G)`. A bare homogeneous signature omits that origin state and cannot stand in for the complete affine model. [Affine dynamics](AFFINE.md) defines the fit and forecast interface.

## From signatures to wells

A signature retains K/G, residual RMS, order and pairing. A well aggregates fitted signature statistics under a declared routing and weighting rule. Packet absorption weights predictive blocks; `WellNetwork` counts one whole-trajectory signature per learned observation. These are distinct measures.

A mean of fitted coefficients is generally **not** the least-squares fit of the pooled original samples: solving a regression and averaging its parameters do not commute. Well statistics describe remembered parameter geometry; `WitnessHistory` describes lag evidence for a fit. Neither operation silently replaces the other. [Memory](MEMORY.md) and [learning](LEARNING.md) define their respective update policies.

`ContextRelations` retains the real multivariate counterpart `(count,mean,C)` for a host-defined context. Its full instantaneous cross-channel covariance preserves coupling omitted by independent pair fits. Correlation is a normalized view, not a replacement for measured levels. Conditioning and temporal fitting select which relationships survive; neither supplies a universally sufficient decision state. [Relational moments](RELATIONS.md) defines those choices.

## Composition has a type

For moment populations with weights a,b and mean difference delta, unordered union uses `C = C_A+C_B+(a*b/(a+b))*delta*delta*` and `mu = mu_A+(b/(a+b))*delta`. It is associative and commutative in exact arithmetic. Relational summaries use the real transpose version.

Ordered history concatenation also ages earlier evidence when decay is active and inserts up to two lag triples across the seam. It is associative in exact arithmetic and generally noncommutative. Independent episodes must not acquire seam triples unless the host intends continuous time.

These operations combine sufficient statistics. They do not reproduce intermediate well assignments, action values, eligibility traces or arbitrary learned policies. Floating-point grouping is tolerance-based throughout.

## Generation and replay

The companion state `[z_t,z_(t-1)]` advances under `[[K,-G],[1,0]]`; its roots solve `lambda^2-K*lambda+G=0`. Magnitude and argument describe characteristic growth, decay and rotation. Initial conditions remain essential. Stable roots alone do not bound transients, arbitrary switching or output range.

Statistical coefficient averages need not remain stable. Schur/reflection coordinates provide an explicit admissible space for frozen generation blends without changing learned coefficient sums. They do not redefine a learned centroid. [Dynamics](DYNAMICS.md) gives the criterion and numerical boundaries.

Packets retain block-local initial conditions and a residual hierarchy. Macro motion is predicted first, micro predictors model its residual, and quantized values retain the remaining error. Linear mode uses K=2/G=1; static mode uses K=1/G=0; raw mode stores samples directly. V4 predictive residuals are signed int8 values with pairwise scales; full coefficients and causal block state are explicit. [Formats](FORMATS.md) owns the byte layout and compatibility rules.

Predictive capture is `100*(1-residual_energy/signal_energy)`, clamped below at zero with a low-energy convention. It excludes header cost and subsequent reconstruction effects. Quantized micro recurrence can amplify perturbations; the optional reconstruction budget checks actual full-wire replay and selects raw storage when required. Decoded error, byte size, fit error and host task performance remain separate measurements.
