# Formats and compatibility

Extensions alone do not identify a format. Inspect the magic and version before selecting a reader.

| Data | Identifier | Content | Reader |
|---|---|---|---|
| Reference trajectory/statistics | `EN`, byte version 4, flags byte | Explicit block state for full packets; basis and statistics retained in compact packets | Rust full/compact readers, Python `from_wire` |
| Historical full / compact | `EN`, byte versions 3 / 2 | Earlier inferred block state / basis-less statistics | Same readers, with compatibility limits below |
| Rust memory | `ENBR`, byte version 1 | Dimension, pairing, K sums/counts and dream history | Rust `brain_io::load` |
| Reference memory | ZIP archive, manifest `format: engram`, version 8 | Complete reference state with full-precision histories | Python `EngramFile.load` |
| Historical memory | ZIP v4–v7 or Manifold ENDB v2 | Producer-dependent state, with missing fields identified on import | Python `EngramFile.load` |

## Reference packets v4

All multi-byte values are little-endian. Header: `EN` (2 bytes), version u8=4, flags u8 (0=full, 1=compact), dim u16, sample length u32, alpha f32, macro block u32, micro block u32, block count u32. Both forms then carry dim u16 pairing indices, a validated permutation.

Each block starts with mode u8, start u32, length u32 and micro span u32, then four f64 measurements: signal energy, residual energy, macro capture and micro capture. Modes are cascaded=0, linear=1, static=2, raw=3. Macro K and G arrays each contain dim/2 complex numbers: interleaved f64 real/imag for full packets, f32 for compact packets. A u8 RMS flag (0/1) is followed by dim/2 f32 RMS values when present.

Compact blocks end there. Full blocks additionally carry two dim-length f32 seed arrays, micro-predictor count u32, K/G complex-f64 arrays for each micro predictor, dim/2 f32 maximum residual magnitudes, residual byte count u32 and residual bytes. Raw blocks store exact row-major float32 samples; static blocks have no residual bytes; predictive blocks store row-major signed int8 residuals in [-127,127]. Reconstruction multiplies by scale/127. Rust converts its internal quantization steps at this boundary.

Block coverage begins at zero, or at two when initial conditions are stored separately, and must end at the declared sample length without gaps. Current encoders use the two-sample prefix except for tiny/raw fallback packets. Explicit per-block seeds and micro spans avoid accumulated inter-block seed drift and ambiguous partial tails. Compact packets are flagged non-reconstructable in memory and decoders reject them.

The format retains enough state for the current mathematical model; it is not an entropy compressor. Header/parameter cost can exceed raw storage. The optional error-budget encoder may intentionally choose a larger raw representation.

## Historical full v3

The historical V3 header uses u16 block sizes/count and carries one global seed pair. Blocks infer starts and later seeds; macro/micro coefficients use float32 and residuals use zigzag-varints. Readers retain that layout's inference rules. Saving with the current writer produces V4, not a byte-for-byte rewrite of V3. Frozen fixtures retain actual earlier packets for compatibility checks.

## Historical full v1

The inherited Python and Rust implementations used the same version byte with different scale conventions, block positioning and seed restoration behavior. Rust rejects V1. Python requires `from_wire(data, legacy_v1='python')` to read the historical Python layout, rather than guessing the producer. Historical Rust source is retained under `legacy/rust-portfolio`.

An old Rust v1 file is not safely converted by changing its version byte. A converter must know its producer, restore block positions/seeds, normalize scales, and check decoded output. No universal converter ships here.

## Compact v2

V2 compact packets omit pairing, initial conditions, micro predictors and residuals. Cascaded blocks retain K/G and f16 pair RMS; linear blocks retain f16 pair RMS with implied K=2/G=1; static/raw blocks have no additional predictor payload. The parsed packet has `basis_known=false`; identity is only a placeholder. Attach the host's original basis with `with_pairing` before serialization or absorption. V4 compact packets carry their basis directly.

Older Python compact writers omitted the linear RMS payload. Their malformed linear streams are not promised compatible with the current layout. Keep the original producer revision for such files.

## Brain formats

Rust ENBR v1 is described in `src/brain_io.rs`. The reference importer reads that native profile, Manifold ENDB v2 and ZIP v4–v8. Smaller formats do not contain the full reference state: `import_info` records unavailable fields instead of treating zero-filled storage as learned statistics. Exporting the full reference state to a smaller profile would require a loss policy and is not silently performed.

ZIP v8 retains V7 well/anchor, graph, per-well history and metadata sections. Its `dream.bin` stores a u32 entry count followed by K/G complex128 arrays and float64 RMS arrays per entry. Earlier ZIP versions used complex64/float32 here. Manifest state also retains `last_well` and `pairing_initialized`, preserving the next transition and an empty memory's uninitialized basis. Supported unknown manifest fields and opaque top-level sections survive read/write. New writes use V8; older consumer readers need an explicit migration before consuming it.

The old `ENGRAM_FORMAT_V1.md` under research remains a historical proposal, not the current format definition. `engram_memory_io.py` and `EngramFile.save` implement the reference archive.

The bundled [Alexandria research artifact](../../research/README.md#alexandria-research-artifact) is an older ZIP v4 snapshot that the current Python reader accepts. It is preserved in its original format. Manifold consumes an ENDB v2 representation of that same content, which is a separate consumer format from Rust ENBR.

## Limits

Reference packet limits are 256 MiB input, dim at most 4096, and at most 16,777,216 scalar trajectory values. Readers validate finite values, counts, permutation, block coverage, micro bounds and exact consumption; truncations and trailing data are rejected. Historical Rust decoding is preceded by a bounded checked-layout pass. ZIP imports also limit expanded data, section counts and duplicates. These are engineering limits and executable checks, not a security certification or a whole-program proof.
