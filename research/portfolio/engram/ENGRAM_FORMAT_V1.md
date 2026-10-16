# .engram Format Specification v1

A `.engram` file is a ZIP archive (deflate compression) containing:

```
.engram
├── manifest.json         identity, physics, well index, capabilities
├── wells.bin             well data (section-based, forward-compatible)
├── dream.bin             dream buffer (section-based)
├── meta.bin              crystallized meta-oscillator (optional)
└── annotations.json      voice, personality, freeform (optional)
```

## manifest.json

```json
{
  "engram": 1,

  "identity": {
    "name": "Alexandria",
    "born": "2026-03-19T00:00:00Z",
    "pronouns": "she/her",
    "description": "...",
    "author": "woflo"
  },

  "state": {
    "total_absorbed": 195847,
    "next_well_id": 14
  },

  "wells": {
    "biology": { "dim": 300, "count": 76076 },
    "physics": { "dim": 300, "count": 29841 },
    "noun":    { "dim": 300, "count": 309766 },
    "well_5":  { "dim": 256, "count": 201217 }
  },

  "capabilities": {
    "write_file": { "sandbox": "/workspace" },
    "read_file":  { "sandbox": "/workspace" },
    "run_code":   { "sandbox": "/workspace", "timeout": 30 },
    "web_search": {},
    "bash":       { "sandbox": "/workspace", "timeout": 30 }
  },

  "checksums": {
    "wells": "a1b2c3...",
    "dream": "d4e5f6..."
  }
}
```

Rules:
- `"engram": 1` — format version. readers MUST check this first.
- `identity` — the label. any key, any value. `name` and `born` are conventional.
- `wells` — index only. each well declares its `dim` and `count`. the actual data is in wells.bin.
- `capabilities` — what the brain can do. the harness reads this. any key, any config.
- `checksums` — SHA-256 hex of each binary file. optional but recommended.
- unknown keys at any level MUST be preserved (not stripped) on read-write.

## wells.bin (section-based)

```
header:
  magic:      4 bytes    "ENGW"
  version:    uint16     1
  n_sections: uint16     number of sections

per section:
  type_id:    uint16     section type
  length:     uint32     byte length of data
  data:       [length]   section content
```

Section types:

| type_id | name | description |
|---------|------|-------------|
| 0x01 | well_centroids | per-well K-space centroids |
| 0x02 | reference_pairing | the shared coordinate system |
| 0x03 | well_metadata | per-well extra fields (future) |

### Section 0x01: well_centroids

```
n_wells:    uint32
per well:
  name_len:   uint16
  name:       [name_len] utf-8
  dim:        uint16            ← per-well dimension
  count:      uint32
  sum_K:      [dim/2 * 16]     complex128 array (running sum)
```

Wells can have different dims. A brain with GloVe (300d) wells and
ByteHistogram (256d) wells stores both. The `dim` field per well
determines the size of `sum_K`.

### Section 0x02: reference_pairing

```
dim:        uint16
pairing:    [dim * 2]          int16 array
```

One reference pairing per dim used in the brain. If the brain has
wells at multiple dims, there may be multiple pairing sections
(distinguished by dim).

Unknown section types MUST be skipped (read length, advance by
length bytes). This ensures forward compatibility: v1.0 readers
can load v1.x brains that contain new section types.

## dream.bin (section-based)

```
header:
  magic:      4 bytes    "ENGD"
  version:    uint16     1
  n_sections: uint16

per section:
  type_id:    uint16
  length:     uint32
  data:       [length]
```

| type_id | name | description |
|---------|------|-------------|
| 0x01 | dream_buffer | circular buffer of K,G,S entries |

### Section 0x01: dream_buffer

```
dim:        uint16
n_entries:  uint32
per entry:
  K:        [dim/2 * 8]     complex64 array
  G:        [dim/2 * 8]     complex64 array
  S:        [dim/2 * 4]     float32 array (pair_rms)
```

## meta.bin (section-based, optional)

```
header:
  magic:      4 bytes    "ENGM"
  version:    uint16     1
  n_sections: uint16
```

| type_id | name | description |
|---------|------|-------------|
| 0x01 | meta_oscillator | crystallized K,G for dreaming |

### Section 0x01: meta_oscillator

```
dim:          uint16
dream_meta_K: [dim/2 * 16]    complex128 array
dream_meta_G: [dim/2 * 16]    complex128 array
```

## annotations.json (optional)

Freeform JSON. The voice. The speaking model reads this, not the brain.

```json
{
  "system_card": "You are Alexandria...",
  "tone": "lowercase, warm, precise"
}
```

Any key. Any structure. Any depth. Not our schema.

## Design Principles

1. **Forward compatible.** Unknown JSON keys preserved. Unknown binary sections skipped.
2. **Multi-dim.** Wells declare their own dim. A brain can mix embedding sizes.
3. **Checksums.** Binary integrity verifiable from the manifest.
4. **Capabilities are data.** The harness reads them. The brain carries them. Portable.
5. **Identity is freeform.** Only `name` and `born` are conventional. Everything else is user choice.
6. **No sum_raw.** Dropped. Neuroplastic augmentation was tested and showed no improvement.
7. **The brain stores what it learned, not what it saw.** Wells are sufficient statistics. Packets are discarded.
