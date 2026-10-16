"""Loss-aware import of Engram ZIP, Neuron ENBR and Manifold ENDB memories."""

import io
import json
import struct
from pathlib import Path
import zipfile
import numpy as np
from engram_wire import Reader, MAX_BYTES, MAX_VALUES, MAX_DIM

KNOWN = {
    "manifest.json",
    "wells.bin",
    "dream.bin",
    "annotations.json",
    "meta.bin",
    "transitions.json",
    "well_dreams.bin",
}


def _json(data):
    return json.loads(
        bytes(data).decode("utf-8"),
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON")),
    )


def _name(r):
    return bytes(r.take(r.scalar("<H"))).decode("utf-8")


def _new(manifest, location):
    from engram_codec import EngramFile

    if not isinstance(manifest, dict) or manifest.get("format") != "engram":
        raise ValueError("invalid memory manifest")
    physics = manifest.get("physics", {})
    d = physics.get("dim", 0)
    if (
        not isinstance(d, int)
        or not 2 <= d <= MAX_DIM
        or d % 2
        or physics.get("pairs", d // 2) != d // 2
    ):
        raise ValueError("invalid memory dimensions")
    alpha = physics.get("alpha", 0.005)
    if not np.isfinite(alpha):
        raise ValueError("invalid memory parameters")
    memory = EngramFile(location)
    ident = dict(manifest.get("identity", {}))
    memory.name = ident.pop("name", "")
    memory.born = ident.pop("born", "")
    memory.metadata = ident
    memory.dim, memory.alpha = d, float(alpha)
    state = manifest.get("state", {})
    for field in ["total_absorbed", "next_well_id"]:
        value = state.get(field, 0)
        if not isinstance(value, int) or value < 0:
            raise ValueError("invalid memory counter")
        setattr(memory, "_" + field, value)
    memory._source_manifest = manifest
    memory.import_info = dict(manifest.get("origin", {}))
    memory._extra_sections = {}
    return memory


def _basis(memory, values):
    basis = np.asarray(values, dtype=np.int32)
    if not np.array_equal(np.sort(basis), np.arange(memory.dim)):
        raise ValueError("invalid memory pairing")
    memory._reference_pairing = basis


def _well(memory, name, k, count, g=None, s=None, raw=None):
    if name in memory._well_sum_K or count <= 0:
        raise ValueError("duplicate well or invalid count")
    p = memory.dim // 2
    memory._well_sum_K[name] = k
    memory._well_sum_G[name] = np.zeros(p, dtype=complex) if g is None else g
    memory._well_sum_S[name] = np.zeros(p, dtype=float) if s is None else s
    memory._well_sum_raw[name] = np.zeros(256) if raw is None else raw
    memory._well_count[name] = count


def _dream(memory, r, count, precision="<c8", rms_precision="<f4"):
    p = memory.dim // 2
    if count * p * 5 > MAX_VALUES:
        raise ValueError("memory history exceeds value budget")
    for _ in range(count):
        memory._dream_Ks.append(r.array(p, precision).astype(complex))
        memory._dream_Gs.append(r.array(p, precision).astype(complex))
        values = r.array(p, rms_precision).astype(float)
        if np.any(values < 0):
            raise ValueError("negative residual RMS")
        memory._dream_Ss.append(values)


def _zip(data, location):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        infos = z.infolist()
        if (
            len(infos) > 64
            or len({i.filename for i in infos}) != len(infos)
            or sum(i.file_size for i in infos) > MAX_BYTES
            or any("/" in i.filename or "\\" in i.filename for i in infos)
        ):
            raise ValueError("memory archive exceeds budget or has duplicate sections")
        sections = {i.filename: z.read(i) for i in infos}
    manifest = _json(sections["manifest.json"])
    version = manifest.get("version", 4)
    if version not in (4, 5, 6, 7, 8):
        raise ValueError("unsupported ZIP memory version")
    memory = _new(manifest, location)
    p, d = memory.dim // 2, memory.dim
    r = Reader(sections["wells.bin"])
    _basis(memory, r.array(d, "<i2"))
    n = r.scalar("<I")
    if n > MAX_VALUES // max(1, p):
        raise ValueError("well count exceeds budget")
    for _ in range(n):
        name = _name(r)
        k = r.array(p, "<c16")
        g = r.array(p, "<c16") if version >= 5 else None
        s = r.array(p, "<f8") if version >= 5 else None
        raw = r.array(256, "<f8")
        count = r.scalar("<I")
        _well(memory, name, k, count, g, s, raw)
        if version >= 7:
            memory._well_anchor_K[name] = r.array(p, "<c16")
            memory._well_anchor_G[name] = r.array(p, "<c16")
            memory._well_anchor_S[name] = r.array(p, "<f8")
    r.finish()
    if manifest.get("state", {}).get("pairing_initialized", True) is False:
        if n:
            raise ValueError("learned wells need an initialized pairing")
        memory._reference_pairing = None
    if manifest.get("state", {}).get("n_wells", n) != n:
        raise ValueError("manifest and well count differ")
    r = Reader(sections["dream.bin"])
    _dream(
        memory,
        r,
        r.scalar("<I"),
        "<c16" if version >= 8 else "<c8",
        "<f8" if version >= 8 else "<f4",
    )
    r.finish()
    if "meta.bin" in sections:
        r = Reader(sections["meta.bin"])
        memory.dream_meta_K = r.array(p, "<c16")
        memory.dream_meta_G = r.array(p, "<c16")
        r.finish()
    if "transitions.json" in sections:
        transitions = _json(sections["transitions.json"])
        if not isinstance(transitions, dict) or any(
            not isinstance(bs, dict)
            or any(not isinstance(c, int) or c < 0 for c in bs.values())
            for bs in transitions.values()
        ):
            raise ValueError("invalid transition graph")
        memory._well_transitions = transitions
    if "well_dreams.bin" in sections:
        r = Reader(sections["well_dreams.bin"])
        n = r.scalar("<I")
        if n > len(memory._well_sum_K):
            raise ValueError("invalid per-well history count")
        for _ in range(n):
            name = _name(r)
            count = r.scalar("<I")
            if (
                name not in memory._well_sum_K
                or name in memory._well_dream_Ks
                or count * p > MAX_VALUES
            ):
                raise ValueError("invalid per-well history")
            memory._well_dream_Ks[name] = [r.array(p, "<c16") for _ in range(count)]
            flag = r.scalar("<B")
            if flag not in (0, 1):
                raise ValueError("invalid per-well meta flag")
            if flag:
                memory._well_meta_K[name] = r.array(p, "<c16")
                memory._well_meta_G[name] = r.array(p, "<c16")
        r.finish()
    if "annotations.json" in sections:
        memory.annotations = _json(sections["annotations.json"])
    memory._extra_sections = {k: v for k, v in sections.items() if k not in KNOWN}
    memory._last_well = manifest.get("state", {}).get("last_well")
    if memory._last_well is not None and memory._last_well not in memory._well_sum_K:
        raise ValueError("invalid last-well state")
    if version < 5 and not memory.import_info:
        memory.import_info = {
            "format": "ZIP",
            "version": version,
            "unavailable": [
                "per-well G/S",
                "routing anchors",
                "transition graph",
                "per-well histories",
            ],
        }
    return memory


def _native(data, location):
    r = Reader(data)
    r.take(4)
    version = r.scalar("<B") if data[:4] == b"ENBR" else r.scalar("<I")
    native = data[:4] == b"ENBR"
    if (native and version != 1) or (not native and version != 2):
        raise ValueError("unsupported memory version")
    d, p = r.unpack("<II")
    if not 2 <= d <= MAX_DIM or d % 2 or p != d // 2:
        raise ValueError("invalid memory dimensions")
    if native:
        alpha = r.scalar("<f")
        total, next_id = r.unpack("<II")
        name = _name(r)
        memory = _new(
            {
                "format": "engram",
                "physics": {"dim": d, "alpha": alpha},
                "identity": {"name": name},
                "state": {"total_absorbed": total, "next_well_id": next_id},
            },
            location,
        )
        flag = r.scalar("<B")
        if flag not in (0, 1):
            raise ValueError("invalid pairing flag")
        if flag:
            _basis(memory, r.array(d, "<i4"))
        n = r.scalar("<I")
        if n * p > MAX_VALUES:
            raise ValueError("well count exceeds budget")
        for _ in range(n):
            name = _name(r)
            count = r.scalar("<I")
            _well(memory, name, r.array(p, "<c16"), count)
        _dream(memory, r, r.scalar("<I"), "<c16")
        r.finish()
        memory.import_info = {
            "format": "ENBR",
            "version": 1,
            "unavailable": [
                "per-well G/S",
                "raw centroids",
                "routing anchors",
                "transition graph",
                "per-well histories",
                "annotations",
            ],
        }
    else:
        n, flags = r.unpack("<II")
        if flags & ~15 or n * p > MAX_VALUES:
            raise ValueError("invalid ENDB flags/count")
        memory = _new({"format": "engram", "physics": {"dim": d}}, location)
        _basis(memory, r.array(d, "<i2"))
        for _ in range(n):
            name = _name(r)
            count = r.scalar("<I")
            k = r.array(p, "<c16")
            raw = None
            if flags & 1:
                size = r.scalar("<I")
                if size != 256 * 8:
                    raise ValueError("unsupported raw centroid shape")
                raw = r.array(256, "<f8")
            _well(memory, name, k, count, raw=raw)
        if flags & 2:
            _dream(memory, r, r.scalar("<I"))
        if flags & 4:
            memory.annotations = _json(r.take(r.scalar("<I")))
        if flags & 8:
            manifest = _json(r.take(r.scalar("<I")))
            original = _new(manifest, location)
            if original.dim != d or manifest.get("state", {}).get("n_wells", n) != n:
                raise ValueError("ENDB manifest disagrees with payload")
            memory.name, memory.born, memory.metadata = (
                original.name,
                original.born,
                original.metadata,
            )
            memory.alpha = original.alpha
            memory._total_absorbed = original._total_absorbed
            memory._next_well_id = original._next_well_id
            memory._source_manifest = manifest
        r.finish()
        memory.import_info = {
            "format": "ENDB",
            "version": 2,
            "unavailable": [
                "per-well G/S",
                "routing anchors",
                "transition graph",
                "per-well histories",
            ],
        }
    return memory


def load_memory(source):
    location = str(source) if isinstance(source, (str, Path)) else ""
    if location:
        path = Path(source)
        if path.stat().st_size > MAX_BYTES:
            raise ValueError("memory exceeds byte budget")
        data = path.read_bytes()
    else:
        data = bytes(source)
    if len(data) > MAX_BYTES:
        raise ValueError("memory exceeds byte budget")
    try:
        return (
            _native(data, location)
            if data[:4] in (b"ENBR", b"ENDB")
            else _zip(data, location)
        )
    except (
        KeyError,
        struct.error,
        UnicodeError,
        zipfile.BadZipFile,
        OverflowError,
        TypeError,
        RecursionError,
    ) as error:
        raise ValueError("malformed memory snapshot") from error


def validate_memory(memory):
    d = memory.dim
    if (
        not isinstance(d, int)
        or not 2 <= d <= MAX_DIM
        or d % 2
        or not np.isfinite(memory.alpha)
    ):
        raise ValueError("invalid memory dimensions or parameters")
    if memory._reference_pairing is not None:
        _basis(memory, memory._reference_pairing)
    p = d // 2
    for name, k in memory._well_sum_K.items():
        if (
            not isinstance(name, str)
            or len(name.encode("utf-8")) > 65535
            or memory._well_count.get(name, 0) <= 0
        ):
            raise ValueError("invalid well name/count")
        if np.shape(k) != (p,) or not np.isfinite(k).all():
            raise ValueError("invalid K statistics")
        for attr, shape in [
            ("_well_sum_G", (p,)),
            ("_well_sum_S", (p,)),
            ("_well_sum_raw", (256,)),
            ("_well_anchor_K", (p,)),
            ("_well_anchor_G", (p,)),
            ("_well_anchor_S", (p,)),
            ("_well_meta_K", (p,)),
            ("_well_meta_G", (p,)),
        ]:
            values = getattr(memory, attr).get(name)
            if values is not None and (
                np.shape(values) != shape or not np.isfinite(values).all()
            ):
                raise ValueError("invalid well statistics")
    if not len(memory._dream_Ks) == len(memory._dream_Gs) == len(memory._dream_Ss):
        raise ValueError("inconsistent recent history")
    for values in [*memory._dream_Ks, *memory._dream_Gs, *memory._dream_Ss]:
        if np.shape(values) != (p,) or not np.isfinite(values).all():
            raise ValueError("invalid recent history")
