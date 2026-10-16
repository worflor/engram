"""Bounded reference packet format. V4 stores block state and its coordinate basis."""

import struct
import numpy as np

MAX_BYTES = 256 * 1024 * 1024
MAX_VALUES = 16 * 1024 * 1024
MAX_DIM = 4096


class Reader:
    def __init__(self, data):
        if len(data) > MAX_BYTES:
            raise ValueError("packet exceeds byte budget")
        self.data = memoryview(data)
        self.pos = 0

    def take(self, size):
        if size < 0 or size > len(self.data) - self.pos:
            raise ValueError("truncated packet")
        result = self.data[self.pos : self.pos + size]
        self.pos += size
        return result

    def unpack(self, fmt):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))

    def scalar(self, fmt):
        return self.unpack(fmt)[0]

    def array(self, count, dtype):
        if count < 0 or count > MAX_VALUES:
            raise ValueError("array exceeds value budget")
        dt = np.dtype(dtype)
        out = np.frombuffer(self.take(count * dt.itemsize), dtype=dt).copy()
        if not np.isfinite(out).all():
            raise ValueError("non-finite packet value")
        return out

    def finish(self):
        if self.pos != len(self.data):
            raise ValueError("trailing packet bytes")


def validate(packet, *, require_full=False):
    from engram_codec import Mode

    d, t, p = packet.dim, packet.length, packet.n_pairs
    if not isinstance(d, int) or d < 2 or d > MAX_DIM or d % 2 or p != d // 2:
        raise ValueError("invalid packet dimension")
    if t < 0 or t * d > MAX_VALUES or len(packet.blocks) > t:
        raise ValueError("packet exceeds value budget")
    if (
        not np.isfinite(packet.alpha)
        or not 0 < packet.macro_block <= 65535
        or not 0 < packet.micro_block <= 65535
    ):
        raise ValueError("invalid packet parameters")
    pairing = np.asarray(packet.pairing)
    if (
        pairing.shape != (d,)
        or not np.issubdtype(pairing.dtype, np.integer)
        or not np.array_equal(np.sort(pairing), np.arange(d))
    ):
        raise ValueError("invalid coordinate pairing")
    if require_full and not packet.reconstructable:
        raise ValueError("compact statistics cannot reconstruct a trajectory")
    start = packet.blocks[0].start if packet.blocks else 0
    if start not in (0, 2) or (t and not packet.blocks):
        raise ValueError("invalid initial block position")
    for b in packet.blocks:
        if not isinstance(b.micro_size, int) or not 0 <= b.micro_size <= 65535:
            raise ValueError("invalid micro span")
        if (
            b.mode not in list(Mode)
            or b.length <= 0
            or b.start != start
            or b.length * d > MAX_VALUES
        ):
            raise ValueError("invalid block coverage")
        start += b.length
        for a in (b.macro_K, b.macro_G):
            if np.shape(a) != (p,) or not np.isfinite(a).all():
                raise ValueError("invalid oscillator parameters")
        if b.pair_rms is not None and (
            np.shape(b.pair_rms) != (p,)
            or not np.isfinite(b.pair_rms).all()
            or np.any(b.pair_rms < 0)
        ):
            raise ValueError("invalid residual RMS")
        if (
            not all(
                np.isfinite(v)
                for v in (
                    b.signal_energy,
                    b.residual_energy,
                    b.macro_capture,
                    b.micro_capture,
                )
            )
            or b.signal_energy < 0
            or b.residual_energy < 0
        ):
            raise ValueError("invalid block measurements")
        if packet.reconstructable:
            for a in (b.seed1, b.seed2):
                if np.shape(a) != (d,) or not np.isfinite(a).all():
                    raise ValueError("invalid block initial conditions")
            if (
                np.shape(b.residuals) != (b.length, d)
                or not np.isfinite(b.residuals).all()
            ):
                raise ValueError("invalid residual shape")
            if (
                np.shape(b.scales) != (p,)
                or not np.isfinite(b.scales).all()
                or np.any(b.scales < 0)
            ):
                raise ValueError("invalid residual scales")
            if len(b.micro_Ks) != len(b.micro_Gs):
                raise ValueError("inconsistent micro predictors")
            for a in [*b.micro_Ks, *b.micro_Gs]:
                if np.shape(a) != (p,) or not np.isfinite(a).all():
                    raise ValueError("invalid micro predictor")
            if b.micro_Ks:
                span = b.micro_size or max(4, b.length // len(b.micro_Ks))
                if (len(b.micro_Ks) - 1) * span >= b.length:
                    raise ValueError("micro partition exceeds its block")
    if start != t:
        raise ValueError("block coverage does not match trajectory length")


def to_wire(packet, q_bits=8, compact=False):
    from engram_codec import Mode, quantize

    if q_bits != 8:
        raise ValueError("reference wire format uses 8-bit residuals")
    validate(packet, require_full=not compact)
    if not packet.basis_known:
        raise ValueError("attach the original coordinate basis before serialization")
    d, p = packet.dim, packet.n_pairs
    out = bytearray(b"EN\x04")
    out += struct.pack(
        "<BHI fIII",
        int(compact),
        d,
        packet.length,
        packet.alpha,
        packet.macro_block,
        packet.micro_block,
        len(packet.blocks),
    )
    out += np.asarray(packet.pairing, dtype="<u2").tobytes()
    for b in packet.blocks:
        span = b.micro_size or (
            max(4, b.length // len(b.micro_Ks)) if b.micro_Ks else 0
        )
        out += struct.pack(
            "<BIII4d",
            int(b.mode),
            b.start,
            b.length,
            span,
            b.signal_energy,
            b.residual_energy,
            b.macro_capture,
            b.micro_capture,
        )
        dtype = "<c8" if compact else "<c16"
        for values in (b.macro_K, b.macro_G):
            encoded = np.asarray(values, dtype=dtype)
            if not np.isfinite(encoded).all():
                raise ValueError("oscillator exceeds wire precision")
            out += encoded.tobytes()
        out.append(int(b.pair_rms is not None))
        if b.pair_rms is not None:
            out += np.asarray(b.pair_rms, dtype="<f4").tobytes()
        if compact:
            continue
        out += (
            np.asarray(b.seed1, dtype="<f4").tobytes()
            + np.asarray(b.seed2, dtype="<f4").tobytes()
        )
        out += struct.pack("<I", len(b.micro_Ks))
        for k, g in zip(b.micro_Ks, b.micro_Gs):
            out += (
                np.asarray(k, dtype="<c16").tobytes()
                + np.asarray(g, dtype="<c16").tobytes()
            )
        if b.mode == Mode.RAW:
            scales = np.zeros(p, dtype="<f4")
            payload = np.asarray(b.residuals, dtype="<f4").tobytes()
        elif b.mode == Mode.STATIC:
            scales = np.zeros(p, dtype="<f4")
            payload = b""
        else:
            q, scales = quantize(b.residuals)
            payload = q.astype("i1").tobytes()
        out += (
            np.asarray(scales, dtype="<f4").tobytes()
            + struct.pack("<I", len(payload))
            + payload
        )
    if len(out) > MAX_BYTES:
        raise ValueError("packet exceeds byte budget")
    return bytes(out)


def from_wire(data, q_bits=8, *, legacy_v1=None):
    from engram_codec import Block, Packet, Mode, dequantize

    if q_bits != 8:
        raise ValueError("reference wire format uses 8-bit residuals")
    r = Reader(data)
    if bytes(r.take(2)) != b"EN":
        raise ValueError("bad packet magic")
    version = r.scalar("<B")
    if version == 1 and legacy_v1 != "python":
        raise ValueError(
            "ambiguous legacy v1; explicitly select legacy_v1='python' for Python-produced data"
        )
    if version in (1, 2, 3):
        return _legacy(r, version)
    if version != 4:
        raise ValueError("unsupported packet version")
    flags, d, t, alpha, macro, micro, n = r.unpack("<BHI fIII")
    if (
        flags not in (0, 1)
        or d < 2
        or d > MAX_DIM
        or d % 2
        or t * d > MAX_VALUES
        or n > t
    ):
        raise ValueError("invalid packet header or value budget")
    p = d // 2
    pairing = r.array(d, "<u2").astype(np.int32)
    blocks = []
    for _ in range(n):
        mode, start, length, span, se, re, mc, ic = r.unpack("<BIII4d")
        if length == 0 or length * d > MAX_VALUES or span > 65535:
            raise ValueError("invalid block size")
        mode = Mode(mode)
        dtype = "<c8" if flags else "<c16"
        k, g = r.array(p, dtype).astype(complex), r.array(p, dtype).astype(complex)
        present = r.scalar("<B")
        if present not in (0, 1):
            raise ValueError("invalid RMS flag")
        rms = r.array(p, "<f4") if present else None
        s1, s2 = np.zeros(d, dtype=np.float32), np.zeros(d, dtype=np.float32)
        ks, gs = [], []
        scales = np.zeros(p, dtype=np.float32)
        resid = np.empty((0, d), dtype=np.float32)
        if not flags:
            s1, s2 = r.array(d, "<f4"), r.array(d, "<f4")
            count = r.scalar("<I")
            if count > length or count * p * 2 > MAX_VALUES or (count and not span):
                raise ValueError("invalid micro count")
            for _ in range(count):
                ks.append(r.array(p, "<c16"))
                gs.append(r.array(p, "<c16"))
            scales = r.array(p, "<f4")
            size = r.scalar("<I")
            expected = (
                length * d * 4
                if mode == Mode.RAW
                else (0 if mode == Mode.STATIC else length * d)
            )
            if size != expected:
                raise ValueError("invalid residual byte count")
            if mode == Mode.RAW:
                resid = r.array(length * d, "<f4").reshape(length, d)
            elif mode == Mode.STATIC:
                resid = np.zeros((length, d), dtype=np.float32)
            else:
                q = r.array(length * d, "i1").reshape(length, d)
                if np.any(q == -128):
                    raise ValueError("invalid quantized residual")
                resid = dequantize(q, scales)
        blocks.append(
            Block(
                mode,
                start,
                length,
                k,
                g,
                ks,
                gs,
                resid,
                s1,
                s2,
                scales,
                se,
                re,
                mc,
                ic,
                rms,
                span,
            )
        )
    r.finish()
    packet = Packet(d, p, t, alpha, macro, micro, pairing, blocks, not flags)
    validate(packet)
    return packet


def _legacy(r, version):
    from engram_codec import Block, Packet, Mode, dequantize, decode_block

    d, t, alpha, macro, micro, n = r.unpack("<HI fHHH")
    if d < 2 or d > MAX_DIM or d % 2 or t * d > MAX_VALUES or n > t:
        raise ValueError("invalid legacy header")
    p, compact = d // 2, version == 2
    pairing = (
        np.arange(d, dtype=np.int32) if compact else r.array(d, "<u2").astype(np.int32)
    )
    s1 = np.zeros(d, dtype=np.float32) if compact else r.array(d, "<f4")
    s2 = np.zeros(d, dtype=np.float32) if compact else r.array(d, "<f4")
    prev_k, prev_g = np.zeros(p, dtype=complex), np.zeros(p, dtype=complex)
    start, blocks = 0, []
    for _ in range(n):
        mode, length = r.unpack("<BH")
        mode = Mode(mode)
        if not length or length * d > MAX_VALUES:
            raise ValueError("invalid legacy block size")
        k, g = np.full(p, 2 + 0j), np.full(p, 1 + 0j)
        ks, gs, rms = [], [], None
        if mode == Mode.CASCADED:
            reset = r.scalar("<B")
            if reset == 1:
                k, g = (
                    r.array(p, "<c8").astype(complex),
                    r.array(p, "<c8").astype(complex),
                )
            elif reset == 0:
                dr, di, gr, gi = [r.array(p, "<i2") / 16384 for _ in range(4)]
                k, g = prev_k + dr + 1j * di, prev_g + gr + 1j * gi
            else:
                raise ValueError("invalid reset flag")
            prev_k, prev_g = k.copy(), g.copy()
            if compact:
                rms = r.array(p, "<f2").astype(np.float32)
            else:
                count = r.scalar("<H")
                if count > length:
                    raise ValueError("invalid micro count")
                for _ in range(count):
                    ks.append(r.array(p, "<c8").astype(complex))
                    gs.append(r.array(p, "<c8").astype(complex))
        elif mode == Mode.LINEAR and compact:
            rms = r.array(p, "<f2").astype(np.float32)
        elif mode == Mode.STATIC:
            k, g = np.ones(p, dtype=complex), np.zeros(p, dtype=complex)
        scales, resid = (
            np.zeros(p, dtype=np.float32),
            np.empty((0, d), dtype=np.float32),
        )
        if not compact:
            scales = r.array(p, "<f4")
            payload = Reader(r.take(r.scalar("<I")))
            values = []
            for _ in range(length * d):
                v = 0
                for shift in range(0, 35, 7):
                    b = payload.scalar("<B")
                    v |= (b & 127) << shift
                    if not b & 128:
                        break
                else:
                    raise ValueError("overlong residual varint")
                signed = (v >> 1) ^ -(v & 1)
                if not -127 <= signed <= 127:
                    raise ValueError("invalid residual varint")
                values.append(signed)
            payload.finish()
            resid = dequantize(
                np.array(values, dtype=np.int8).reshape(length, d), scales
            )
        b = Block(
            mode,
            start,
            length,
            k,
            g,
            ks,
            gs,
            resid,
            s1.copy(),
            s2.copy(),
            scales,
            pair_rms=rms,
        )
        blocks.append(b)
        start += length
        if not compact:
            decoded = decode_block(b)
            s2, s1 = (decoded[-2] if length >= 2 else s1).copy(), decoded[-1].copy()
    r.finish()
    if t >= 2 and start == t - 2:
        for b in blocks:
            b.start += 2
    packet = Packet(
        d, p, t, alpha, macro, micro, pairing, blocks, not compact, not compact
    )
    validate(packet)
    return packet
