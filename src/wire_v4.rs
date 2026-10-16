//! Explicit, bounded packet state shared with the Python reference.
use crate::types::{Block, Mode, Packet};
use num_complex::Complex64;

pub(crate) const MAX_BYTES: usize = 256 * 1024 * 1024;
const MAX_VALUES: usize = 16 * 1024 * 1024;
const MAX_DIM: usize = 4096;

pub(crate) struct Reader<'a> {
    pub data: &'a [u8],
    pub pos: usize,
}
impl<'a> Reader<'a> {
    pub fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(n)?;
        let value = self.data.get(self.pos..end)?;
        self.pos = end;
        Some(value)
    }
    pub fn u8(&mut self) -> Option<u8> {
        Some(*self.take(1)?.first()?)
    }
    pub fn u16(&mut self) -> Option<usize> {
        Some(u16::from_le_bytes(self.take(2)?.try_into().ok()?) as usize)
    }
    pub fn u32(&mut self) -> Option<usize> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?) as usize)
    }
    pub fn f32(&mut self) -> Option<f32> {
        let v = f32::from_le_bytes(self.take(4)?.try_into().ok()?);
        v.is_finite().then_some(v)
    }
    pub fn f64(&mut self) -> Option<f64> {
        let v = f64::from_le_bytes(self.take(8)?.try_into().ok()?);
        v.is_finite().then_some(v)
    }
    pub fn floats(&mut self, n: usize) -> Option<Vec<f32>> {
        if n > MAX_VALUES || n > self.data.len().saturating_sub(self.pos) / 4 {
            return None;
        }
        (0..n).map(|_| self.f32()).collect()
    }
    pub fn complex(&mut self, n: usize) -> Option<Vec<Complex64>> {
        if n > MAX_VALUES || n > self.data.len().saturating_sub(self.pos) / 16 {
            return None;
        }
        (0..n)
            .map(|_| Some(Complex64::new(self.f64()?, self.f64()?)))
            .collect()
    }
    pub fn complex32(&mut self, n: usize) -> Option<Vec<Complex64>> {
        if n > MAX_VALUES || n > self.data.len().saturating_sub(self.pos) / 8 {
            return None;
        }
        (0..n)
            .map(|_| {
                Some(Complex64::new(
                    f64::from(self.f32()?),
                    f64::from(self.f32()?),
                ))
            })
            .collect()
    }
}

pub(crate) fn valid(packet: &Packet, require_full: bool) -> bool {
    let d = packet.dim;
    let p = packet.pairs;
    if !(2..=MAX_DIM).contains(&d)
        || !d.is_multiple_of(2)
        || p != d / 2
        || packet.length > MAX_VALUES / d
        || packet.blocks.len() > packet.length
        || !packet.alpha.is_finite()
        || !(1..=65535).contains(&packet.macro_block)
        || !(1..=65535).contains(&packet.micro_block)
        || (require_full && !packet.reconstructable)
        || packet.pairing.len() != d
    {
        return false;
    }
    let mut pairing = packet.pairing.clone();
    pairing.sort_unstable();
    if pairing.iter().enumerate().any(|(i, &v)| v != i as i32) {
        return false;
    }
    let mut start = packet.blocks.first().map_or(0, |b| b.start);
    if ![0, 2].contains(&start) || (packet.length > 0 && packet.blocks.is_empty()) {
        return false;
    }
    for b in &packet.blocks {
        if b.micro_size > 65535 || b.micro_ks.len() > b.length {
            return false;
        }
        if b.length == 0 || b.length > MAX_VALUES / d || b.start != start {
            return false;
        }
        let Some(end) = start.checked_add(b.length) else {
            return false;
        };
        start = end;
        if b.macro_k.len() != p
            || b.macro_g.len() != p
            || b.macro_k.iter().chain(&b.macro_g).any(|z| !z.is_finite())
            || [
                b.signal_energy,
                b.residual_energy,
                b.macro_capture,
                b.micro_capture,
            ]
            .iter()
            .any(|v| !v.is_finite())
            || b.signal_energy < 0.
            || b.residual_energy < 0.
            || b.pair_rms
                .as_ref()
                .is_some_and(|v| v.len() != p || v.iter().any(|x| !x.is_finite() || *x < 0.))
        {
            return false;
        }
        if packet.reconstructable
            && (b.seed1.len() != d
                || b.seed2.len() != d
                || b.residuals.len() != b.length * d
                || b.scales.len() != p
                || b.seed1
                    .iter()
                    .chain(&b.seed2)
                    .chain(&b.residuals)
                    .any(|v| !v.is_finite())
                || b.scales.iter().any(|v| !v.is_finite() || *v < 0.)
                || b.micro_ks.len() != b.micro_gs.len()
                || b.micro_ks
                    .iter()
                    .chain(&b.micro_gs)
                    .any(|v| v.len() != p || v.iter().any(|z| !z.is_finite()))
                || (!b.micro_ks.is_empty()
                    && (b.micro_ks.len() - 1)
                        * if b.micro_size > 0 {
                            b.micro_size
                        } else {
                            (b.length / b.micro_ks.len()).max(4)
                        }
                        >= b.length))
        {
            return false;
        }
    }
    start == packet.length
}

fn w32(out: &mut Vec<u8>, v: usize) {
    out.extend_from_slice(&(v as u32).to_le_bytes());
}
fn floats(out: &mut Vec<u8>, v: &[f32]) {
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
}
fn complex(out: &mut Vec<u8>, v: &[Complex64]) {
    for z in v {
        out.extend_from_slice(&z.re.to_le_bytes());
        out.extend_from_slice(&z.im.to_le_bytes());
    }
}

pub(crate) fn write(packet: &Packet, compact: bool) -> Vec<u8> {
    assert!(
        packet.basis_known,
        "attach the original coordinate basis before serialization"
    );
    assert!(valid(packet, !compact), "invalid packet state");
    let mut out = b"EN\x04".to_vec();
    out.push(u8::from(compact));
    out.extend_from_slice(&(packet.dim as u16).to_le_bytes());
    w32(&mut out, packet.length);
    floats(&mut out, &[packet.alpha]);
    w32(&mut out, packet.macro_block);
    w32(&mut out, packet.micro_block);
    w32(&mut out, packet.blocks.len());
    for &i in &packet.pairing {
        out.extend_from_slice(&(i as u16).to_le_bytes());
    }
    for b in &packet.blocks {
        let span = if b.micro_size > 0 {
            b.micro_size
        } else if b.micro_ks.is_empty() {
            0
        } else {
            (b.length / b.micro_ks.len()).max(4)
        };
        out.push(b.mode as u8);
        w32(&mut out, b.start);
        w32(&mut out, b.length);
        w32(&mut out, span);
        for v in [
            b.signal_energy,
            b.residual_energy,
            b.macro_capture,
            b.micro_capture,
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        if compact {
            for z in b.macro_k.iter().chain(&b.macro_g) {
                let re = z.re as f32;
                let im = z.im as f32;
                assert!(
                    re.is_finite() && im.is_finite(),
                    "oscillator exceeds wire precision"
                );
                floats(&mut out, &[re, im]);
            }
        } else {
            complex(&mut out, &b.macro_k);
            complex(&mut out, &b.macro_g);
        }
        out.push(u8::from(b.pair_rms.is_some()));
        if let Some(rms) = &b.pair_rms {
            floats(&mut out, rms);
        }
        if compact {
            continue;
        }
        floats(&mut out, &b.seed1);
        floats(&mut out, &b.seed2);
        w32(&mut out, b.micro_ks.len());
        for (k, g) in b.micro_ks.iter().zip(&b.micro_gs) {
            complex(&mut out, k);
            complex(&mut out, g);
        }
        let scales: Vec<f32> = b.scales.iter().map(|v| v * 127.).collect();
        floats(&mut out, &scales);
        match b.mode {
            Mode::Raw => {
                w32(&mut out, b.residuals.len() * 4);
                floats(&mut out, &b.residuals);
            }
            Mode::Static => w32(&mut out, 0),
            _ => {
                w32(&mut out, b.residuals.len());
                for &v in &b.residuals {
                    assert!(
                        (-127. ..=127.).contains(&v) && v.fract() == 0.,
                        "invalid quantized residual"
                    );
                    out.push(v as i8 as u8);
                }
            }
        }
    }
    assert!(out.len() <= MAX_BYTES, "packet exceeds byte budget");
    out
}

pub(crate) fn read(data: &[u8], expect_compact: bool) -> Option<Packet> {
    if data.len() > MAX_BYTES {
        return None;
    }
    let mut r = Reader { data, pos: 0 };
    if r.take(3)? != b"EN\x04" {
        return None;
    }
    let compact = match r.u8()? {
        0 => false,
        1 => true,
        _ => return None,
    };
    if compact != expect_compact {
        return None;
    }
    let d = r.u16()?;
    let t = r.u32()?;
    let alpha = r.f32()?;
    let macro_block = r.u32()?;
    let micro_block = r.u32()?;
    let n = r.u32()?;
    if !(2..=MAX_DIM).contains(&d) || !d.is_multiple_of(2) || t > MAX_VALUES / d || n > t {
        return None;
    }
    let p = d / 2;
    let pairing: Vec<i32> = (0..d)
        .map(|_| Some(r.u16()? as i32))
        .collect::<Option<_>>()?;
    let mut blocks = Vec::new();
    for _ in 0..n {
        let mode = match r.u8()? {
            0 => Mode::Cascaded,
            1 => Mode::Linear,
            2 => Mode::Static,
            3 => Mode::Raw,
            _ => return None,
        };
        let start = r.u32()?;
        let length = r.u32()?;
        let span = r.u32()?;
        if length == 0 || length > MAX_VALUES / d || span > 65535 {
            return None;
        }
        let se = r.f64()?;
        let re = r.f64()?;
        let mc = r.f64()?;
        let ic = r.f64()?;
        let (k, g) = if compact {
            (r.complex32(p)?, r.complex32(p)?)
        } else {
            (r.complex(p)?, r.complex(p)?)
        };
        let rms = match r.u8()? {
            0 => None,
            1 => Some(r.floats(p)?),
            _ => return None,
        };
        let mut b = Block {
            mode,
            start,
            length,
            micro_size: span,
            macro_k: k,
            macro_g: g,
            micro_ks: vec![],
            micro_gs: vec![],
            seed1: vec![0.; d],
            seed2: vec![0.; d],
            scales: vec![0.; p],
            residuals: vec![],
            pair_rms: rms,
            signal_energy: se,
            residual_energy: re,
            macro_capture: mc,
            micro_capture: ic,
        };
        if !compact {
            b.seed1 = r.floats(d)?;
            b.seed2 = r.floats(d)?;
            let count = r.u32()?;
            if count > length
                || count.saturating_mul(p).saturating_mul(2) > MAX_VALUES
                || (count > 0 && span == 0)
            {
                return None;
            }
            for _ in 0..count {
                b.micro_ks.push(r.complex(p)?);
                b.micro_gs.push(r.complex(p)?);
            }
            b.scales = r.floats(p)?.iter().map(|v| v / 127.).collect();
            let size = r.u32()?;
            let expected = match mode {
                Mode::Raw => length * d * 4,
                Mode::Static => 0,
                _ => length * d,
            };
            if size != expected {
                return None;
            }
            b.residuals = match mode {
                Mode::Raw => r.floats(length * d)?,
                Mode::Static => vec![0.; length * d],
                _ => {
                    let raw = r.take(size)?;
                    if raw.contains(&128) {
                        return None;
                    }
                    raw.iter().map(|v| f32::from(*v as i8)).collect()
                }
            };
        }
        blocks.push(b);
    }
    let packet = Packet {
        dim: d,
        pairs: p,
        length: t,
        alpha,
        macro_block,
        micro_block,
        pairing,
        blocks,
        reconstructable: !compact,
        basis_known: true,
    };
    (r.pos == data.len() && valid(&packet, false)).then_some(packet)
}

/// Bounds-check the historical layout before its decoder can allocate or index.
pub(crate) fn legacy_valid(data: &[u8], expect_compact: bool) -> bool {
    fn check(data: &[u8], expect_compact: bool) -> Option<()> {
        if data.len() > MAX_BYTES {
            return None;
        }
        let mut r = Reader { data, pos: 0 };
        if r.take(2)? != b"EN" {
            return None;
        }
        let version = r.u8()?;
        if ![2, 3].contains(&version) {
            return None;
        }
        let compact = version == 2;
        if compact != expect_compact {
            return None;
        }
        let d = r.u16()?;
        let t = r.u32()?;
        r.f32()?;
        let macro_size = r.u16()?;
        let micro = r.u16()?;
        let n = r.u16()?;
        if !(2..=MAX_DIM).contains(&d)
            || !d.is_multiple_of(2)
            || t > MAX_VALUES / d
            || n > t
            || macro_size == 0
            || micro == 0
        {
            return None;
        }
        let p = d / 2;
        if !compact {
            let mut pairing: Vec<usize> = (0..d).map(|_| r.u16()).collect::<Option<_>>()?;
            pairing.sort_unstable();
            if pairing.iter().enumerate().any(|(i, &v)| i != v) {
                return None;
            }
            r.floats(d * 2)?;
        }
        let mut covered = 0usize;
        for _ in 0..n {
            let mode = r.u8()?;
            let length = r.u16()?;
            if mode > 3 || length == 0 || length > MAX_VALUES / d {
                return None;
            }
            covered = covered.checked_add(length)?;
            if covered > t {
                return None;
            }
            if mode == 0 {
                match r.u8()? {
                    1 => {
                        r.floats(p * 4)?;
                    }
                    0 => {
                        r.take(p * 8)?;
                    }
                    _ => return None,
                }
                if !compact {
                    let count = r.u16()?;
                    if count > length.div_euclid(4).max(1) {
                        return None;
                    }
                    r.floats(count * p * 4)?;
                }
            }
            if compact && (mode == 0 || mode == 1) {
                for _ in 0..p {
                    let bits = r.u16()?;
                    if bits & 0x7c00 == 0x7c00 {
                        return None;
                    }
                }
            }
            if !compact {
                if r.floats(p)?.iter().any(|v| *v < 0.) {
                    return None;
                }
                let size = r.u32()?;
                let payload = r.take(size)?;
                let mut q = Reader {
                    data: payload,
                    pos: 0,
                };
                for _ in 0..length * d {
                    let mut value = 0u32;
                    let mut done = false;
                    for shift in (0..35).step_by(7) {
                        let b = q.u8()?;
                        if shift == 28 && b > 15 {
                            return None;
                        }
                        value |= u32::from(b & 127) << shift;
                        if b & 128 == 0 {
                            done = true;
                            break;
                        }
                    }
                    if !done || value > 254 {
                        return None;
                    }
                }
                if q.pos != payload.len() {
                    return None;
                }
            }
        }
        if r.pos != data.len() || !(covered == t || (t >= 2 && covered == t - 2)) {
            return None;
        }
        Some(())
    }
    check(data, expect_compact).is_some()
}
