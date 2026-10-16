//! Diagnostics and signatures shared with the functional reference.
use crate::{
    fit::fit_pair,
    predict::to_complex,
    segment::{apply_pairing, derive_pairing},
};
use num_complex::Complex64;

#[derive(Debug, Clone)]
pub struct Dynamics {
    pub roots: [Complex64; 2],
    pub spectral_radius: f64,
    pub half_life_samples: Option<f64>,
    pub oscillation_period_samples: Option<f64>,
    pub classification: &'static str,
    pub repeated_root: bool,
}

fn max_component(z: Complex64) -> f64 {
    z.re.abs().max(z.im.abs())
}

fn binary_exponent(value: f64) -> Option<i32> {
    let bits = value.abs().to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    if exponent != 0 {
        Some(exponent - 1023)
    } else if fraction != 0 {
        Some((63 - fraction.leading_zeros() as i32) - 1074)
    } else {
        None
    }
}

fn scale_pow2(value: f64, mut exponent: i32, reject_underflow: bool) -> Result<f64, &'static str> {
    if value == 0. {
        return Ok(value);
    }
    let original = value;
    let mut result = value;
    while exponent > 0 {
        let step = exponent.min(1023);
        result *= 2_f64.powi(step);
        exponent -= step;
    }
    while exponent < 0 {
        let step = (-exponent).min(1022);
        result *= 2_f64.powi(-step);
        exponent += step;
    }
    if !result.is_finite() || (reject_underflow && result == 0. && original != 0.) {
        return Err("unrepresentable dynamics roots");
    }
    Ok(result)
}

fn scale_complex_pow2(
    value: Complex64,
    exponent: i32,
    reject_underflow: bool,
) -> Result<Complex64, &'static str> {
    Ok(Complex64::new(
        scale_pow2(value.re, exponent, reject_underflow)?,
        scale_pow2(value.im, exponent, reject_underflow)?,
    ))
}

fn divide_by_scaled_denominator(
    numerator: Complex64,
    denominator: Complex64,
) -> Result<Complex64, &'static str> {
    let scale = max_component(denominator);
    if scale == 0. || !scale.is_finite() {
        return Err("unrepresentable dynamics roots");
    }
    let scaled_numerator = Complex64::new(numerator.re / scale, numerator.im / scale);
    let scaled_denominator = denominator / scale;
    if !scaled_numerator.is_finite() || !scaled_denominator.is_finite() {
        return Err("unrepresentable dynamics roots");
    }
    let result = scaled_numerator / scaled_denominator;
    if result.is_finite() {
        Ok(result)
    } else {
        Err("unrepresentable dynamics roots")
    }
}

fn scaled_root_exponent(k: Complex64, g: Complex64) -> i32 {
    let k_exponent = binary_exponent(max_component(k));
    let g_exponent = binary_exponent(max_component(g)).map(|exponent| (exponent + 1).div_euclid(2));
    k_exponent.into_iter().chain(g_exponent).max().unwrap_or(0)
}

fn quadratic_roots(k: Complex64, g: Complex64) -> Result<[Complex64; 2], &'static str> {
    let exponent = scaled_root_exponent(k, g);
    let scaled_k = scale_complex_pow2(k, -exponent, false)?;
    let scaled_g = scale_complex_pow2(g, -2 * exponent, false)?;
    let discriminant = scaled_k * scaled_k - 4. * scaled_g;
    if !discriminant.is_finite() {
        return Err("unrepresentable dynamics roots");
    }
    let delta = discriminant.sqrt();
    let plus = scaled_k + delta;
    let minus = scaled_k - delta;
    let (large, plus_branch) = if plus.norm() >= minus.norm() {
        (plus / 2., true)
    } else {
        (minus / 2., false)
    };
    let large = scale_complex_pow2(large, exponent, true)?;
    let other = if g == Complex64::ZERO {
        Complex64::ZERO
    } else {
        divide_by_scaled_denominator(g, large)?
    };
    if !large.is_finite()
        || !other.is_finite()
        || (g != Complex64::ZERO && other == Complex64::ZERO)
    {
        return Err("unrepresentable dynamics roots");
    }
    let roots = if plus_branch {
        [large, other]
    } else {
        [other, large]
    };
    if !roots.iter().all(|root| root.norm().is_finite()) {
        return Err("unrepresentable dynamics roots");
    }
    Ok(roots)
}

pub fn dynamics(k: Complex64, g: Complex64, tolerance: f64) -> Result<Dynamics, &'static str> {
    if !k.is_finite() || !g.is_finite() || !tolerance.is_finite() || tolerance < 0. {
        return Err("invalid dynamics input");
    }
    let [a, b] = quadratic_roots(k, g)?;
    let radius = a.norm().max(b.norm());
    if !radius.is_finite() {
        return Err("unrepresentable dynamics roots");
    }
    let classification = if radius < 1. - tolerance {
        "decaying"
    } else if radius > 1. + tolerance {
        "growing"
    } else {
        "unit-radius"
    };
    let half_life_samples = if radius > 0. && radius < 1. {
        Some(-2_f64.ln() / radius.ln())
    } else {
        None
    };
    let oscillation_period_samples = if k.im.abs() <= tolerance
        && g.im.abs() <= tolerance
        && g.re > 0.
        && k.re.abs() / 2. < g.re.sqrt()
    {
        let omega = a.arg().abs();
        if omega > 0. {
            Some(2. * std::f64::consts::PI / omega)
        } else {
            None
        }
    } else {
        None
    };
    let root_scale =
        a.re.abs()
            .max(a.im.abs())
            .max(b.re.abs())
            .max(b.im.abs())
            .max(1.);
    let normalized_a = a / root_scale;
    let normalized_b = b / root_scale;
    let repeated_root = (normalized_a - normalized_b).norm()
        <= tolerance
            * normalized_a
                .norm()
                .max(normalized_b.norm())
                .max(1. / root_scale);
    Ok(Dynamics {
        roots: [a, b],
        spectral_radius: radius,
        half_life_samples,
        oscillation_period_samples,
        classification,
        repeated_root,
    })
}

pub fn fit_ar1(z: &[Complex64]) -> Result<(Complex64, f64), &'static str> {
    if z.len() < 2 || z.iter().any(|v| !v.is_finite()) {
        return Err("AR(1) needs two finite samples");
    }
    let energy = z[..z.len() - 1]
        .iter()
        .map(Complex64::norm_sqr)
        .sum::<f64>();
    let ridge = f64::from(f32::EPSILON).max(1e-4 * energy);
    let k = z.windows(2).map(|w| w[0].conj() * w[1]).sum::<Complex64>() / (energy + ridge);
    let rms = (z
        .windows(2)
        .map(|w| (w[1] - k * w[0]).norm_sqr())
        .sum::<f64>()
        / (z.len() - 1) as f64)
        .sqrt();
    Ok((k, rms))
}

pub fn cosine(a: &[Complex64], b: &[Complex64]) -> Result<f64, &'static str> {
    if a.len() != b.len() || a.iter().chain(b).any(|z| !z.is_finite()) {
        return Err("equal finite vectors required");
    }
    let scale_a = a
        .iter()
        .fold(0_f64, |scale, value| scale.max(max_component(*value)));
    let scale_b = b
        .iter()
        .fold(0_f64, |scale, value| scale.max(max_component(*value)));
    if scale_a == 0. || scale_b == 0. {
        return Ok(0.);
    }
    let scaled_a = a.iter().map(|value| *value / scale_a);
    let scaled_b = b.iter().map(|value| *value / scale_b);
    let (mut dot, mut norm_a, mut norm_b) = (0., 0., 0.);
    for (left, right) in scaled_a.zip(scaled_b) {
        dot += (left.conj() * right).re;
        norm_a += left.norm_sqr();
        norm_b += right.norm_sqr();
    }
    let similarity = dot / norm_a.sqrt() / norm_b.sqrt();
    if !similarity.is_finite() {
        return Err("unrepresentable cosine result");
    }
    Ok(similarity.clamp(-1., 1.))
}

#[derive(Debug, Clone)]
pub struct Signature {
    pub k: Vec<Complex64>,
    pub g: Vec<Complex64>,
    pub pair_rms: Vec<f64>,
    pub pairing: Vec<i32>,
    pub samples: usize,
    pub order: usize,
}
#[derive(Debug, Clone)]
pub struct Comparison {
    pub k_cosine: f64,
    pub g_cosine: f64,
    pub joint_cosine: f64,
    pub kg_rms_distance: f64,
}
impl Signature {
    pub fn from_trajectory(
        x: &[f32],
        t: usize,
        dim: usize,
        pairing: Option<&[i32]>,
    ) -> Result<Self, &'static str> {
        crate::validation::trajectory(x, t, dim, pairing)?;
        if t < 2 {
            return Err("signature needs two samples");
        }
        let basis = pairing.map_or_else(|| derive_pairing(x, t, dim), <[i32]>::to_vec);
        let paired = apply_pairing(x, t, dim, &basis);
        let z = to_complex(&paired, t, dim);
        let p = dim / 2;
        let mut k = Vec::with_capacity(p);
        let mut g = Vec::with_capacity(p);
        let mut pair_rms = Vec::with_capacity(p);
        for j in 0..p {
            let values: Vec<Complex64> = (0..t).map(|i| z[i * p + j]).collect();
            let (a, b, r) = if t == 2 {
                let (a, r) = fit_ar1(&values)?;
                (a, Complex64::ZERO, r)
            } else {
                let f = fit_pair(&values);
                (f.k, f.g, f.rms)
            };
            k.push(a);
            g.push(b);
            pair_rms.push(r);
        }
        Ok(Self {
            k,
            g,
            pair_rms,
            pairing: basis,
            samples: t,
            order: if t == 2 { 1 } else { 2 },
        })
    }
    pub fn compare(&self, other: &Self) -> Result<Comparison, &'static str> {
        if self.pairing != other.pairing || self.order != other.order {
            return Err("signature basis/order mismatch");
        }
        let k = cosine(&self.k, &other.k)?;
        let g = cosine(&self.g, &other.g)?;
        if self.k.is_empty() || self.g.len() != self.k.len() || other.g.len() != self.k.len() {
            return Err("invalid signature shape");
        }
        let distance = (self
            .k
            .iter()
            .zip(&other.k)
            .zip(self.g.iter().zip(&other.g))
            .map(|((a, b), (c, d))| (a - b).norm_sqr() + (c - d).norm_sqr())
            .sum::<f64>()
            / self.k.len() as f64)
            .sqrt();
        Ok(Comparison {
            k_cosine: k,
            g_cosine: g,
            joint_cosine: (k.max(0.) * g.max(0.)).sqrt(),
            kg_rms_distance: distance,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{cosine, dynamics};
    use num_complex::Complex64 as C;

    #[test]
    fn cosine_scales_huge_tiny_and_complex_vectors() {
        for magnitude in [1e308, 1e-300] {
            let a = [
                C::new(magnitude, magnitude),
                C::new(-magnitude, magnitude / 2.),
            ];
            assert!((cosine(&a, &a).unwrap() - 1.).abs() < 1e-14);
            assert!((cosine(&a, &a.map(|value| -value)).unwrap() + 1.).abs() < 1e-14);
            let rotated = a.map(|value| C::new(-value.im, value.re));
            assert!(cosine(&a, &rotated).unwrap().abs() < 1e-14);
        }
    }

    #[test]
    fn cosine_zero_empty_and_invalid_inputs_keep_the_public_contract() {
        assert_eq!(cosine(&[], &[]).unwrap(), 0.);
        assert_eq!(cosine(&[C::ZERO], &[C::new(1., 0.)]).unwrap(), 0.);
        assert!(cosine(&[C::new(1., 0.)], &[]).is_err());
        assert!(cosine(&[C::new(f64::NAN, 0.)], &[C::new(1., 0.)]).is_err());
    }

    #[test]
    fn stable_quadratic_roots_keep_plus_minus_order_when_separated() {
        let result = dynamics(C::new(1e12, 0.), C::new(1., 0.), 1e-12).unwrap();
        assert!((result.roots[0].re - 1e12).abs() / 1e12 < 1e-14);
        assert!((result.roots[1].re - 1e-12).abs() / 1e-12 < 1e-14);
        assert!((result.roots[0] * result.roots[1] - C::new(1., 0.)).norm() < 1e-14);
    }

    #[test]
    fn stable_roots_preserve_large_coefficients_and_small_product_partner() {
        let result = dynamics(C::new(1e308, 0.), C::new(1e308, 0.), 1e-12).unwrap();
        assert!((result.roots[0].re - 1e308).abs() / 1e308 < 1e-14);
        assert!((result.roots[1].re - 1.).abs() < 1e-14);
        assert!((result.roots[0] * result.roots[1] - C::new(1e308, 0.)).norm() / 1e308 < 1e-14);
    }

    #[test]
    fn ordinary_classification_repeated_roots_and_frequency_are_preserved() {
        let decaying = dynamics(C::new(1.25, 0.), C::new(0.375, 0.), 1e-6).unwrap();
        assert_eq!(decaying.classification, "decaying");
        assert!((decaying.roots[0].re - 0.75).abs() < 1e-14);
        assert!((decaying.roots[1].re - 0.5).abs() < 1e-14);

        let repeated = dynamics(C::new(2., 0.), C::new(1., 0.), 1e-6).unwrap();
        assert!(repeated.repeated_root);
        assert_eq!(repeated.classification, "unit-radius");

        let oscillator = dynamics(C::new(0., 0.), C::new(0.25, 0.), 1e-6).unwrap();
        assert!((oscillator.oscillation_period_samples.unwrap() - 4.).abs() < 1e-14);
    }

    #[test]
    fn frequency_test_avoids_squared_coefficient_overflow() {
        let result = dynamics(C::new(1e200, 0.), C::new(1e308, 0.), 1e-6).unwrap();
        assert!(result.oscillation_period_samples.is_none());
        let complex_pair = dynamics(C::new(0., 0.), C::new(1e308, 0.), 1e-6).unwrap();
        assert!((complex_pair.oscillation_period_samples.unwrap() - 4.).abs() < 1e-14);
    }

    #[test]
    fn unrepresentable_root_magnitude_and_invalid_inputs_return_errors() {
        assert!(dynamics(C::new(1.7e308, 1.7e308), C::ZERO, 1e-6).is_err());
        assert!(dynamics(C::new(f64::NAN, 0.), C::ZERO, 1e-6).is_err());
        assert!(dynamics(C::ZERO, C::ZERO, f64::INFINITY).is_err());
    }
}
