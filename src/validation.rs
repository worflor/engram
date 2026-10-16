pub fn trajectory(
    x: &[f32],
    t: usize,
    dim: usize,
    pairing: Option<&[i32]>,
) -> Result<(), &'static str> {
    if !(2..=4096).contains(&dim)
        || !dim.is_multiple_of(2)
        || t > 16 * 1024 * 1024 / dim
        || x.len() != t * dim
        || x.iter().any(|v| !v.is_finite())
    {
        return Err("invalid finite row-major trajectory");
    }
    if let Some(basis) = pairing {
        let mut sorted = basis.to_vec();
        sorted.sort_unstable();
        if sorted.len() != dim || sorted.iter().enumerate().any(|(i, &v)| v != i as i32) {
            return Err("invalid pairing permutation");
        }
    }
    Ok(())
}
