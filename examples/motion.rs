//! Host-defined motion signatures; no calibrated recognition threshold.
use engram::{fit::fit_pair, predict::to_complex};
use num_complex::Complex64;

fn signature(points: &[[f32; 2]]) -> (Complex64, Complex64) {
    let mean = points
        .iter()
        .fold([0.0; 2], |sum, p| [sum[0] + p[0], sum[1] + p[1]])
        .map(|v| v / points.len() as f32);
    let scale = (points
        .iter()
        .map(|p| (p[0] - mean[0]).powi(2) + (p[1] - mean[1]).powi(2))
        .sum::<f32>()
        / points.len() as f32)
        .sqrt()
        .max(f32::EPSILON);
    let centered: Vec<f32> = points
        .iter()
        .flat_map(|p| [(p[0] - mean[0]) / scale, (p[1] - mean[1]) / scale])
        .collect();
    // Two coordinates form one explicit complex plane; no derived re-pairing.
    let fit = fit_pair(&to_complex(&centered, points.len(), 2));
    (fit.k, fit.g)
}

fn main() {
    let loop_path: Vec<[f32; 2]> = (0..96)
        .map(|i| {
            let t = i as f32 * 0.08;
            [t.cos(), t.sin()]
        })
        .collect();
    let ramp: Vec<[f32; 2]> = (0..96).map(|i| [i as f32, i as f32 * 0.5]).collect();
    let query: Vec<[f32; 2]> = loop_path
        .iter()
        .map(|p| [p[0] * 3.0 + 7.0, p[1] * 3.0 - 2.0])
        .collect();
    let (qk, qg) = signature(&query);
    for (label, reference) in [("loop", &loop_path), ("ramp", &ramp)] {
        let (k, g) = signature(reference);
        let distance = ((qk - k).norm_sqr() + (qg - g).norm_sqr()).sqrt();
        println!("{label}: K/G distance {distance:.6}");
    }
    println!("Same cadence and length; centering/scale normalization are host choices.");
}
