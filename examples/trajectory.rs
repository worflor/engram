use engram::{brain::Brain, brain_io, decode::decode, encode::encode, wire};

fn main() {
    let dim = 4;
    let samples: Vec<f32> = (0..128)
        .flat_map(|t| (0..dim).map(move |d| (t as f32 * 0.07 + d as f32).sin()))
        .collect();
    let packet = encode(&samples, 128, dim, None);
    let bytes = wire::to_wire(&packet);
    let restored = wire::from_wire(&bytes).expect("self-produced full packet");
    let reconstructed = decode(&restored);
    let rms = (samples
        .iter()
        .zip(&reconstructed)
        .map(|(a, b)| f64::from(a - b).powi(2))
        .sum::<f64>()
        / samples.len() as f64)
        .sqrt();

    let mut brain = Brain::new(dim, 0.005);
    brain.absorb(&packet, Some("oscillation"));
    let snapshot = brain_io::save(&brain);
    let loaded = brain_io::load(&snapshot).expect("self-produced brain");
    println!(
        "{} samples, {} blocks, {} wire bytes, RMS error {:.6}, {} wells",
        packet.length,
        packet.blocks.len(),
        bytes.len(),
        rms,
        loaded.wells.len()
    );
}
