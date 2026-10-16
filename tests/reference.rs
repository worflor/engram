use engram::{
    analysis::{Signature, dynamics, fit_ar1},
    decode::{decode, try_decode},
    encode::{encode, try_encode},
    streaming::{StreamDecoder, StreamEncoder},
    wire,
};
use num_complex::Complex64 as C;

#[test]
fn saved_orbits_reconstruct_with_independent_block_state() {
    for dim in [2, 4, 8, 16] {
        for length in [3, 7, 13, 31, 128] {
            let x: Vec<f32> = (0..length)
                .flat_map(|t| (0..dim).map(move |d| (t as f64 * 0.07 + d as f64).sin() as f32))
                .collect();
            let packet = encode(&x, length, dim, None);
            let bytes = wire::to_wire(&packet);
            let restored = wire::from_wire(&bytes).expect("own packet");
            let y = decode(&restored);
            let max_error = x
                .iter()
                .zip(y)
                .map(|(a, b)| (a - b).abs())
                .fold(0., f32::max);
            assert!(max_error < 0.02, "{dim}D/{length}: {max_error}");
            for (a, b) in packet.blocks.iter().zip(&restored.blocks) {
                assert_eq!(a.seed1, b.seed1);
                assert_eq!(a.seed2, b.seed2);
                assert_eq!(a.micro_size, b.micro_size);
            }
        }
    }
}

#[test]
fn bounded_readers_reject_every_prefix_and_hostile_lengths() {
    let p = encode(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 4, 2, None);
    for bytes in [wire::to_wire(&p), wire::to_wire_compact(&p)] {
        for end in 0..bytes.len() {
            assert!(wire::from_wire(&bytes[..end]).is_none());
            assert!(wire::from_wire_compact(&bytes[..end]).is_none());
        }
        let mut trailing = bytes;
        trailing.push(0);
        assert!(wire::from_wire(&trailing).is_none());
        assert!(wire::from_wire_compact(&trailing).is_none());
    }
    let mut bytes = wire::to_wire(&p);
    bytes[6..10].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(wire::from_wire(&bytes).is_none());
    let compact = wire::from_wire_compact(&wire::to_wire_compact(&p)).unwrap();
    assert!(try_decode(&compact).is_err());
    assert!(try_encode(&[f32::NAN; 8], 4, 2, None).is_err());
    assert!(try_encode(&[0.; 8], 4, 2, Some(&[0, 0])).is_err());
}

#[test]
fn deterministic_mutations_cannot_panic_packet_readers() {
    let x: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
    let bytes = wire::to_wire(&encode(&x, 32, 2, None));
    let mut seed = 0x5eeda11_u64;
    for _ in 0..600 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut damaged = bytes.clone();
        let index = (seed as usize) % damaged.len();
        damaged[index] ^= (seed >> 32) as u8;
        let _ = wire::from_wire(&damaged);
        let _ = wire::from_wire_compact(&damaged);
    }
}

#[test]
fn source_kernel_receipts_match_native_signatures() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/source-kernels.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let dim = case["dim"].as_u64().unwrap() as usize;
        let t = case["length"].as_u64().unwrap() as usize;
        let values: Vec<f32> = case["samples"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        let basis: Vec<i32> = (0..dim as i32).collect();
        let signature = Signature::from_trajectory(&values, t, dim, Some(&basis)).unwrap();
        for (language, result) in case["implementations"].as_object().unwrap() {
            let tolerance = if language == "java" { 0.002 } else { 1e-8 };
            for (field, values) in [("k", &signature.k), ("g", &signature.g)] {
                for (j, z) in values.iter().enumerate() {
                    let expected = C::new(
                        result[field][j][0].as_f64().unwrap(),
                        result[field][j][1].as_f64().unwrap(),
                    );
                    assert!(
                        (z - expected).norm() < tolerance,
                        "{language} {} {field}: {:?} != {:?}",
                        case["name"],
                        z,
                        expected
                    );
                }
            }
        }
    }
}

#[test]
fn diagnostics_and_streams_follow_the_reference_contract() {
    let linear = dynamics(C::new(2., 0.), C::new(1., 0.), 1e-6).unwrap();
    assert!(linear.repeated_root);
    assert_eq!(linear.classification, "unit-radius");
    assert!(linear.half_life_samples.is_none());
    let (k, _) = fit_ar1(&[C::new(1., 0.), C::new(2., 0.)]).unwrap();
    assert!((k.re - 2. / 1.0001).abs() < 1e-12);
    for length in [1, 2, 3, 17, 97] {
        let x: Vec<f32> = (0..length)
            .flat_map(|t| (0..4).map(move |d| (t as f32 * 0.15 + d as f32).sin()))
            .collect();
        let mut encoder = StreamEncoder::new(4);
        let mut decoder = StreamDecoder::new(4);
        for row in x.chunks_exact(4) {
            if let Some(b) = encoder.push(row) {
                decoder.push(&b).unwrap();
            }
        }
        if let Some(b) = encoder.flush() {
            decoder.push(&b).unwrap();
        }
        assert_eq!(decoder.trajectory().len(), x.len());
        assert!(
            x.iter()
                .zip(decoder.trajectory())
                .all(|(a, b)| (a - b).abs() < 0.02)
        );
    }
}

#[test]
fn error_budget_and_legacy_reading_are_explicit() {
    let full = wire::from_wire(include_bytes!("fixtures/legacy-full-v3.en")).unwrap();
    assert_eq!(full.length, 64);
    let compact = wire::from_wire_compact(include_bytes!("fixtures/legacy-compact-v2.en")).unwrap();
    assert!(!compact.reconstructable);
    let x: Vec<f32> = (0..128)
        .flat_map(|i| {
            (0..16).map(move |d| (i as f32 * if i < 64 { 0.06 } else { 0.23 } + d as f32).sin())
        })
        .collect();
    let p = engram::encode::encode_with_tolerance(&x, 128, 16, None, 0.).unwrap();
    assert_eq!(decode(&wire::from_wire(&wire::to_wire(&p)).unwrap()), x);
}
