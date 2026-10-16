use engram::{decode::decode, encode::encode, wire};

#[test]
fn full_wire_reconstructs_contiguous_blocks_and_fractional_raw_samples() {
    for length in [1, 2, 3, 4, 64] {
        let samples: Vec<f32> = (0..length)
            .flat_map(|t| [t as f32 * 0.01 + 0.13, t as f32 * 0.02 + 0.21])
            .collect();
        let packet = encode(&samples, length, 2, None);
        let bytes = wire::to_wire(&packet);
        assert_eq!(&bytes[..3], b"EN\x04");
        let restored = wire::from_wire(&bytes).expect("own full packet");
        let decoded = decode(&restored);
        assert_eq!(decoded.len(), samples.len());
        let mut start = restored.blocks.first().map_or(0, |b| b.start);
        for block in &restored.blocks {
            assert_eq!(block.start, start);
            start += block.length;
        }
        assert_eq!(start, length);
        let max_error = samples
            .iter()
            .zip(&decoded)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f32::max);
        assert!(max_error < 0.01, "length {length}, max error {max_error}");
    }
}

#[test]
fn historical_and_unknown_full_versions_are_rejected() {
    let packet = encode(&[0.1, 0.2, 0.3, 0.4], 2, 2, None);
    for version in [0, 1, 2, 5, 255] {
        let mut bytes = wire::to_wire(&packet);
        bytes[2] = version;
        assert!(wire::from_wire(&bytes).is_none());
    }
}
