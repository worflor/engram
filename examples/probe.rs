//! Machine-readable bridge for cross-language and instrument checks.
use engram::{decode::decode, encode::encode, wire};
use serde::Deserialize;
use serde_json::json;
use std::io::{self, Read};

#[derive(Deserialize)]
struct Request {
    dim: usize,
    #[serde(default)]
    samples: Vec<f32>,
    pairing: Option<Vec<i32>>,
    wire: Option<Vec<u8>>,
    #[serde(default)]
    compact: bool,
    #[serde(default)]
    analyze: bool,
    #[serde(default)]
    memory: bool,
}

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).expect("stdin");
    let request: Request = serde_json::from_str(&input).expect("JSON request");
    assert!(request.dim >= 2 && request.dim.is_multiple_of(2));
    if request.analyze {
        let s = engram::analysis::Signature::from_trajectory(
            &request.samples,
            request.samples.len() / request.dim,
            request.dim,
            request.pairing.as_deref(),
        )
        .expect("signature input");
        println!(
            "{}",
            json!({"k":s.k.iter().map(|z|[z.re,z.im]).collect::<Vec<_>>(),
            "g":s.g.iter().map(|z|[z.re,z.im]).collect::<Vec<_>>(),"rms":s.pair_rms,"order":s.order})
        );
        return;
    }
    if request.memory {
        let mut brain = engram::brain::Brain::new(request.dim, 0.005);
        brain.name = "native-fixture".into();
        let packet = encode(
            &request.samples,
            request.samples.len() / request.dim,
            request.dim,
            request.pairing.as_deref(),
        );
        brain.absorb(&packet, Some("fixture"));
        println!(
            "{}",
            json!({"bytes":engram::brain_io::save(&brain),"wells":brain.wells.len(),"absorbed":brain.total_absorbed})
        );
        return;
    }
    let packet = if let Some(bytes) = request.wire {
        if request.compact {
            let packet = wire::from_wire_compact(&bytes).expect("valid compact test packet");
            println!(
                "{}",
                json!({"dim":packet.dim,"length":packet.length,
                "blocks":packet.blocks.iter().map(|b| json!({"mode": b.mode as u8,
                    "macro_k":b.macro_k.iter().map(|z|[z.re,z.im]).collect::<Vec<_>>(),
                    "pair_rms": b.pair_rms})).collect::<Vec<_>>() })
            );
            return;
        }
        wire::from_wire(&bytes).expect("valid full test packet")
    } else {
        assert_eq!(request.samples.len() % request.dim, 0);
        encode(
            &request.samples,
            request.samples.len() / request.dim,
            request.dim,
            request.pairing.as_deref(),
        )
    };
    let blocks: Vec<_> = packet
        .blocks
        .iter()
        .map(|b| {
            json!({
                "start": b.start, "length": b.length, "mode": format!("{:?}", b.mode),
                "macro_k": b.macro_k.iter().map(|z| [z.re, z.im]).collect::<Vec<_>>(),
                "macro_g": b.macro_g.iter().map(|z| [z.re, z.im]).collect::<Vec<_>>()
            })
        })
        .collect();
    let full = wire::to_wire(&packet);
    let wire_decoded = decode(&wire::from_wire(&full).expect("own wire packet"));
    println!(
        "{}",
        json!({
            "dim": packet.dim, "length": packet.length, "pairing": packet.pairing,
            "capture": packet.capture(), "decoded": decode(&packet), "blocks": blocks,
            "full": full, "wire_decoded": wire_decoded, "compact": wire::to_wire_compact(&packet)
        })
    );
}
