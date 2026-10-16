//! The Whisper Engram Universal Trajectory Codec
//!
//! Any signal that moves through a high-dimensional space over time is a
//! trajectory. A thought in embedding space, a gesture in the complex plane,
//! a melody in spectral coordinates, a brain wave across electrode channels.
//! The codec finds orbits in all of them.
//!
//! The predictor is the damped harmonic oscillator:
//!
//! ```text
//! z[n] = K * z[n-1] - G * z[n-2]
//! ```
//!
//! Generalized from one complex pair to D/2 independent pairs and cascaded
//! into two hierarchical levels. One algorithm (fit_pair) at every level.
//! No neural network, no training loop.

pub mod types;
pub mod fit;
pub mod predict;
pub mod segment;
pub mod encode;
pub mod decode;
pub mod wire;
pub mod brain;
pub mod brain_io;
pub mod streaming;
pub mod histogram;
