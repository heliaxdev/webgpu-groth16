// TODO: write some unit tests to check the correctness of
// each module. then, write an integration test that calls
// `prover::create_proof` with a dummy circuit, and uses
// `bellman::groth16::verify_proof` to verify the generated proof.
// this will be the ultimate test of correctness.

pub mod bucket;
pub mod gpu;
pub mod prover;
