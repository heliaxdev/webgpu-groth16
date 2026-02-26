# Finish src/gpu.rs

The file `src/gpu.rs` still seems incomplete, and it is not performing operations in batches. For maximum GPU efficiency, operations must be batched (in their entirety or in windows/buckets), rather than sent in sequence to the GPU and awaited on.

# Next Steps

- ~~Host-Side Bucket Sorting~~ - DONE: Implemented in `src/bucket.rs`

# Finally

- Write unit tests.
- Write test vectors against the `bellman` lib to verify the proofs we have created with this library.
    - Must write an adaptor pattern circuit, to have `bellman` circuits interface with this library.
