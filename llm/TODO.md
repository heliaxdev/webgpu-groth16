# Finish src/gpu.rs

The file `src/gpu.rs` still seems incomplete, and it is not performing operations in batches. For maximum GPU efficiency, operations must be batched (in their entirety or in windows/buckets), rather than sent in sequence to the GPU and awaited on.

# Next Steps

The final missing piece of the puzzle is the **Host-Side Bucket Sorting** required by the Luo-Fu-Gong MSM. Because WGSL doesn't support atomic operations on 384-bit memory (which prevents standard concurrent bucket aggregation on the GPU), the Rust host must group the points by their radix slice (the $b_{ij}$ values) into the `buckets_buf` before calling `gpu.execute_msm`.

We must write the Rust algorithm that slices the 255-bit `blstrs::Scalar` variables into $c$-bit radix windows and sorts the corresponding $A_i$ and $B_i$ points into the WebGPU arrays.

# Finally

- Write unit tests.
- Write test vectors against the `bellman` lib to verify the proofs we have created with this library.
    - Must write an adaptor pattern circuit, to have `bellman` circuits interface with this library.
