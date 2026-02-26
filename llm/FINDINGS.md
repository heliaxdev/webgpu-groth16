# Project Findings & Architectural Decisions

## Overview
We are building a production-grade, WebGPU-accelerated Groth16 zero-knowledge prover targeted for the browser via Rust and WASM. The prover targets the BLS12-381 elliptic curve and interfaces with the standard `bellman` synthesis API. 

## Cryptographic & Mathematical Constraints
Generating a Groth16 proof requires computing a quotient polynomial $h(x)$ and evaluating three proof elements: $A, B \in \mathbb{G}_1$ and $C \in \mathbb{G}_2$. Because WebGPU lacks native 64-bit integer support and dynamic memory allocation, all cryptographic primitives must be mapped to fixed-size arrays of 32-bit unsigned integers (`u32`).

* **Scalar Field ($\mathbb{F}_r$):** 255-bit prime, requiring an 8-limb `u32` array (256 bits).
* **Base Field ($\mathbb{F}_q$):** 381-bit prime, requiring a 12-limb `u32` array (384 bits).

## Algorithms Selected for WGSL Implementation

1.  **Finite Field Arithmetic (Montgomery Reduction):** Division modulo a prime is incredibly slow on a GPU. We use **Coarsely Integrated Operand Scanning (CIOS)** for Montgomery multiplication. The CIOS algorithm integrates the multiplication and reduction steps into a single loop, minimizing register pressure and eliminating the need for a double-width accumulator.
2.  **Elliptic Curve Arithmetic (Jacobian Coordinates):**
    To avoid expensive modular inversions during point addition and doubling, we map all coordinates into the Jacobian projective space $(X, Y, Z)$ where $x = X/Z^2$ and $y = Y/Z^3$.
3.  **Multi-Scalar Multiplication (MSM):**
    Standard Pippenger's bucket method is inefficient for zkSNARKs due to the large, inconsecutive gaps between scalar values. We use the optimized bucket set construction and **Subsum Accumulation Algorithm II** by Luo, Fu, and Gong. This allows accumulating $m$ intermediate subsums using at most $2m+d-3$ additions, where the maximum bucket index gap $d=6$ for BLS12-381.
4.  **Hardware-Level Workarounds:**
    Because GPUs do not support atomic operations on 384-bit structs, the host (Rust) must perform the scalar q-ary conversions and sort the points into bucket indices before transferring the memory to the GPU.
5. **Hierarchical 4-Step NTT:**
   Instead of a standard Radix-2 NTT, we will utilize Bailey's 4-step algorithm (Generic Algorithm). By treating an $N$-element NTT as a 2D matrix of $N_1 \times N_2$, we can execute $N_1$ smaller NTTs that fit perfectly into WebGPU `workgroup` shared memory, perform a global quadrant-swap transpose, and execute the final $N_2$ NTTs.
6. **MVM/MSM Tiling:**
   To overcome GPU memory bandwidth bottlenecks during bucket accumulation, we will implement tiling. We will load blocks of $k$ points into `workgroup` memory to maximize data reuse and spatial locality before writing the accumulated sums back to global VRAM.

## Rust Project Structure
```text
.
└─── src
    ├── lib.rs          # Module definitions
    ├── traits.rs       # Bellman-compatible Circuit/ConstraintSystem traits
    ├── qap.rs          # Host-side QAP reduction
    ├── gpu.rs          # WebGPU initialization & pipeline orchestration (Pending)
    ├── prover.rs       # Groth16 proof assembly (Pending)
    └── shader
        └── bls12_381
            ├── fp.wgsl     # 384-bit/256-bit BigInt & CIOS Montgomery
            ├── curve.wgsl  # G1 and G2 Jacobian addition/doubling
            ├── msm.wgsl    # Luo-Fu-Gong Subsum Accumulation
            └── ntt.wgsl    # Cooley-Tukey FFT
```
