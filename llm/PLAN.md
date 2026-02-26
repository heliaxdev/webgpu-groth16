# Project Plan: WebGPU-Accelerated Groth16 Prover (Rust + WASM)

## Overview

This project aims to implement a production-grade, in-browser Groth16 zero-knowledge prover using Rust, WebAssembly (WASM), and WebGPU (`wgpu`). By offloading the computationally heavy Fast Fourier Transforms (FFTs) and Multi-Scalar Multiplications (MSMs) to the client's GPU, we can achieve significant performance gains for generating proofs directly in the browser.

The prover will target the BLS12-381 curve and support the standard `bellman` circuit synthesis API. Generating a Groth16 proof requires computing a quotient polynomial $h(x)$ and evaluating three proof elements: $A$, $B$, and $C$.

## Phase 1: Project Setup & API Skeleton

**Goal:** Establish the Rust project structure, WASM compilation target, and `bellman`-compatible traits.

* Initialize a Rust library crate with `cdylib` crate type for WASM binding.
* Configure `wasm-bindgen` and `wgpu` dependencies with the `webgl` and `webgpu` backends enabled.
* Implement the `Circuit` and `ConstraintSystem` traits exactly as specified by the `bellman` API.
* Create a QAP (Quadratic Arithmetic Program) reduction module to transform the synthesized `ConstraintSystem` into polynomial matrices $U$, $V$, and $W$.

## Phase 2: WGSL Mathematical Foundations

**Goal:** Implement the low-level mathematical primitives required for BLS12-381 inside WebGPU shaders (WGSL).

* **381-bit Big Integer Arithmetic:** Implement custom WGSL structs and functions to handle 381-bit integer addition, subtraction, and multiplication using arrays of 32-bit unsigned integers (`u32`).
* **Prime Field Arithmetic:** Implement Montgomery reduction and modular arithmetic for the BLS12-381 scalar field $\mathbb{Z}_p$.
* 
**Elliptic Curve Arithmetic:** Implement point addition and point doubling in Jacobian coordinates for both $\mathbb{G}_1$ and $\mathbb{G}_2$.



## Phase 3: GPU Cryptographic Primitives

**Goal:** Build the heavy-lifting cryptographic algorithms in WGSL compute shaders.

* 
**Number Theoretic Transform (NTT):** Implement a Cooley-Tukey NTT shader to compute the coefficients of polynomials in $O(n \log n)$ time. This is required to compute the quotient polynomial $h(x)$.


* **Multi-Scalar Multiplication (MSM):** Implement Pippenger's algorithm in WGSL. This requires building bucket accumulation shaders and parallel prefix sums to handle the $m+3n-l+3$ exponentiations in $\mathbb{G}_1$ and $n+1$ exponentiations in $\mathbb{G}_2$.



## Phase 4: Host-Device Orchestration

**Goal:** Manage the memory transfer and execution pipeline between the Rust (WASM) host and the WebGPU device.

* Initialize the `wgpu::Instance`, `wgpu::Adapter`, and `wgpu::Device` from the browser environment.
* Implement buffer management to serialize the QAP constraints ($U$, $V$, $W$) and the witness assignments ($a_i$) into `wgpu::Buffer` instances.
* Create WebGPU compute pipelines and bind groups for the NTT and MSM shaders.
* Implement the command encoder logic to dispatch workgroups based on the QAP degree ($n$) and the number of variables ($m$).

## Phase 5: Groth16 Proof Assembly

**Goal:** Tie the GPU computations together to output the final, valid Groth16 proof.

* Dispatch the NTT pipeline to evaluate $\sum_{i=0}^m a_i u_i(X)$, $\sum_{i=0}^m a_i v_i(X)$, and $\sum_{i=0}^m a_i w_i(X)$, and compute the quotient polynomial $h(X)$.


* Sample random scalars $r, s \in \mathbb{Z}_p$ for zero-knowledge randomization.


* Dispatch the MSM pipeline to compute $A \in \mathbb{G}_1$.


* Dispatch the MSM pipeline to compute $B \in \mathbb{G}_2$.


* Dispatch the MSM pipeline to compute $C \in \mathbb{G}_1$.


* Read the final $A$, $B$, and $C$ coordinates back from the GPU buffers into Rust.
* Format the resulting proof into a structured Rust object that can be serialized and sent to a verifier.
