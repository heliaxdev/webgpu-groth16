//! GPU-side constraint system for Groth16 witness generation.
//!
//! Implements the `bellman::ConstraintSystem` trait to collect witness values
//! and linear combinations during circuit synthesis. The constraint system
//! tracks density masks for A/B assignments to enable sparse MSM dispatch.

use ff::{Field, PrimeField};

use crate::bellman;
use crate::gpu::curve::GpuCurve;

/// Constraint system that collects witness values and linear combinations.
///
/// Starts with a single input (the implicit constant ONE required by Groth16).
/// During synthesis, tracks which aux/input variables appear in A and B
/// linear combinations via density masks, allowing the prover to skip
/// zero-scalar bases in MSM dispatch.
pub(crate) struct GpuConstraintSystem<G: GpuCurve> {
    pub inputs: Vec<G::Scalar>,
    pub aux: Vec<G::Scalar>,
    pub a_aux_density: Vec<bool>,
    pub b_input_density: Vec<bool>,
    pub b_aux_density: Vec<bool>,
    pub a_lcs: Vec<Vec<(bellman::Variable, G::Scalar)>>,
    pub b_lcs: Vec<Vec<(bellman::Variable, G::Scalar)>>,
    pub c_lcs: Vec<Vec<(bellman::Variable, G::Scalar)>>,
    pub _marker: std::marker::PhantomData<G>,
}

impl<G: GpuCurve> Default for GpuConstraintSystem<G> {
    fn default() -> Self {
        Self::new()
    }
}

impl<G: GpuCurve> GpuConstraintSystem<G> {
    pub fn new() -> Self {
        GpuConstraintSystem {
            inputs: vec![G::Scalar::ONE],
            aux: Vec::new(),
            a_aux_density: Vec::new(),
            b_input_density: vec![false],
            b_aux_density: Vec::new(),
            a_lcs: Vec::new(),
            b_lcs: Vec::new(),
            c_lcs: Vec::new(),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<G: GpuCurve + Send> bellman::ConstraintSystem<G::Scalar> for GpuConstraintSystem<G> {
    type Root = Self;
    fn alloc<F, A, AR>(
        &mut self,
        _annotation: A,
        f: F,
    ) -> Result<bellman::Variable, bellman::SynthesisError>
    where
        F: FnOnce() -> Result<G::Scalar, bellman::SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let value = f()?;
        self.aux.push(value);
        self.a_aux_density.push(false);
        self.b_aux_density.push(false);
        Ok(bellman::Variable::new_unchecked(bellman::Index::Aux(
            self.aux.len() - 1,
        )))
    }
    fn alloc_input<F, A, AR>(
        &mut self,
        _annotation: A,
        f: F,
    ) -> Result<bellman::Variable, bellman::SynthesisError>
    where
        F: FnOnce() -> Result<G::Scalar, bellman::SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let value = f()?;
        self.inputs.push(value);
        self.b_input_density.push(false);
        Ok(bellman::Variable::new_unchecked(bellman::Index::Input(
            self.inputs.len() - 1,
        )))
    }
    fn enforce<A, AR, LA, LB, LC>(&mut self, _annotation: A, a: LA, b: LB, c: LC)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
        LA: FnOnce(bellman::LinearCombination<G::Scalar>) -> bellman::LinearCombination<G::Scalar>,
        LB: FnOnce(bellman::LinearCombination<G::Scalar>) -> bellman::LinearCombination<G::Scalar>,
        LC: FnOnce(bellman::LinearCombination<G::Scalar>) -> bellman::LinearCombination<G::Scalar>,
    {
        let a_lc = a(bellman::LinearCombination::zero());
        let b_lc = b(bellman::LinearCombination::zero());
        let c_lc = c(bellman::LinearCombination::zero());

        let a_vec = lc_to_vec(a_lc);
        for (var, coeff) in &a_vec {
            if *coeff == G::Scalar::ZERO {
                continue;
            }
            if let bellman::Index::Aux(i) = var.get_unchecked() {
                self.a_aux_density[i] = true;
            }
        }

        let b_vec = lc_to_vec(b_lc);
        for (var, coeff) in &b_vec {
            if *coeff == G::Scalar::ZERO {
                continue;
            }
            match var.get_unchecked() {
                bellman::Index::Input(i) => self.b_input_density[i] = true,
                bellman::Index::Aux(i) => self.b_aux_density[i] = true,
            }
        }

        self.a_lcs.push(a_vec);
        self.b_lcs.push(b_vec);
        self.c_lcs.push(lc_to_vec(c_lc));
    }
    fn push_namespace<NR, N>(&mut self, _name_fn: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
    }
    fn pop_namespace(&mut self) {}
    fn get_root(&mut self) -> &mut Self::Root {
        self
    }
}

pub(crate) fn lc_to_vec<S: PrimeField>(
    lc: bellman::LinearCombination<S>,
) -> Vec<(bellman::Variable, S)> {
    #[cfg(feature = "bellman-provider-bellman")]
    let lc_iter = lc.as_ref().iter();

    #[cfg(feature = "bellman-provider-nam-bellperson")]
    let lc_iter = lc.iter();

    lc_iter.map(|(var, coeff)| (var, *coeff)).collect()
}
