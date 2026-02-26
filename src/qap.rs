use crate::traits::{ConstraintSystem, Index, LinearCombination, Variable};
use ff::PrimeField;

/// Represents a single R1CS constraint: A * B = C
#[derive(Clone, Debug)]
pub struct Constraint<Scalar: PrimeField> {
    pub a: LinearCombination<Scalar>,
    pub b: LinearCombination<Scalar>,
    pub c: LinearCombination<Scalar>,
}

/// The Constraint System used to trace QAP matrices U, V, and W.
pub struct ProvingCS<Scalar: PrimeField> {
    pub constraints: Vec<Constraint<Scalar>>,
    pub inputs: Vec<Scalar>,
    pub aux: Vec<Scalar>,

    // Tracks the current namespace depth
    namespace_stack: Vec<String>,
}

impl<Scalar: PrimeField> ProvingCS<Scalar> {
    pub fn new() -> Self {
        let mut cs = ProvingCS {
            constraints: Vec::new(),
            inputs: Vec::new(),
            aux: Vec::new(),
            namespace_stack: Vec::new(),
        };
        // Allocate the constant 1 (a_0) in the inputs
        cs.inputs.push(Scalar::ONE);
        cs
    }
}

impl<Scalar: PrimeField> ConstraintSystem<Scalar> for ProvingCS<Scalar> {
    type Root = Self;

    fn alloc<F, A, AR>(&mut self, _annotation: A, f: F) -> anyhow::Result<Variable>
    where
        F: FnOnce() -> anyhow::Result<Scalar>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let value = f()?;
        self.aux.push(value);
        Ok(Variable::new_unchecked(Index::Aux(self.aux.len() - 1)))
    }

    fn alloc_input<F, A, AR>(&mut self, _annotation: A, f: F) -> anyhow::Result<Variable>
    where
        F: FnOnce() -> anyhow::Result<Scalar>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let value = f()?;
        self.inputs.push(value);
        Ok(Variable::new_unchecked(Index::Input(self.inputs.len() - 1)))
    }

    fn enforce<A, AR, LA, LB, LC>(&mut self, _annotation: A, a: LA, b: LB, c: LC)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
        LA: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
        LB: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
        LC: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
    {
        let a_lc = a(LinearCombination::zero());
        let b_lc = b(LinearCombination::zero());
        let c_lc = c(LinearCombination::zero());

        self.constraints.push(Constraint {
            a: a_lc,
            b: b_lc,
            c: c_lc,
        });
    }

    fn push_namespace<NR, N>(&mut self, name_fn: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        self.namespace_stack.push(name_fn().into());
    }

    fn pop_namespace(&mut self) {
        self.namespace_stack.pop();
    }

    fn get_root(&mut self) -> &mut Self::Root {
        self
    }
}
