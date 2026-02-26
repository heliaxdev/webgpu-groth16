use ff::PrimeField;
use std::marker::PhantomData;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Index {
    Input(usize),
    Aux(usize),
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Variable(Index);

impl Variable {
    pub fn new_unchecked(idx: Index) -> Self {
        Variable(idx)
    }
    pub fn get_unchecked(&self) -> Index {
        self.0
    }
}

#[derive(Clone, Debug)]
pub struct LinearCombination<Scalar: PrimeField>(Vec<(Variable, Scalar)>);

impl<Scalar: PrimeField> AsRef<[(Variable, Scalar)]> for LinearCombination<Scalar> {
    fn as_ref(&self) -> &[(Variable, Scalar)] {
        &self.0
    }
}

impl<Scalar: PrimeField> Default for LinearCombination<Scalar> {
    fn default() -> Self {
        LinearCombination(Vec::new())
    }
}

impl<Scalar: PrimeField> LinearCombination<Scalar> {
    pub fn zero() -> Self {
        LinearCombination::default()
    }

    pub fn add(mut self, var: (Variable, Scalar)) -> Self {
        self.0.push(var);
        self
    }
}

pub trait ConstraintSystem<Scalar: PrimeField>: Sized {
    type Root: ConstraintSystem<Scalar>;

    fn alloc<F, A, AR>(&mut self, annotation: A, f: F) -> Result<Variable, String>
    where
        F: FnOnce() -> Result<Scalar, String>,
        A: FnOnce() -> AR,
        AR: Into<String>;

    fn alloc_input<F, A, AR>(&mut self, annotation: A, f: F) -> Result<Variable, String>
    where
        F: FnOnce() -> Result<Scalar, String>,
        A: FnOnce() -> AR,
        AR: Into<String>;

    fn enforce<A, AR, LA, LB, LC>(&mut self, annotation: A, a: LA, b: LB, c: LC)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
        LA: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
        LB: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
        LC: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>;

    fn push_namespace<NR, N>(&mut self, name_fn: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR;

    fn pop_namespace(&mut self);

    fn get_root(&mut self) -> &mut Self::Root;

    fn one() -> Variable {
        Variable::new_unchecked(Index::Input(0))
    }
}

pub trait Circuit<Scalar: PrimeField> {
    fn synthesize<CS: ConstraintSystem<Scalar>>(self, cs: &mut CS) -> Result<(), String>;
}
