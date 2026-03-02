use std::iter::{Extend, FromIterator};
use std::mem;

use ff::PrimeField;

const LEN_USIZE: u32 = mem::size_of::<usize>() as u32;

// NOTE: this only compiles on 64-bit hosts, but it's
// not like there are mainstream 32-bit build machines :-)
const LOG2_LEN_USIZE: u32 = LEN_USIZE.ilog2();

#[derive(Debug, Clone)]
pub struct Mask {
    len: u32,
    set: Vec<usize>,
}

impl Mask {
    pub const fn new() -> Self {
        Self {
            len: 0u32,
            set: Vec::new(),
        }
    }

    pub const fn len(&self) -> usize {
        self.len as usize
    }

    pub fn push(&mut self, val: bool) {
        let global_insert_pos = self.len;

        self.len += 1;

        if fast_rem::is_divisible(global_insert_pos, LEN_USIZE) {
            self.make_room_for_new_value_slow();
        }

        if val {
            let usize_insert_pos =
                fast_rem::fast_mod(global_insert_pos, LEN_USIZE);

            self.set_last_index_slow(usize_insert_pos);
        }
    }

    #[cold]
    fn make_room_for_new_value_slow(&mut self) {
        self.set.push(0);
    }

    #[cold]
    fn set_last_index_slow(&mut self, insert_pos: u32) {
        *self.set.last_mut().unwrap() |= 1 << insert_pos;
    }

    pub fn is_set(&self, global_index: usize) -> bool {
        let vec_index = global_index >> LOG2_LEN_USIZE;
        let usize_index = fast_rem::fast_mod(global_index as u32, LEN_USIZE);

        self.set[vec_index] & (1 << usize_index) != 0
    }

    pub fn set(&mut self, global_index: usize, to: bool) {
        let vec_index = global_index >> LOG2_LEN_USIZE;
        let usize_index = fast_rem::fast_mod(global_index as u32, LEN_USIZE);

        self.set[vec_index] |= (to as usize) << usize_index;
    }
}

impl FromIterator<bool> for Mask {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = bool>,
    {
        let mut mask = Mask::new();
        mask.extend(iter);
        mask
    }
}

impl Extend<bool> for Mask {
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = bool>,
    {
        for value in iter {
            self.push(value);
        }
    }
}

pub fn dense_assignment_from_masks<S: PrimeField>(
    inputs: &[S],
    aux: &[S],
    input_mask: &Mask,
    aux_mask: &Mask,
) -> Vec<S> {
    let mut out = Vec::new();
    for (i, s) in inputs.iter().enumerate() {
        if i < input_mask.len() && input_mask.is_set(i) {
            out.push(*s);
        }
    }
    for (i, s) in aux.iter().enumerate() {
        if i < aux_mask.len() && aux_mask.is_set(i) {
            out.push(*s);
        }
    }
    out
}

mod fast_rem {
    //! This module implements fast remainder calculations based on the paper
    //! "Faster Remainder by Direct Computation", by Daniel Lemire, Owen Kaser,
    //! and Nathan Kurz.
    //!
    //! It computes the remainder directly by using the fractional portion of
    //! the product of the numerator and the inverse of the divisor.

    /// Precompute the approximate reciprocal `c` for a given divisor `d`.
    #[inline]
    pub const fn compute_c(d: u32) -> u64 {
        // Equivalent to UINT64_C(0xFFFFFFFFFFFFFFFF) / d + 1
        (u64::MAX / (d as u64)) + 1
    }

    /// Compute (n % d) using the precomputed constant `c`.
    ///
    /// This in hot loops where `c` has already been calculated.
    #[inline]
    pub const fn fast_mod_with_c(n: u32, d: u32, c: u64) -> u32 {
        let lowbits: u64 = c.wrapping_mul(n as u64);

        let result: u128 = (lowbits as u128) * (d as u128);
        (result >> 64) as u32
    }

    /// Compute (n % d) by calculating the reciprocal on the fly.
    ///
    /// ## Warning
    ///
    /// Only use this if `d` is a compile-time constant! If `d` is a runtime
    /// variable, this will perform a slow hardware division, negating the
    /// performance benefits of this algorithm.
    #[inline]
    pub const fn fast_mod(n: u32, d: u32) -> u32 {
        let c = compute_c(d);
        fast_mod_with_c(n, d, c)
    }

    /// Checks if `n` is divisible by `d` (i.e., `n % d == 0`) using the
    /// precomputed constant `c`.
    ///
    /// Use this in hot loops where `c` has already been calculated.
    #[inline]
    pub const fn is_divisible_with_c(n: u32, c: u64) -> bool {
        let lowbits: u64 = c.wrapping_mul(n as u64);
        // Using wrapping_sub(1) correctly handles the c - 1 check.
        lowbits <= c.wrapping_sub(1)
    }

    /// Checks if `n` is divisible by `d` (i.e., `n % d == 0`) by calculating
    /// the reciprocal on the fly.
    ///
    /// ## Warning
    ///
    /// Only use this if `d` is a compile-time constant! If `d` is a runtime
    /// variable, this will perform a slow hardware division.
    #[inline]
    pub const fn is_divisible(n: u32, d: u32) -> bool {
        let c = compute_c(d);
        is_divisible_with_c(n, c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_density_mask() {
        let mut mask = Mask::new();

        let decomposed_bits: u32 = 0b11111011010101010;

        for i in 0..17 {
            let value = ((1 << i) & decomposed_bits) != 0;
            mask.push(value);
            println!("value at {i} is {value}");
        }

        assert_eq!(mask.len, 17);
        assert_eq!(mask.set.len(), 3);

        for i in 0..17 {
            let expected_value = ((1 << i) & decomposed_bits) != 0;

            assert_eq!(
                expected_value,
                mask.is_set(i),
                "failed at index {i}, current set has {:?}",
                mask.set
            );
        }
    }
}
