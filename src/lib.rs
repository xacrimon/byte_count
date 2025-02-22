#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(slice_as_chunks)]
#![feature(stmt_expr_attributes)]
#![feature(portable_simd)]
#![feature(array_chunks)]

use std::array;
use std::convert::TryInto;
use std::simd::{prelude::*, ToBytes};

#[inline(never)]
pub fn naive_1b(haystack: &[u8], needle: u8) -> usize {
    haystack.iter().copied().filter(|&x| x == needle).count()
}

#[inline(never)]
pub fn interleaved_pipelined_1b(haystack: &[u8], needle: u8) -> usize {
    const LANES: usize = 16;
    const BATCH_SIZE: usize = 128;

    let needle = Simd::<u8, LANES>::splat(needle);
    let mut accum = Simd::splat(0);

    for chunk in haystack.array_chunks::<BATCH_SIZE>() {
        let mut slices = chunk.array_chunks().copied().map(Simd::from_array);
        let mut next_byte = || unsafe { slices.next().unwrap_unchecked() };

        fn select_bit<const N: usize>(eq: Mask<i8, 16>) -> Simd<u8, 16> {
            eq.to_int().cast::<u8>() & Simd::splat(1 << N)
        }

        let eqf = select_bit::<0>(next_byte().simd_eq(needle))
            | select_bit::<1>(next_byte().simd_eq(needle))
            | select_bit::<2>(next_byte().simd_eq(needle))
            | select_bit::<3>(next_byte().simd_eq(needle))
            | select_bit::<4>(next_byte().simd_eq(needle))
            | select_bit::<5>(next_byte().simd_eq(needle))
            | select_bit::<6>(next_byte().simd_eq(needle))
            | select_bit::<7>(next_byte().simd_eq(needle));

        let eqf_32 = Simd::<u32, { LANES / 4 }>::from_le_bytes(eqf);
        accum += eqf_32.count_ones();
    }

    accum.reduce_sum() as usize
}

#[cfg(test)]
mod test {
    use std::vec;

    #[test]
    fn verify_count() {
        use rand::{rngs::SmallRng, Rng, RngCore, SeedableRng};
        let mut rng = SmallRng::seed_from_u64(1);
        let mut occurences_naive = vec![];
        let mut occurences_interleaved_pipelined = vec![];

        for _ in 0..64 {
            let needle = rng.random();
            let mut haystack = vec![0; 1024].into_boxed_slice();
            rng.fill_bytes(&mut haystack);

            occurences_naive.push(super::naive_1b(&haystack, needle));
            occurences_interleaved_pipelined
                .push(super::interleaved_pipelined_1b(&haystack, needle));
        }

        assert_eq!(occurences_naive, occurences_interleaved_pipelined);
    }
}
