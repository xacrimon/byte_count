#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(slice_as_chunks)]
#![feature(stmt_expr_attributes)]
#![feature(portable_simd)]
#![feature(array_chunks)]

use std::array;
use std::convert::TryInto;
use std::simd::{prelude::*, ToBytes};

type Q = i16;

#[no_mangle]
pub fn count_bytes(s: &[Q]) -> usize {
    naive(s)
}

#[inline(never)]
pub fn naive(s: &[Q]) -> usize {
    s.iter()
        .copied()
        .filter(
            #[inline(always)]
            |&x| x == b'\n' as i16,
        )
        .count()
}

#[inline(never)]
pub fn interleaved_pipelined(s: &[Q]) -> usize {
    const IPL: usize = 8;
    const LANES: usize = 16;
    const UNPACK_SZ: usize = 16;
    const BATCH_SIZE: usize = 128;

    #[inline(always)]
    fn select_filter_load(s: &[i16; UNPACK_SZ]) -> Simd<u8, LANES> {
        let raw = s.map(i16::to_ne_bytes);
        let bytes = raw.as_flattened();
        let mid = bytes.len() / 2;

        let [h1, h2] = unsafe {
            [
                Simd::<u8, LANES>::from_array(bytes[..mid].try_into().unwrap_unchecked()),
                Simd::<u8, LANES>::from_array(bytes[mid..].try_into().unwrap_unchecked()),
            ]
        };

        simd_swizzle!(
            h1,
            h2,
            [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
        )
    }

    let needle = Simd::splat(b'\n');
    let popcnt_4b_lut = Simd::from_array(array::from_fn::<u8, LANES, _>(|i| i.count_ones() as u8));

    let mut counter = 0;
    for chunk in s.array_chunks::<BATCH_SIZE>() {
        let mut slices = chunk.array_chunks().map(select_filter_load);
        let mut next_byte = || slices.next().unwrap();

        let mut eqf_a = Simd::<u8, LANES>::splat(0);
        eqf_a |= next_byte().simd_eq(needle).to_int().cast::<u8>() & Simd::splat(0b0000_1000);
        eqf_a |= next_byte().simd_eq(needle).to_int().cast::<u8>() & Simd::splat(0b0000_0100);
        eqf_a |= next_byte().simd_eq(needle).to_int().cast::<u8>() & Simd::splat(0b0000_0010);
        eqf_a |= next_byte().simd_eq(needle).to_int().cast::<u8>() & Simd::splat(0b0000_0001);
        counter += popcnt_4b_lut.swizzle_dyn(eqf_a).reduce_sum() as usize;

        let mut eqf_b = Simd::<u8, LANES>::splat(0);
        eqf_b |= next_byte().simd_eq(needle).to_int().cast::<u8>() & Simd::splat(0b0000_1000);
        eqf_b |= next_byte().simd_eq(needle).to_int().cast::<u8>() & Simd::splat(0b0000_0100);
        eqf_b |= next_byte().simd_eq(needle).to_int().cast::<u8>() & Simd::splat(0b0000_0010);
        eqf_b |= next_byte().simd_eq(needle).to_int().cast::<u8>() & Simd::splat(0b0000_0001);
        counter += popcnt_4b_lut.swizzle_dyn(eqf_b).reduce_sum() as usize;
        debug_assert!(slices.next().is_none());
    }

    counter
}
