#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(slice_as_chunks)]
#![feature(stmt_expr_attributes)]
#![feature(portable_simd)]
#![feature(array_chunks)]

use std::array;
use std::convert::TryInto;
use std::simd::prelude::*;

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
    let lut_lookup = |bitfield| {
        (
            popcnt_4b_lut.swizzle_dyn(bitfield & Simd::splat(0x0F)),
            popcnt_4b_lut.swizzle_dyn(bitfield >> Simd::splat(4)),
        )
    };
    let accum_mask = |offset: u32| Simd::splat(1 << offset);

    let mut counter = 0;
    for chunk in s.array_chunks::<BATCH_SIZE>() {
        let mut slices = chunk.array_chunks().map(select_filter_load);
        let mut next_byte = || slices.next().unwrap();
        let mut accum = Simd::<u8, LANES>::splat(0);

        accum |= next_byte().simd_eq(needle).to_int().cast::<u8>() & accum_mask(0);
        accum |= next_byte().simd_eq(needle).to_int().cast::<u8>() & accum_mask(1);
        accum |= next_byte().simd_eq(needle).to_int().cast::<u8>() & accum_mask(2);
        accum |= next_byte().simd_eq(needle).to_int().cast::<u8>() & accum_mask(3);
        accum |= next_byte().simd_eq(needle).to_int().cast::<u8>() & accum_mask(4);
        accum |= next_byte().simd_eq(needle).to_int().cast::<u8>() & accum_mask(5);
        accum |= next_byte().simd_eq(needle).to_int().cast::<u8>() & accum_mask(6);
        accum |= next_byte().simd_eq(needle).to_int().cast::<u8>() & accum_mask(7);

        let (ca, cb) = lut_lookup(accum);
        counter += (ca + cb).reduce_sum() as usize;
        debug_assert!(slices.next().is_none());
    }

    counter
}
