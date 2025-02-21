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
    let scalar_swiffle = |x: u32| {
        let [mut b0, mut b1, mut b2, mut b3] = x.to_le_bytes();
        // move the high nibbles in b2/b3 to b0/b1's low nibbles
        b0 |= b2 >> 4;
        b1 |= b3 >> 4;
        // clear those high nibbles
        b2 &= 0x0F;
        b3 &= 0x0F;
        u32::from_le_bytes([b0, b1, b2, b3])
    };

    let mut counter = 0;
    for chunk in s.array_chunks::<BATCH_SIZE>() {
        let mut slices = chunk.array_chunks().map(select_filter_load);
        let mut next_byte = || slices.next().unwrap();

        let mut eqfi_1 = Simd::<u32, 4>::default();
        eqfi_1.as_mut_array()[0] = scalar_swiffle(next_byte().simd_eq(needle).to_bitmask() as _);
        eqfi_1.as_mut_array()[1] = scalar_swiffle(next_byte().simd_eq(needle).to_bitmask() as _);
        eqfi_1.as_mut_array()[2] = scalar_swiffle(next_byte().simd_eq(needle).to_bitmask() as _);
        eqfi_1.as_mut_array()[3] = scalar_swiffle(next_byte().simd_eq(needle).to_bitmask() as _);
        let i1 = eqfi_1.to_le_bytes();
        let pcnt_1 = popcnt_4b_lut.swizzle_dyn(i1);
        counter += pcnt_1.reduce_sum() as usize;

        let mut eqfi_2 = Simd::<u32, 4>::default();
        eqfi_2.as_mut_array()[0] = scalar_swiffle(next_byte().simd_eq(needle).to_bitmask() as _);
        eqfi_2.as_mut_array()[1] = scalar_swiffle(next_byte().simd_eq(needle).to_bitmask() as _);
        eqfi_2.as_mut_array()[2] = scalar_swiffle(next_byte().simd_eq(needle).to_bitmask() as _);
        eqfi_2.as_mut_array()[3] = scalar_swiffle(next_byte().simd_eq(needle).to_bitmask() as _);
        let i2 = eqfi_2.to_le_bytes();
        let pcnt_2 = popcnt_4b_lut.swizzle_dyn(i2);
        counter += pcnt_2.reduce_sum() as usize;

        debug_assert!(slices.next().is_none());
    }

    counter
}
