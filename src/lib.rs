#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(slice_as_chunks)]
#![feature(stmt_expr_attributes)]
#![feature(portable_simd)]
#![feature(array_chunks)]

use std::convert::TryInto;
use std::simd::prelude::*;

type Q = i16;

const GROUP_SIZE: usize = 32;
pub const W: usize = 64;

#[no_mangle]
pub fn count_bytes(s: &[Q; W]) -> usize {
    divide_simd(s)
}

#[inline(never)]
pub fn naive(s: &[Q; W]) -> usize {
    s.iter()
        .copied()
        .filter(
            #[inline(always)]
            |&x| x == b'\n' as _,
        )
        .count()
}

#[inline(never)]
pub fn divide(s: &[Q; W]) -> usize {
    let mut count = 0;
    let s = Slice::<'_, _, W>::from(s);
    s.apply(
        &mut #[inline(always)]
        |bl: Slice<'_, _, { GROUP_SIZE }>| {
            count += bl.0.iter().copied().filter(|&x| x == b'\n' as _).count()
        },
    );
    count
}

//#[inline(always)]
//pub fn divide_simd(s: &[Q; W]) -> usize {
//    use std::simd::{Simd, cmp::SimdPartialEq, num::SimdInt};
//    let mut count = 0;
//    let s = Slice::<'_,_,W>::from(s);
//    s.apply(&mut #[inline(always)] |bl: Slice<'_, _, { GROUP_SIZE }>| count += Simd::from_array(*bl.0).simd_eq(Simd::splat(b'\n' as Q)).to_bitmask().count_ones() as usize);
//    count
//}

#[inline(never)]
pub fn divide_simd(s: &[Q; W]) -> usize {
    #[inline(always)]
    fn count_block(s: &[i16; GROUP_SIZE]) -> usize {
        let raw = s.map(i16::to_ne_bytes);
        let bytes = raw.as_flattened();
        let mid = GROUP_SIZE / 2;

        let [lo, hi] =
            [&bytes[..mid], &bytes[mid..]].map(Simd::<u8, { GROUP_SIZE / 2 }>::from_slice);

        let (i16_lo_bytes, _) = lo.deinterleave(hi);

        let needle = Simd::splat(b'\n');
        let matches = i16_lo_bytes.simd_eq(needle);

        matches.to_bitmask().count_ones() as usize
    }

    s.array_chunks::<GROUP_SIZE>().map(count_block).sum()
}

#[inline(never)]
pub fn interleaved_pipelined(s: &[Q; W]) -> usize {
    const CHK_SZ: usize = 32;
    const VEC_SZ_16: usize = 8;
    const VEC_SZ_8: usize = 16;
    //const VEC_CNT: usize = CHK_SZ / VEC_SZ_16;
    const NEEDLE: Simd<u8, VEC_SZ_8> = Simd::splat(b'\n');

    let low = |[a, b]: [Simd<u8, VEC_SZ_8>;2]| Simd::deinterleave(a, b).0;
    let flatten_array = |arr: [[u8;2];8]| unsafe { arr.as_flattened().try_into().unwrap_unchecked() };

    s.array_chunks::<CHK_SZ>()
        .map(|chunk| {
            let mut vecs = chunk
                .array_chunks::<VEC_SZ_16>()
                .copied()
                .map(|arr| arr.map(i16::to_ne_bytes))
                .map(flatten_array)
                .map(Simd::<u8,VEC_SZ_8>::from_array);

            let [r0, r1, r2, r3] = unsafe {[
                vecs.next().unwrap_unchecked(),
                vecs.next().unwrap_unchecked(),
                vecs.next().unwrap_unchecked(),
                vecs.next().unwrap_unchecked(),
            ]};

            let hay0 = low([r0, r1]);
            let hay1 = low([r2, r3]);

            let mask0 = NEEDLE.simd_eq(hay0);
            let mask1 = NEEDLE.simd_eq(hay1);

            let cmb = mask0.to_bitmask() << 0 | mask1.to_bitmask() << 16;
            cmb.count_ones() as usize
            
            //let l: mask8x4 = simd_swizzle!(mask0, mask1, [0, 1, 2, 3]);

            //mask0.to_bit
        })
        .sum()
}

enum Assert<const COND: bool> {}

trait IsTrue {}

impl IsTrue for Assert<true> {}

enum Len<const N: usize> {}

const fn do_split(x: usize) -> bool {
    x.is_power_of_two() && x > GROUP_SIZE
}

trait Split {}

impl<const N: usize> Split for Len<N> where Assert<{ do_split(N) }>: IsTrue {}

struct Slice<'a, T, const N: usize>(&'a [T; N])
where
    [T; N]:;

impl<'a, T, const N: usize> Slice<'a, T, N>
where
    [(); N]:,
{
    #[inline(always)]
    fn from(s: &'a [T]) -> Self {
        Self(s.try_into().unwrap())
    }
}

trait Apply<'a, T: 'a> {
    fn apply(self, f: &mut (impl FnMut(Slice<'a, T, { GROUP_SIZE }>) + 'a));
}

impl<'a, T, const N: usize> Apply<'a, T> for Slice<'a, T, N>
where
    Len<N>: Split,
    Slice<'a, T, { N / 2 }>: Apply<'a, T>,
{
    #[inline(always)]
    fn apply(self, f: &mut (impl FnMut(Slice<'a, T, { GROUP_SIZE }>) + 'a)) {
        let ch = unsafe { self.0.as_chunks_unchecked::<{ N / 2 }>() };
        let [l, r]: [Slice<'a, T, { N / 2 }>; 2] = [Slice(&ch[0]), Slice(&ch[1])];
        l.apply(f);
        r.apply(f);
    }
}

impl<'a, T> Apply<'a, T> for Slice<'a, T, { GROUP_SIZE }> {
    #[inline(always)]
    fn apply(self, f: &mut (impl FnMut(Slice<'a, T, { GROUP_SIZE }>) + 'a)) {
        f(self);
    }
}
