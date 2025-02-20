#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(slice_as_chunks)]
#![feature(stmt_expr_attributes)]
#![feature(portable_simd)]
#![feature(array_chunks)]

use std::convert::TryInto;

type Q = i16;

const GROUP_SIZE: usize = 32;
pub const W: usize = 256;

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
    use std::simd::prelude::*;

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
