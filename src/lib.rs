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

const GROUP_SIZE: usize = 32;
pub const W: usize = 512;

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
    #[inline(always)]
    fn select_filter_load(s: &[i16; 16]) -> Simd<u8, 16> {
        let raw = s.map(i16::to_ne_bytes);
        let bytes = raw.as_flattened();

        let zero = Simd::splat(0);
        let mask = Mask::from_bitmask(0b0101_0101_0101_0101);
        let h1 = Simd::load_select(&bytes[..16], mask, zero);
        let h2 = Simd::load_select(&bytes[16..], mask, zero);

        h1 | h2.rotate_elements_right::<1>()
    }

    let needle = Simd::splat(b'\n');
    let popcnt_4b_lut = Simd::from_array(array::from_fn::<u8, 16, _>(|i| i as u8)).count_ones();

    let mut acc = 0;
    for chunk in s.array_chunks::<128>() {
        let mut slices = chunk.array_chunks::<16>().map(select_filter_load);

        let mut take = || slices.next().unwrap();
        let mut found_field = Simd::<u8, 16>::splat(0);

        let cand_a = take();
        let eq_mask_a = cand_a.simd_eq(needle);
        found_field |= eq_mask_a.to_int().cast::<u8>() & Simd::splat(1 << 0);

        let cand_b = take();
        let eq_mask_b = cand_b.simd_eq(needle);
        found_field |= eq_mask_b.to_int().cast::<u8>() & Simd::splat(1 << 1);

        let cand_c = take();
        let eq_mask_c = cand_c.simd_eq(needle);
        found_field |= eq_mask_c.to_int().cast::<u8>() & Simd::splat(1 << 2);

        let cand_d = take();
        let eq_mask_d = cand_d.simd_eq(needle);
        found_field |= eq_mask_d.to_int().cast::<u8>() & Simd::splat(1 << 3);

        let cand_e = take();
        let eq_mask_e = cand_e.simd_eq(needle);
        found_field |= eq_mask_e.to_int().cast::<u8>() & Simd::splat(1 << 4);

        let cand_f = take();
        let eq_mask_f = cand_f.simd_eq(needle);
        found_field |= eq_mask_f.to_int().cast::<u8>() & Simd::splat(1 << 5);

        let cand_g = take();
        let eq_mask_g = cand_g.simd_eq(needle);
        found_field |= eq_mask_g.to_int().cast::<u8>() & Simd::splat(1 << 6);

        let cand_h = take();
        let eq_mask_h = cand_h.simd_eq(needle);
        found_field |= eq_mask_h.to_int().cast::<u8>() & Simd::splat(1 << 7);

        let ca = popcnt_4b_lut.swizzle_dyn(found_field & Simd::splat(0b1111_0000));
        let cb = popcnt_4b_lut.swizzle_dyn(found_field & Simd::splat(0b0000_1111));
        acc += (ca + cb).reduce_sum() as usize;
        debug_assert!(slices.next().is_none());
    }

    acc
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
