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
pub const W: usize = 128;

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

        let [h1, h2] = unsafe {
            [
                Simd::<u8, 16>::from_array(bytes[..16].try_into().unwrap_unchecked()),
                Simd::<u8, 16>::from_array(bytes[16..].try_into().unwrap_unchecked()),
            ]
        };

        simd_swizzle!(
            h1,
            h2,
            [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
        )
    }

    let needle = Simd::splat(b'\n');
    let lut_masks = [Simd::splat(0b1111_0000), Simd::splat(0b0000_1111)];
    let popcnt_4b_lut = Simd::from_array(array::from_fn::<u8, 16, _>(|i| i.count_ones() as u8));
    let lut_lookup = |bitfield| lut_masks.map(|mask| popcnt_4b_lut.swizzle_dyn(bitfield & mask));
    let accum_mask = |offset: u32| Simd::splat(1 << offset);

    let mut counter = 0;
    for chunk in s.array_chunks::<128>() {
        let mut slices = chunk.array_chunks::<16>().map(select_filter_load);
        let mut next_byte = || slices.next().unwrap();
        let mut accum = Simd::<u8, 16>::splat(0);

        accum |= next_byte().simd_eq(needle).to_int().cast::<u8>() & accum_mask(0);
        accum |= next_byte().simd_eq(needle).to_int().cast::<u8>() & accum_mask(1);
        accum |= next_byte().simd_eq(needle).to_int().cast::<u8>() & accum_mask(2);
        accum |= next_byte().simd_eq(needle).to_int().cast::<u8>() & accum_mask(3);
        accum |= next_byte().simd_eq(needle).to_int().cast::<u8>() & accum_mask(4);
        accum |= next_byte().simd_eq(needle).to_int().cast::<u8>() & accum_mask(5);
        accum |= next_byte().simd_eq(needle).to_int().cast::<u8>() & accum_mask(6);
        accum |= next_byte().simd_eq(needle).to_int().cast::<u8>() & accum_mask(7);

        let [ca, cb] = lut_lookup(accum);
        counter += (ca + cb).reduce_sum() as usize;
        debug_assert!(slices.next().is_none());
    }

    counter
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
