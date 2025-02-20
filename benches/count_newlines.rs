#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use count_bytes::W;
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

const fn generate_data() -> [i16; W] {
    let mut data: [i16; W] = [0; W];

    let mut i = 0;
    while i < W {
        data[i] = (i.wrapping_sub(673)).wrapping_mul(97) as i16;
        i += 1;
    }

    data
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("count_newlines");
    group.throughput(criterion::Throughput::Bytes(count_bytes::W as u64));

    //group.bench_function("naive", |b| {
    //    b.iter_batched_ref(
    //        || generate_data(),
    //        |data| count_bytes::naive(black_box(data)),
    //        criterion::BatchSize::SmallInput,
    //    )
    //});
//
    //group.bench_function("divide & conquer", |b| {
    //    b.iter_batched_ref(
    //        || generate_data(),
    //        |data| count_bytes::divide(black_box(data)),
    //        criterion::BatchSize::SmallInput,
    //    )
    //});

    group.bench_function("divide & conquer + simd", |b| {
        b.iter_batched_ref(
            || generate_data(),
            |data| count_bytes::divide_simd(black_box(data)),
            criterion::BatchSize::SmallInput,
        )
    });

    group.bench_function("interleaved pipelined", |b| {
        b.iter_batched_ref(
            || generate_data(),
            |data| count_bytes::interleaved_pipelined(black_box(data)),
            criterion::BatchSize::SmallInput,
        )
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
