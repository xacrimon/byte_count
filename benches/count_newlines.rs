#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

const W: usize = 64 * 1024;

fn generate_data() -> Vec<i16> {
    let mut data = vec![0; W];

    let mut i = 0;
    while i < W {
        data[i] = (i.wrapping_sub(673)).wrapping_mul(97) as i16;
        i += 1;
    }

    black_box(data)
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("count_newlines");
    group.throughput(criterion::Throughput::Bytes(W as u64));

    group.bench_function("naive", |b| {
        b.iter_batched_ref(
            || generate_data(),
            |data| count_bytes::naive(black_box(data)),
            criterion::BatchSize::LargeInput,
        )
    });

    group.bench_function("interleaved pipelined", |b| {
        b.iter_batched_ref(
            || generate_data(),
            |data| count_bytes::interleaved_pipelined(black_box(data)),
            criterion::BatchSize::LargeInput,
        )
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
