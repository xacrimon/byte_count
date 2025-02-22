#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use criterion::{criterion_group, criterion_main, Criterion};
use rand::{Rng, RngCore};
use std::hint::black_box;

const W: usize = 32 * 8192;

fn pick_inputs() -> (u8, Box<[u8]>) {
    let mut rng = rand::rng();
    let needle = rng.random();
    let mut haystack = vec![0; W].into_boxed_slice();
    rand::rng().fill_bytes(&mut haystack);
    (needle, haystack)
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("count");
    group.throughput(criterion::Throughput::Bytes(W as u64));

    group.bench_function("naive_1b", |b| {
        b.iter_batched_ref(
            || pick_inputs(),
            |(needle, haystack)| count_bytes::naive_1b(black_box(haystack), *needle),
            criterion::BatchSize::SmallInput,
        )
    });

    group.bench_function("interleaved_pipelined_1b", |b| {
        b.iter_batched_ref(
            || pick_inputs(),
            |(needle, haystack)| {
                count_bytes::interleaved_pipelined_1b(black_box(haystack), *needle)
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
