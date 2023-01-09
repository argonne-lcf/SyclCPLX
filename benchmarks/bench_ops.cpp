#include <complex>
#include <functional>
#include <type_traits>

#include <sycl/sycl.hpp>

#include "sycl_ext_complex.hpp"

#include <benchmark/benchmark.h>

#include "benchmark_common.h"
using namespace benchmark_common;

enum class OpName {
  PLUS,
  MINUS,
  MULTIPLIES,
  DIVIDES,
  MULTIPLIES_REAL,
  DIVIDES_REAL
};

template <typename T, OpName opname> struct Op;

template <typename T> struct Op<T, OpName::PLUS> {
  using type1 = T;
  using type2 = T;
  T operator()(const T &a, const T &b) const { return a + b; }
};

template <typename T> struct Op<T, OpName::MINUS> {
  using type1 = T;
  using type2 = T;
  T operator()(const T &a, const T &b) const { return a - b; }
};

template <typename T> struct Op<T, OpName::MULTIPLIES> {
  using type1 = T;
  using type2 = T;
  T operator()(const T &a, const T &b) const { return a * b; }
};

template <typename T> struct Op<T, OpName::DIVIDES> {
  using type1 = T;
  using type2 = T;
  T operator()(const T &a, const T &b) const { return a / b; }
};

template <typename T> struct Op<T, OpName::MULTIPLIES_REAL> {
  using type1 = T;
  using type2 = typename complex_value_type<T>::type;
  T operator()(const T &a, const type2 &b) const { return a * b; }
};

template <typename T> struct Op<T, OpName::DIVIDES_REAL> {
  using type1 = T;
  using type2 = typename complex_value_type<T>::type;
  T operator()(const T &a, const type2 &b) const { return a / b; }
};

template <Cplx cplx, typename R, OpName opname, std::uint32_t SEED = 777>
static void BM_binary_op(benchmark::State &state) {
  sycl::queue Q(sycl::gpu_selector_v);

  using T = complex_t<cplx, R>;
  using OpClass = Op<T, opname>;

  int n = state.range(0);

  auto h_a = sycl::malloc_host<T>(n, Q);
  auto h_b = sycl::malloc_host<typename OpClass::type2>(n, Q);
  fill_random(h_a, n);
  fill_random(h_b, n);

  auto a = sycl::malloc_device<T>(n, Q);
  auto b = sycl::malloc_device<typename OpClass::type2>(n, Q);
  auto c = sycl::malloc_device<T>(n, Q);

  Q.copy(h_a, a, n);
  Q.copy(h_b, b, n);
  Q.wait();

  sycl::free(h_a, Q);
  sycl::free(h_b, Q);

  OpClass op{};

  for (auto _ : state) {
    Q.parallel_for(sycl::range<1>(n),
                   [=](sycl::id<1> i) { c[i] = op(a[i], b[i]); });
    Q.wait();
  }

  Q.wait();

  sycl::free(a, Q);
  sycl::free(b, Q);
  sycl::free(c, Q);
}

// Size of each vector is N * 16 bytes for complex double,
// so with three vectors, that is 16 * 16 * 3 = 768 MB.
constexpr int N = 16 * 1024 * 1024;

BENCHMARK(BM_binary_op<Cplx::EXT, float, OpName::PLUS>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_binary_op<Cplx::STD, float, OpName::PLUS>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_binary_op<Cplx::EXT, double, OpName::PLUS>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_binary_op<Cplx::STD, double, OpName::PLUS>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_binary_op<Cplx::EXT, float, OpName::MINUS>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_binary_op<Cplx::STD, float, OpName::MINUS>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_binary_op<Cplx::EXT, double, OpName::MINUS>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_binary_op<Cplx::STD, double, OpName::MINUS>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_binary_op<Cplx::EXT, float, OpName::MULTIPLIES>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_binary_op<Cplx::STD, float, OpName::MULTIPLIES>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_binary_op<Cplx::EXT, double, OpName::MULTIPLIES>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_binary_op<Cplx::STD, double, OpName::MULTIPLIES>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_binary_op<Cplx::EXT, float, OpName::DIVIDES>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_binary_op<Cplx::STD, float, OpName::DIVIDES>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_binary_op<Cplx::EXT, double, OpName::DIVIDES>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_binary_op<Cplx::STD, double, OpName::DIVIDES>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_binary_op<Cplx::EXT, float, OpName::MULTIPLIES_REAL>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_binary_op<Cplx::STD, float, OpName::MULTIPLIES_REAL>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_binary_op<Cplx::EXT, double, OpName::MULTIPLIES_REAL>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_binary_op<Cplx::STD, double, OpName::MULTIPLIES_REAL>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_binary_op<Cplx::EXT, float, OpName::DIVIDES_REAL>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_binary_op<Cplx::STD, float, OpName::DIVIDES_REAL>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_binary_op<Cplx::EXT, double, OpName::DIVIDES_REAL>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_binary_op<Cplx::STD, double, OpName::DIVIDES_REAL>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
