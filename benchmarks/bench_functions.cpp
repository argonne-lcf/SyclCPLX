#include <cmath>
#include <complex>
#include <functional>
#include <type_traits>

#include <sycl/sycl.hpp>

#include "sycl_ext_complex.hpp"

#include <benchmark/benchmark.h>

#include "benchmark_common.h"
using namespace benchmark_common;

enum class FunctionName {
  SIN,
  SINH,
  ASIN,
  ASINH,
  COS,
  COSH,
  ACOS,
  ACOSH,
  TAN,
  TANH,
  ATAN,
  ATANH,
  EXP,
  LOG,
  LOG10,
  SQRT
};

template <Cplx cplx, typename R, FunctionName F> struct complex_function;

template <Cplx cplx, typename R>
struct complex_function<cplx, R, FunctionName::SIN> {
  using T = complex_t<cplx, R>;

  T operator()(const T &a) const {
    if constexpr (cplx == Cplx::EXT) {
      return sycl::ext::cplx::sin(a);
    } else {
      return std::sin(a);
    }
  }
};

template <Cplx cplx, typename R>
struct complex_function<cplx, R, FunctionName::ASIN> {
  using T = complex_t<cplx, R>;

  T operator()(const T &a) const {
    if constexpr (cplx == Cplx::EXT) {
      return sycl::ext::cplx::asin(a);
    } else {
      return std::asin(a);
    }
  }
};

template <Cplx cplx, typename R>
struct complex_function<cplx, R, FunctionName::SINH> {
  using T = complex_t<cplx, R>;

  T operator()(const T &a) const {
    if constexpr (cplx == Cplx::EXT) {
      return sycl::ext::cplx::sinh(a);
    } else {
      return std::sinh(a);
    }
  }
};

template <Cplx cplx, typename R>
struct complex_function<cplx, R, FunctionName::LOG> {
  using T = complex_t<cplx, R>;

  T operator()(const T &a) const {
    if constexpr (cplx == Cplx::EXT) {
      return sycl::ext::cplx::log(a);
    } else {
      return std::log(a);
    }
  }
};

template <Cplx cplx, typename R>
struct complex_function<cplx, R, FunctionName::SQRT> {
  using T = complex_t<cplx, R>;

  T operator()(const T &a) const {
    if constexpr (cplx == Cplx::EXT) {
      return sycl::ext::cplx::sqrt(a);
    } else {
      return std::sqrt(a);
    }
  }
};

template <Cplx cplx, typename R>
struct complex_function<cplx, R, FunctionName::EXP> {
  using T = complex_t<cplx, R>;

  T operator()(const T &a) const {
    if constexpr (cplx == Cplx::EXT) {
      return sycl::ext::cplx::exp(a);
    } else {
      return std::exp(a);
    }
  }
};

template <Cplx cplx, typename R, FunctionName F, std::uint32_t SEED = 777>
static void BM_function(benchmark::State &state) {
  sycl::queue Q(sycl::gpu_selector_v);

  using T = complex_t<cplx, R>;

  int n = state.range(0);

  complex_function<cplx, R, F> fn{};

  auto h_a = sycl::malloc_host<T>(n, Q);
  fill_random(h_a, n);

  auto a = sycl::malloc_device<T>(n, Q);
  auto b = sycl::malloc_device<T>(n, Q);

  Q.copy(h_a, a, n);
  Q.wait();

  for (auto _ : state) {
    Q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { b[i] = fn(a[i]); });
    Q.wait();
  }

  sycl::free(a, Q);
  sycl::free(b, Q);
}

// Size of each vector is N * 16 bytes for complex double,
// so with three vectors, that is 16 * 16 * 3 = 768 MB.
constexpr int N = 16 * 1024 * 1024;

BENCHMARK(BM_function<Cplx::EXT, float, FunctionName::SIN>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_function<Cplx::STD, float, FunctionName::SIN>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_function<Cplx::EXT, double, FunctionName::SIN>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_function<Cplx::STD, double, FunctionName::SIN>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_function<Cplx::EXT, float, FunctionName::SINH>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_function<Cplx::STD, float, FunctionName::SINH>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_function<Cplx::EXT, double, FunctionName::SINH>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_function<Cplx::STD, double, FunctionName::SINH>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_function<Cplx::EXT, float, FunctionName::ASIN>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_function<Cplx::STD, float, FunctionName::ASIN>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_function<Cplx::EXT, double, FunctionName::ASIN>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_function<Cplx::STD, double, FunctionName::ASIN>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_function<Cplx::EXT, float, FunctionName::LOG>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_function<Cplx::STD, float, FunctionName::LOG>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_function<Cplx::EXT, double, FunctionName::LOG>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_function<Cplx::STD, double, FunctionName::LOG>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_function<Cplx::EXT, float, FunctionName::SQRT>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_function<Cplx::STD, float, FunctionName::SQRT>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_function<Cplx::EXT, double, FunctionName::SQRT>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_function<Cplx::STD, double, FunctionName::SQRT>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_function<Cplx::EXT, float, FunctionName::EXP>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_function<Cplx::STD, float, FunctionName::EXP>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_function<Cplx::EXT, double, FunctionName::EXP>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_function<Cplx::STD, double, FunctionName::EXP>)
    ->Args({N})
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
