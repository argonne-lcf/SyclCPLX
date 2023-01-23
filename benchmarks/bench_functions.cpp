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

template <typename R> class BenchmarkData {
public:
  BenchmarkData(std::size_t max_n)
      : q_(sycl::default_selector_v), max_n_(max_n) {
    R *h_random_data = sycl::malloc_host<R>(max_n * 4, q_);

    d_a_ = sycl::malloc_device<R>(max_n * 2, q_);
    d_b_ = sycl::malloc_device<R>(max_n * 2, q_);

    fill_random(h_random_data, max_n * 4);
    q_.copy(h_random_data, d_a_, 2 * max_n);
    q_.copy(h_random_data + 2 * max_n, d_b_, 2 * max_n);
    q_.wait();

    sycl::free(h_random_data, q_);
  }

  ~BenchmarkData() {
    sycl::free(d_a_, q_);
    sycl::free(d_b_, q_);
  }

  template <typename T> T *get_device_input(std::size_t n) {
    assert(n <= max_n_);
    return reinterpret_cast<T *>(d_a_);
  }

  template <typename T> T *get_device_output(std::size_t n) {
    assert(n <= max_n_);
    return reinterpret_cast<T *>(d_b_);
  }

  sycl::queue &get_queue() { return q_; }

private:
  sycl::queue q_;
  std::size_t max_n_;
  R *h_random_data_;
  R *d_a_;
  R *d_b_;
};

template <typename R> auto get_benchmark_data(std::size_t max_n) {
  static BenchmarkData<R> *data = nullptr;
  if (data == nullptr) {
    data = new BenchmarkData<R>(max_n);
  }
  return data;
}

template <Cplx cplx, typename R, FunctionName F, std::uint32_t SEED = 777>
static void BM_function(benchmark::State &state) {
  using T = complex_t<cplx, R>;

  int n = state.range(0);

  complex_function<cplx, R, F> fn{};

  auto bench_data = get_benchmark_data<R>(n);

  auto a = bench_data->template get_device_input<T>(n);
  auto b = bench_data->template get_device_output<T>(n);

  sycl::queue &Q = bench_data->get_queue();

  for (auto _ : state) {
    Q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { b[i] = fn(a[i]); });
    Q.wait();
  }
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
