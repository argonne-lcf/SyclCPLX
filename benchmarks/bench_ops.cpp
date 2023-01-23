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

template <typename R> class BenchmarkData {
public:
  BenchmarkData(std::size_t max_n)
      : q_(sycl::default_selector_v), max_n_(max_n) {
    R *h_random_data = sycl::malloc_host<R>(max_n * 4, q_);

    d_a_ = sycl::malloc_device<R>(max_n * 2, q_);
    d_b_ = sycl::malloc_device<R>(max_n * 2, q_);
    d_c_ = sycl::malloc_device<R>(max_n * 2, q_);

    fill_random(h_random_data, max_n * 4);
    q_.copy(h_random_data, d_a_, 2 * max_n);
    q_.copy(h_random_data + 2 * max_n, d_b_, 2 * max_n);
    q_.wait();

    sycl::free(h_random_data, q_);
  }

  ~BenchmarkData() {
    sycl::free(d_a_, q_);
    sycl::free(d_b_, q_);
    sycl::free(d_c_, q_);
  }

  template <typename T> T *get_device_input1(std::size_t n) {
    assert(n <= max_n_);
    return reinterpret_cast<T *>(d_a_);
  }

  template <typename T> T *get_device_input2(std::size_t n) {
    assert(n <= max_n_);
    return reinterpret_cast<T *>(d_b_);
  }

  template <typename T> T *get_device_output(std::size_t n) {
    assert(n <= max_n_);
    return reinterpret_cast<T *>(d_c_);
  }

  sycl::queue &get_queue() { return q_; }

private:
  sycl::queue q_;
  std::size_t max_n_;
  R *h_random_data_;
  R *d_a_;
  R *d_b_;
  R *d_c_;
};

template <typename R> auto get_benchmark_data(std::size_t max_n) {
  static BenchmarkData<R>* data = nullptr;
  if (data == nullptr) {
    data = new BenchmarkData<R>(max_n);
  }
  return data;
}

template <Cplx cplx, typename R, OpName opname, std::uint32_t SEED = 777>
static void BM_binary_op(benchmark::State &state) {
  using T = complex_t<cplx, R>;
  using OpClass = Op<T, opname>;

  int n = state.range(0);

  auto bench_data = get_benchmark_data<R>(n);

  auto a = bench_data->template get_device_input1<typename OpClass::type1>(n);
  auto b = bench_data->template get_device_input2<typename OpClass::type2>(n);
  auto c = bench_data->template get_device_output<T>(n);

  sycl::queue &Q = bench_data->get_queue();

  OpClass op{};

  for (auto _ : state) {
    Q.parallel_for(sycl::range<1>(n),
                   [=](sycl::id<1> i) { c[i] = op(a[i], b[i]); });
    Q.wait();
  }
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
