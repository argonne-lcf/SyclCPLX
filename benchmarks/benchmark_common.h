#ifndef BENCHMARK_COMMON_I
#define BENCHMARK_COMMON_I

#include <random>

namespace benchmark_common {

enum class Cplx { EXT, STD };

template <Cplx cplx, typename R>
using complex_t =
    typename std::conditional_t<cplx == Cplx::EXT, sycl::ext::cplx::complex<R>,
                                std::complex<R>>;

template <typename R, std::uint32_t SEED = 777>
inline void fill_random(R *data, size_t n) {
  std::uniform_real_distribution<R> dist(-1.0, 1.0);
  std::mt19937 engine(SEED);
  for (int i = 0; i < n; i++) {
    data[i] = dist(engine);
  }
}

template <typename R, std::uint32_t SEED = 777>
inline void fill_random(std::complex<R> *data, size_t n) {
  fill_random<R, SEED>(reinterpret_cast<R *>(data), 2 * n);
}

template <typename R, std::uint32_t SEED = 777>
inline void fill_random(sycl::ext::cplx::complex<R> *data, size_t n) {
  fill_random<R, SEED>(reinterpret_cast<R *>(data), 2 * n);
}

template <typename T> struct complex_value_type;

template <typename R> struct complex_value_type<std::complex<R>> {
  using type = R;
};

template <typename R> struct complex_value_type<sycl::ext::cplx::complex<R>> {
  using type = R;
};

} // namespace benchmark_common

#endif // BENCHMARK_COMMON_I
