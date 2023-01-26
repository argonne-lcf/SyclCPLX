#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <iomanip>
#include <sycl/sycl.hpp>

#undef __FAST_MATH__
#undef _GLIBCXX_FAST_MATH
#undef _M_FP_FAST

#include "sycl_ext_complex.hpp"

#define SYCL_CPLX_TOL_ULP 5

// Helpers for check if type is supported
template <typename T> inline bool is_type_supported(sycl::queue &Q) {
  return false;
}

template <> inline bool is_type_supported<double>(sycl::queue &Q) {
  return Q.get_device().has(sycl::aspect::fp64);
}

template <> inline bool is_type_supported<float>(sycl::queue &Q) {
  return true;
}

template <> inline bool is_type_supported<sycl::half>(sycl::queue &Q) {
  return Q.get_device().has(sycl::aspect::fp16);
}

// Helper for passing infinity and nan values
template <typename T>
inline constexpr T inf_val = std::numeric_limits<T>::infinity();

template <typename T>
inline constexpr T nan_val = std::numeric_limits<T>::quiet_NaN();

// Helper class for passing complex arguments

template <typename T> struct cmplx {
  constexpr cmplx() : re(0), im(0) {}
  constexpr cmplx(T real, T imag) : re(real), im(imag) {}

  template <typename X> cmplx(cmplx<X> c) {
    re = c.re;
    im = c.im;
  }

  T re;
  T im;
};

// Helpers for comparison

// Do not define for public `intel/llvm` and oneAPI
// oneAPI define `__INTEL_LLVM_COMPILER` but `intel/llvm` doesn't.
// So we use the intel-implementation-only macro `__SYCL_COMPILER_VERSION` as an
// ID
#ifndef __SYCL_COMPILER_VERSION
namespace std {

// Specialization of std::numeric<sycl::half> for almost_equal
template <> struct numeric_limits<sycl::half> {
  static constexpr const sycl::half(min)() noexcept { return 6.103515625e-05f; }

  static constexpr const sycl::half epsilon() noexcept { return 9.765625e-04f; }
};

} // namespace std
#endif

template <typename T, typename U> inline bool is_nan_or_inf(T x, U y) {
  return (std::isnan(x.real()) && std::isnan(y.real())) ||
         (std::isnan(x.imag()) && std::isnan(y.imag())) ||
         (std::isinf(x.real()) && std::isinf(y.real())) ||
         (std::isinf(x.imag()) && std::isinf(y.imag()));
}

namespace detail {
template <typename T> bool almost_equal(T x, T y, int ulp) {
  if (std::isnan(x) && std::isnan(y))
    return true;
  else if (std::isinf(x) && std::isinf(y))
    return true;

  return std::abs(x - y) <=
             std::numeric_limits<T>::epsilon() * std::abs(x + y) * ulp ||
         std::abs(x - y) < std::numeric_limits<T>::min();
}

template <typename T>
bool almost_equal(sycl::ext::cplx::complex<T> x, std::complex<T> y, int ulp) {
  auto diff = std::abs((std::complex<T>)x - y);
  return diff <= std::numeric_limits<T>::epsilon() *
                     std::abs((std::complex<T>)x + y) * ulp ||
         diff < std::numeric_limits<T>::min() || is_nan_or_inf(x, y);
}

// Helpers for testing half

inline std::complex<float>
sycl_half_to_float(sycl::ext::cplx::complex<sycl::half> c) {
  auto c_sycl_float = static_cast<sycl::ext::cplx::complex<float>>(c);
  return static_cast<std::complex<float>>(c_sycl_float);
}

inline std::complex<float> trunc_float(std::complex<float> c) {
  auto c_sycl_half = static_cast<sycl::ext::cplx::complex<sycl::half>>(c);
  return sycl_half_to_float(c_sycl_half);
}
} // namespace detail

// Helper for initializing std::complex values for tests only needed because
// sycl::half cases are emulated with float for std::complex class

template <typename T_in> auto constexpr init_std_complex(cmplx<T_in> c) {
  return std::complex<T_in>(c.re, c.im);
}

template <> auto constexpr init_std_complex(cmplx<sycl::half> c) {
  return detail::trunc_float(std::complex<float>(c.re, c.im));
}

template <typename T_in> auto constexpr init_deci(T_in re) { return re; }

template <> auto constexpr init_deci(sycl::half re) {
  return static_cast<float>(re);
}

// Helpers for comparing SyclCPLX and standard c++ results

template <typename T>
void check_results(sycl::ext::cplx::complex<T> output,
                   std::complex<T> reference, int tol_multiplier = 1) {
  CHECK(detail::almost_equal(output, reference,
                             tol_multiplier * SYCL_CPLX_TOL_ULP));
}

template <typename T>
void check_results(T output, T reference, int tol_multiplier = 1) {
  CHECK(detail::almost_equal(output, reference,
                             tol_multiplier * SYCL_CPLX_TOL_ULP));
}
