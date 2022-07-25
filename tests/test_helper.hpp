#include <cmath>
#include <iomanip>
#include <sycl/sycl.hpp>

#include "sycl_ext_complex.hpp"

#define SYCL_CPLX_TOL_ULP 5

// Helpers for displaying results

template <typename T> const char *get_typename() { return "Unknown type"; }
template <> const char *get_typename<double>() { return "double"; }
template <> const char *get_typename<float>() { return "float"; }
template <> const char *get_typename<sycl::half>() { return "sycl::half"; }

// Helper for testing each decimal type

template <template <typename> typename action, typename... argsT>
bool test_valid_types(argsT... args) {
  bool test_passes = true;

  {
    action<double> test;
    test_passes &= test(args...);
  }

  {
    action<float> test;
    test_passes &= test(args...);
  }

  {
    action<sycl::half> test;
    test_passes &= test(args...);
  }

  return test_passes;
}

// Helpers for comparison

// Do not define for DPCPP as it already defines this struct
#ifndef __INTEL_LLVM_COMPILER
namespace std {

// Specialization of std::numeric<sycl::half> for almost_equal
template <> struct numeric_limits<sycl::half> {
  static constexpr const sycl::half(min)() noexcept { return 6.103515625e-05f; }

  static constexpr const sycl::half epsilon() noexcept { return 9.765625e-04f; }
};

} // namespace std
#endif

template <typename T> bool almost_equal(T x, T y, int ulp) {
  if (std::isnan(x) && std::isnan(y))
    return true;
  else if (std::isinf(x) && std::isinf(y))
    return true;

  return std::abs(x - y) <=
             std::numeric_limits<T>::epsilon() * std::abs(x + y) * ulp ||
         std::abs(x - y) < std::numeric_limits<T>::min();
}

template <typename T, template <typename> typename T1,
          template <typename> typename T2>
bool almost_equal(T1<T> x, T2<T> y, int ulp) {
  static_assert((std::is_same<T1<T>, std::complex<T>>::value ||
                 std::is_same<T1<T>, sycl::ext::cplx::complex<T>>::value),
                "almost_equal should be used to compare complex number");
  static_assert((std::is_same<T2<T>, std::complex<T>>::value ||
                 std::is_same<T2<T>, sycl::ext::cplx::complex<T>>::value),
                "almost_equal should be used to compare complex number");
  // Somebody should be smart enough to refactor to use `enable_if`...
  return almost_equal(x.real(), y.real(), ulp) &&
         almost_equal(x.imag(), y.imag(), ulp);
}
// Helpers for testing half

std::complex<float> sycl_half_to_float(sycl::ext::cplx::complex<sycl::half> c) {
  auto c_sycl_float = static_cast<sycl::ext::cplx::complex<float>>(c);
  return static_cast<std::complex<float>>(c_sycl_float);
}

std::complex<sycl::half> sycl_float_to_half(std::complex<float> c) {
  auto c_sycl_half = static_cast<sycl::ext::cplx::complex<sycl::half>>(c);
  return static_cast<std::complex<sycl::half>>(c_sycl_half);
}

std::complex<float> trunc_float(std::complex<float> c) {
  auto c_sycl_half = static_cast<sycl::ext::cplx::complex<sycl::half>>(c);
  return sycl_half_to_float(c_sycl_half);
}

// Helper for initializing std::complex values for tests only needed because
// sycl::half cases are emulated with float for std::complex class

template <typename T_in> auto constexpr init_std_complex(T_in re, T_in im) {
  return std::complex<T_in>(re, im);
}

template <> auto constexpr init_std_complex(sycl::half re, sycl::half im) {
  return trunc_float(std::complex<float>(re, im));
}

template <typename T_in> auto constexpr init_deci(T_in re) { return re; }

template <> auto constexpr init_deci(sycl::half re) {
  return static_cast<float>(re);
}

// Helpers for comparing SyclCPLX and standard c++ results

template <typename T>
bool check_results(sycl::ext::cplx::complex<T> output,
                   std::complex<T> reference, bool is_device) {
  if (!almost_equal(output, reference, SYCL_CPLX_TOL_ULP)) {
    std::cerr << std::setprecision(std::numeric_limits<T>::max_digits10)
              << "Test failed with complex_type: " << get_typename<T>()
              << " Computed on " << (is_device ? "device" : "host")
              << " Output: " << output << " Reference: " << reference
              << std::endl;
    return false;
  }
  return true;
}

template <typename T>
bool check_results(T output, T reference, bool is_device) {
  if (!almost_equal(output, reference, SYCL_CPLX_TOL_ULP)) {
    std::cerr << std::setprecision(std::numeric_limits<T>::max_digits10)
              << "Test failed with complex_type: " << get_typename<T>()
              << " Computed on " << (is_device ? "device" : "host")
              << " Output: " << output << " Reference: " << reference
              << std::endl;
    return false;
  }
  return true;
}
