#include <cmath>
#include <iomanip>
#include <sycl/sycl.hpp>

#include "sycl_ext_complex.hpp"

#define SYCL_CPLX_TOL_ULP 5

#define __half_epsilon 9.765625e-04f
#define __half_min 6.103515625e-05f

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

bool almost_equal(double x, double y, int ulp) {
  if (std::isnan(x) && std::isnan(y))
    return true;
  else if (std::isinf(x) && std::isinf(y))
    return true;

  return std::abs(x - y) <=
             std::numeric_limits<double>::epsilon() * std::abs(x + y) * ulp ||
         std::abs(x - y) < std::numeric_limits<double>::min();
}

bool almost_equal(float x, float y, int ulp) {
  if (std::isnan(x) && std::isnan(y))
    return true;
  else if (std::isinf(x) && std::isinf(y))
    return true;

  return std::abs(x - y) <=
             std::numeric_limits<float>::epsilon() * std::abs(x + y) * ulp ||
         std::abs(x - y) < std::numeric_limits<float>::min();
}

bool almost_equal(sycl::half x, sycl::half y, int ulp) {
  if (sycl::isnan(x) && sycl::isnan(y))
    return true;
  else if (sycl::isinf(x) && sycl::isinf(y))
    return true;

  return std::abs(x - y) <= __half_epsilon * std::abs(x + y) * ulp ||
         std::abs(x - y) < __half_min;
}

bool almost_equal(sycl::ext::cplx::complex<double> x,
                  sycl::ext::cplx::complex<double> y, int ulp) {
  return almost_equal(x.real(), y.real(), ulp) &&
         almost_equal(x.imag(), y.imag(), ulp);
}

bool almost_equal(sycl::ext::cplx::complex<double> x, std::complex<double> y,
                  int ulp) {
  return almost_equal(x.real(), y.real(), ulp) &&
         almost_equal(x.imag(), y.imag(), ulp);
}

bool almost_equal(std::complex<double> x, sycl::ext::cplx::complex<double> y,
                  int ulp) {
  return almost_equal(x.real(), y.real(), ulp) &&
         almost_equal(x.imag(), y.imag(), ulp);
}

bool almost_equal(std::complex<double> x, std::complex<double> y, int ulp) {
  return almost_equal(x.real(), y.real(), ulp) &&
         almost_equal(x.imag(), y.imag(), ulp);
}

bool almost_equal(sycl::ext::cplx::complex<float> x,
                  sycl::ext::cplx::complex<float> y, int ulp) {
  return almost_equal(x.real(), y.real(), ulp) &&
         almost_equal(x.imag(), y.imag(), ulp);
}

bool almost_equal(sycl::ext::cplx::complex<float> x, std::complex<float> y,
                  int ulp) {
  return almost_equal(x.real(), y.real(), ulp) &&
         almost_equal(x.imag(), y.imag(), ulp);
}

bool almost_equal(std::complex<float> x, sycl::ext::cplx::complex<float> y,
                  int ulp) {
  return almost_equal(x.real(), y.real(), ulp) &&
         almost_equal(x.imag(), y.imag(), ulp);
}

bool almost_equal(std::complex<float> x, std::complex<float> y, int ulp) {
  return almost_equal(x.real(), y.real(), ulp) &&
         almost_equal(x.imag(), y.imag(), ulp);
}

bool almost_equal(sycl::ext::cplx::complex<sycl::half> x,
                  sycl::ext::cplx::complex<sycl::half> y, int ulp) {
  return almost_equal(x.real(), y.real(), ulp) &&
         almost_equal(x.imag(), y.imag(), ulp);
}

bool almost_equal(sycl::ext::cplx::complex<sycl::half> x,
                  std::complex<sycl::half> y, int ulp) {
  return almost_equal(x.real(), y.real(), ulp) &&
         almost_equal(x.imag(), y.imag(), ulp);
}

bool almost_equal(std::complex<sycl::half> x,
                  sycl::ext::cplx::complex<sycl::half> y, int ulp) {
  return almost_equal(x.real(), y.real(), ulp) &&
         almost_equal(x.imag(), y.imag(), ulp);
}

bool almost_equal(std::complex<sycl::half> x, std::complex<sycl::half> y,
                  int ulp) {
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

template <typename T_in> auto constexpr init_std(T_in re, T_in im) {
  return std::complex<T_in>(re, im);
}

template <> auto constexpr init_std(sycl::half re, sycl::half im) {
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
              << " Output: (" << output.real() << "," << output.imag() << ")"
              << " Reference: (" << reference.real() << "," << reference.imag()
              << ")" << std::endl;
    return false;
  }
  return true;
}
