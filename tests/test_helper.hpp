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

template <template <typename, std::size_t> typename action,
          std::size_t NumElements, typename... argsT>
bool test_valid_types(argsT... args) {
  bool test_passes = true;

  {
    action<double, NumElements> test;
    test_passes &= test(args...);
  }

  {
    action<float, NumElements> test;
    test_passes &= test(args...);
  }

  {
    action<sycl::half, NumElements> test;
    test_passes &= test(args...);
  }

  return test_passes;
}

template <template <typename, std::size_t, std::size_t...> typename action,
          std::size_t NumElements, std::size_t... I, typename... argsT>
bool test_valid_types(sycl::queue Q,
                      std::integer_sequence<std::size_t, I...> int_seq,
                      argsT... args) {
  bool test_passes = true;

  {
    action<double, NumElements, I...> test;
    test_passes &= test(Q, int_seq, args...);
  }

  {
    action<float, NumElements, I...> test;
    test_passes &= test(Q, int_seq, args...);
  }

  {
    action<sycl::half, NumElements, I...> test;
    test_passes &= test(Q, int_seq, args...);
  }

  return test_passes;
}

// Helper classes to handle implicit conversion for passing marray types

template <typename T, std::size_t NumElements> class test_marray {
  sycl::marray<T, NumElements> Data;

public:
  template <typename... ArgTN>
  constexpr test_marray(const ArgTN &... args) : Data{args...} {};

  T operator[](std::size_t index) const { return Data[index]; }

  template <typename X> test_marray(test_marray<X, NumElements> &input) {
    for (std::size_t i = 0; i < NumElements; ++i) {
      Data[i] = input[i];
    }
  }

  sycl::marray<T, NumElements> &get() { return Data; }
  const sycl::marray<T, NumElements> &get() const { return Data; }
};

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

template <typename T, typename U> inline bool is_nan_or_inf(T x, U y) {
  return (std::isnan(x.real()) && std::isnan(y.real())) ||
         (std::isnan(x.imag()) && std::isnan(y.imag())) ||
         (std::isinf(x.real()) && std::isinf(y.real())) ||
         (std::isinf(x.imag()) && std::isinf(y.imag()));
}

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
  auto diff = std::abs((T2<T>)x - y);
  return diff <=
             std::numeric_limits<T>::epsilon() * std::abs((T2<T>)x + y) * ulp ||
         diff < std::numeric_limits<T>::min() || is_nan_or_inf(x, y);
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

template <typename T_in, std::size_t NumElements>
auto constexpr init_std_complex(sycl::marray<T_in, NumElements> re,
                                sycl::marray<T_in, NumElements> im) {
  sycl::marray<std::complex<T_in>, NumElements> rtn;

  for (std::size_t i = 0; i < rtn.size(); ++i)
    rtn[i] = std::complex<T_in>(re[i], im[i]);

  return rtn;
}

template <std::size_t NumElements>
auto constexpr init_std_complex(sycl::marray<sycl::half, NumElements> re,
                                sycl::marray<sycl::half, NumElements> im) {
  sycl::marray<std::complex<float>, NumElements> rtn;

  for (std::size_t i = 0; i < rtn.size(); ++i)
    rtn[i] = trunc_float(std::complex<float>(re[i], im[i]));

  return rtn;
}

template <typename T_in> auto constexpr init_deci(T_in re) { return re; }

template <> auto constexpr init_deci(sycl::half re) {
  return static_cast<float>(re);
}

template <typename T_in, std::size_t NumElements>
auto constexpr init_deci(sycl::marray<T_in, NumElements> re) {
  return re;
}

template <std::size_t NumElements>
auto constexpr init_deci(sycl::marray<sycl::half, NumElements> re) {
  sycl::marray<float, NumElements> rtn;
  for (std::size_t i = 0; i < NumElements; ++i)
    rtn[i] = static_cast<float>(re[i]);
  return rtn;
}

// Helper to change marray of std::complex value type

template <typename Tout, typename Tin, std::size_t NumElements>
auto constexpr convert_marray(sycl::marray<std::complex<Tin>, NumElements> c) {
  sycl::marray<std::complex<Tout>, NumElements> rtn;

  for (std::size_t i = 0; i < NumElements; ++i)
    rtn[i] = c[i];

  return rtn;
}

// Helpers for comparing SyclCPLX and standard c++ results

template <typename T>
bool check_results(sycl::ext::cplx::complex<T> output,
                   std::complex<T> reference, bool is_device,
                   int tol_multiplier = 1) {
  if (!almost_equal(output, reference, tol_multiplier * SYCL_CPLX_TOL_ULP)) {
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
bool check_results(T output, T reference, bool is_device,
                   int tol_multiplier = 1) {
  if (!almost_equal(output, reference, tol_multiplier * SYCL_CPLX_TOL_ULP)) {
    std::cerr << std::setprecision(std::numeric_limits<T>::max_digits10)
              << "Test failed with complex_type: " << get_typename<T>()
              << " Computed on " << (is_device ? "device" : "host")
              << " Output: " << output << " Reference: " << reference
              << std::endl;
    return false;
  }
  return true;
}

template <typename T, std::size_t NumElements>
bool check_results(
    sycl::marray<sycl::ext::cplx::complex<T>, NumElements> output,
    sycl::marray<std::complex<T>, NumElements> reference, bool is_device,
    int tol_multiplier = 1) {
  bool test_passes = true;
  for (std::size_t i = 0; i < NumElements; ++i)
    test_passes &= check_results(output[i], reference[i], is_device);

  return test_passes;
}

template <typename T, std::size_t NumElements>
bool check_results(sycl::marray<T, NumElements> output,
                   sycl::marray<T, NumElements> reference, bool is_device,
                   int tol_multiplier = 1) {
  bool test_passes = true;
  for (std::size_t i = 0; i < NumElements; ++i)
    test_passes &= check_results<T>(output[i], reference[i], is_device);

  return test_passes;
}

template <typename T, std::size_t NumElements>
bool check_results(sycl::marray<T, NumElements> output, T reference,
                   bool is_device, int tol_multiplier = 1) {
  bool test_passes = true;
  for (std::size_t i = 0; i < NumElements; ++i)
    test_passes &= check_results<T>(output[i], reference, is_device);

  return test_passes;
}
