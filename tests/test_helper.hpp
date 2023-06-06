#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_tostring.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>

#include <cmath>
#include <iomanip>

#include <sycl/sycl.hpp>

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

namespace detail {

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

template <typename T>
bool almost_equal(std::complex<T> output, std::complex<T> reference, int ulp) {
  auto diff = std::abs(output - reference);
  return diff <= std::numeric_limits<T>::epsilon() *
                     std::abs(output + reference) * ulp ||
         diff < std::numeric_limits<T>::min() ||
         is_nan_or_inf(output, reference);
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

template <typename T_in, std::size_t NumElements>
auto constexpr init_std_complex(
    sycl::marray<std::complex<T_in>, NumElements> input) {
  return input;
}

template <std::size_t NumElements>
auto constexpr init_std_complex(
    sycl::marray<std::complex<sycl::half>, NumElements> input) {
  sycl::marray<std::complex<float>, NumElements> rtn;
  for (std::size_t i = 0; i < rtn.size(); ++i) {
    rtn[i] = detail::trunc_float(
        std::complex<float>(input[i].real(), input[i].imag()));
  }
  return rtn;
}

template <typename T_in> auto constexpr init_deci(T_in re) { return re; }

template <> auto constexpr init_deci(sycl::half re) {
  return static_cast<float>(re);
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

namespace detail {

template <typename T>
struct is_a_complex
    : std::integral_constant<bool,
                             sycl::ext::cplx::is_gencomplex_v<T> ||
                                 std::is_same_v<std::complex<sycl::half>, T> ||
                                 std::is_same_v<std::complex<float>, T> ||
                                 std::is_same_v<std::complex<double>, T>> {};
template <typename T>
inline constexpr bool is_a_complex_v = is_a_complex<T>::value;

} // namespace detail

/// Specialization for double, float, sycl::half
template <typename T>
typename std::enable_if_t<sycl::ext::cplx::is_genfloat_v<T>, void>
check_results(T output, T reference, int tol_multiplier = 1) {
  CHECK(detail::almost_equal(output, reference,
                             tol_multiplier * SYCL_CPLX_TOL_ULP));
}

/// Specialization for sycl::complex and std::complex
template <typename LHS, typename RHS>
typename std::enable_if_t<
    detail::is_a_complex_v<LHS> && detail::is_a_complex_v<RHS>, void>
check_results(LHS output, RHS reference, int tol_multiplier = 1) {
  using T1 = typename LHS::value_type;
  using T2 = typename RHS::value_type;

  if (!std::is_same_v<T1, T2>) {
    FAIL("check_results can be called with sycl::complex and/or std::complex "
         "but with the same value_type");
  }

  CHECK(detail::almost_equal(static_cast<std::complex<T1>>(output),
                             static_cast<std::complex<T2>>(reference),
                             tol_multiplier * SYCL_CPLX_TOL_ULP));
}

/// Specialization for sycl::marray
template <typename LHS, std::size_t NumElementsLHS, typename RHS,
          std::size_t NumElementsRHS>
void check_results(sycl::marray<LHS, NumElementsLHS> output,
                   sycl::marray<RHS, NumElementsRHS> reference,
                   int tol_multiplier = 1) {
  if (NumElementsLHS != NumElementsRHS) {
    FAIL("check_results can be called with sycl::marray but with the same "
         "NumElement");
  }

  for (std::size_t i = 0; i < NumElementsLHS; ++i) {
    check_results(output[i], reference[i], tol_multiplier);
  }
}

/// Specialization for std::array
template <typename T, std::size_t NumElements>
void check_results(std::array<T, NumElements> output,
                   std::array<T, NumElements> reference,
                   int tol_multiplier = 1) {
  for (std::size_t i = 0; i < NumElements; ++i) {
    check_results(output[i], reference[i], tol_multiplier);
  }
}

template <typename T>
struct SyclFpMatcher : Catch::Matchers::MatcherGenericBase {
  using FpType = decltype((T{}).real());
  SyclFpMatcher(T target, int tol_multiplier = 1)
      : target_(target), tol_multiplier_(tol_multiplier) {}

  template <typename OtherT> bool match(OtherT const &other) const {
    return detail::almost_equal(other, target_,
                                tol_multiplier_ * SYCL_CPLX_TOL_ULP);
  }

  std::string describe() const override {
    auto rel_ulp = std::numeric_limits<FpType>::epsilon() * tol_multiplier_ *
                   SYCL_CPLX_TOL_ULP;
    Catch::ReusableStringStream sstr;
    sstr << "and " << target_ << " are within " << rel_ulp << " relative ULP";
    return sstr.str();
  }

private:
  T target_;
  int tol_multiplier_;
};

template <typename T>
auto SyclWithinULP(const T &target, int tol_multiplier = 1)
    -> SyclFpMatcher<T> {
  return SyclFpMatcher<T>{target, tol_multiplier};
}
