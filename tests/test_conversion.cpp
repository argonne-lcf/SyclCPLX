#include "test_helper.hpp"

#include <array>
#include <cmath>

// Check conversion sycl:complex to std::complex
TEMPLATE_TEST_CASE("Test sycl::complex to std::complex conversion",
                   "[conversion]", double, float, sycl::half) {
  using T = TestType;

  auto arr = std::array<sycl::ext::cplx::complex<T>, 8>{
      sycl::ext::cplx::complex<T>{0, 0},  sycl::ext::cplx::complex<T>{-0, 0},
      sycl::ext::cplx::complex<T>{0, -0}, sycl::ext::cplx::complex<T>{-0, -0},

      sycl::ext::cplx::complex<T>{1, 1},  sycl::ext::cplx::complex<T>{-1, 1},
      sycl::ext::cplx::complex<T>{1, -1}, sycl::ext::cplx::complex<T>{-1, -1},
  };

  for (const auto &lhs : arr) {
    const auto rhs = static_cast<std::complex<T>>(lhs);

    assert(lhs.real() == rhs.real() &&
           "sycl::complex differs from std::complex after conversion");
    assert(lhs.imag() == rhs.imag() &&
           "sycl::complex differs from std::complex after conversion");
  }
}

// Check edge-cases conversion sycl:complex to std::complex
TEMPLATE_TEST_CASE("Test edge-cases sycl::complex to std::complex conversion",
                   "[conversion]", double, float, sycl::half) {
  using T = TestType;

  {
    auto lhs = sycl::ext::cplx::complex<T>{inf_val<T>, inf_val<T>};
    auto rhs = static_cast<std::complex<T>>(lhs);

    assert(std::isinf(lhs.real()) && std::isinf(rhs.real()) &&
           "sycl::complex differs from std::complex after conversion");
    assert(std::isinf(lhs.imag()) && std::isinf(rhs.imag()) &&
           "sycl::complex differs from std::complex after conversion");
  }
  {
    auto lhs = sycl::ext::cplx::complex<T>{inf_val<T>, nan_val<T>};
    auto rhs = static_cast<std::complex<T>>(lhs);

    assert(std::isinf(lhs.real()) && std::isinf(rhs.real()) &&
           "sycl::complex differs from std::complex after conversion");
    assert(std::isnan(lhs.imag()) && std::isnan(rhs.imag()) &&
           "sycl::complex differs from std::complex after conversion");
  }
  {
    auto lhs = sycl::ext::cplx::complex<T>{nan_val<T>, inf_val<T>};
    auto rhs = static_cast<std::complex<T>>(lhs);

    assert(std::isnan(lhs.real()) && std::isnan(rhs.real()) &&
           "sycl::complex differs from std::complex after conversion");
    assert(std::isinf(lhs.imag()) && std::isinf(rhs.imag()) &&
           "sycl::complex differs from std::complex after conversion");
  }
  {
    auto lhs = sycl::ext::cplx::complex<T>{nan_val<T>, nan_val<T>};
    auto rhs = static_cast<std::complex<T>>(lhs);

    assert(std::isnan(lhs.real()) && std::isnan(rhs.real()) &&
           "sycl::complex differs from std::complex after conversion");
    assert(std::isnan(lhs.imag()) && std::isnan(rhs.imag()) &&
           "sycl::complex differs from std::complex after conversion");
  }
}
