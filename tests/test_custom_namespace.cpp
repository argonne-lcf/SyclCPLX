#define _SYCL_CPLX_NAMESPACE custom::sycl_cplx
#include "sycl_ext_complex.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

////////////////////////////////////////////////////////////////////////////////
// COMPLEX TESTS with custom namespace (mainly to test compilation)
////////////////////////////////////////////////////////////////////////////////

TEST_CASE("Calculate complex I*I using custom namespace", "[custom_i_i]") {
  custom::sycl_cplx::complex<double> I = {0.0, 1.0};
  custom::sycl_cplx::complex<double> neg_one = {-1.0, 0.0};
  REQUIRE(I * I == neg_one);
}

TEST_CASE("Calculate complex I^2 using pow function in custom namespace",
          "[custom_pow_i_2]") {
  custom::sycl_cplx::complex<double> I = {0.0, 1.0};
  auto I_squared = custom::sycl_cplx::pow(I, 2.0);
  CHECK_THAT(I_squared.real(), Catch::Matchers::WithinRel(-1.0));
  CHECK_THAT(I_squared.imag(), Catch::Matchers::WithinAbs(0.0, 1e-15));
}
