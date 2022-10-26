#include "test_helper.hpp"

using namespace sycl::ext::cplx;

////////////////////////////////////////////////////////////////////////////////
// COMPLEX TESTS
////////////////////////////////////////////////////////////////////////////////

// Check is_gencomplex
TEST_CASE("Test is_gencomplex", "[gencomplex]") {
  static_assert(is_gencomplex<complex<double>>::value == true);
  static_assert(is_gencomplex<complex<float>>::value == true);
  static_assert(is_gencomplex<complex<sycl::half>>::value == true);

  static_assert(is_gencomplex<complex<long long>>::value == false);
  static_assert(is_gencomplex<complex<long>>::value == false);
  static_assert(is_gencomplex<complex<int>>::value == false);
  static_assert(is_gencomplex<complex<unsigned long long>>::value == false);
  static_assert(is_gencomplex<complex<unsigned long>>::value == false);
  static_assert(is_gencomplex<complex<unsigned int>>::value == false);
}

// Check is_genfloat
TEST_CASE("Test is_genfloat", "[genfloat]") {
  static_assert(is_genfloat<double>::value == true);
  static_assert(is_genfloat<float>::value == true);
  static_assert(is_genfloat<sycl::half>::value == true);

  static_assert(is_genfloat<long long>::value == false);
  static_assert(is_genfloat<long>::value == false);
  static_assert(is_genfloat<int>::value == false);
  static_assert(is_genfloat<unsigned long long>::value == false);
  static_assert(is_genfloat<unsigned long>::value == false);
  static_assert(is_genfloat<unsigned int>::value == false);
}
