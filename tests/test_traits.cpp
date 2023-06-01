#include "test_helper.hpp"

using namespace sycl::ext::cplx;

// Check is_gencomplex
TEST_CASE("Test is_gencomplex", "[traits]") {
  static_assert(is_gencomplex<complex<double>>::value == true);
  static_assert(is_gencomplex<complex<float>>::value == true);
  static_assert(is_gencomplex<complex<sycl::half>>::value == true);

  static_assert(is_gencomplex_v<complex<double>> == true);
  static_assert(is_gencomplex_v<complex<float>> == true);
  static_assert(is_gencomplex_v<complex<sycl::half>> == true);

  static_assert(is_gencomplex<complex<long long>>::value == false);
  static_assert(is_gencomplex<complex<long>>::value == false);
  static_assert(is_gencomplex<complex<int>>::value == false);
  static_assert(is_gencomplex<complex<unsigned long long>>::value == false);
  static_assert(is_gencomplex<complex<unsigned long>>::value == false);
  static_assert(is_gencomplex<complex<unsigned int>>::value == false);

  static_assert(is_gencomplex_v<complex<long long>> == false);
  static_assert(is_gencomplex_v<complex<long>> == false);
  static_assert(is_gencomplex_v<complex<int>> == false);
  static_assert(is_gencomplex_v<complex<unsigned long long>> == false);
  static_assert(is_gencomplex_v<complex<unsigned long>> == false);
  static_assert(is_gencomplex_v<complex<unsigned int>> == false);
}

// Check is_genfloat
TEST_CASE("Test is_genfloat", "[traits]") {
  static_assert(is_genfloat<double>::value == true);
  static_assert(is_genfloat<float>::value == true);
  static_assert(is_genfloat<sycl::half>::value == true);

  static_assert(is_genfloat_v<double> == true);
  static_assert(is_genfloat_v<float> == true);
  static_assert(is_genfloat_v<sycl::half> == true);

  static_assert(is_genfloat<long long>::value == false);
  static_assert(is_genfloat<long>::value == false);
  static_assert(is_genfloat<int>::value == false);
  static_assert(is_genfloat<unsigned long long>::value == false);
  static_assert(is_genfloat<unsigned long>::value == false);
  static_assert(is_genfloat<unsigned int>::value == false);

  static_assert(is_genfloat_v<long long> == false);
  static_assert(is_genfloat_v<long> == false);
  static_assert(is_genfloat_v<int> == false);
  static_assert(is_genfloat_v<unsigned long long> == false);
  static_assert(is_genfloat_v<unsigned long> == false);
  static_assert(is_genfloat_v<unsigned int> == false);
}

// Check is_mgencomplex
TEST_CASE("Test is_mgencomplex", "[traits]") {
  static_assert(is_mgencomplex<sycl::marray<complex<double>, 42>>::value ==
                true);
  static_assert(is_mgencomplex<sycl::marray<complex<float>, 42>>::value ==
                true);
  static_assert(is_mgencomplex<sycl::marray<complex<sycl::half>, 42>>::value ==
                true);

  static_assert(is_mgencomplex_v<sycl::marray<complex<double>, 42>> == true);
  static_assert(is_mgencomplex_v<sycl::marray<complex<float>, 42>> == true);
  static_assert(is_mgencomplex_v<sycl::marray<complex<sycl::half>, 42>> ==
                true);

  static_assert(is_mgencomplex<sycl::marray<complex<long long>, 42>>::value ==
                false);
  static_assert(is_mgencomplex<sycl::marray<complex<long>, 42>>::value ==
                false);
  static_assert(is_mgencomplex<sycl::marray<complex<int>, 42>>::value == false);
  static_assert(
      is_mgencomplex<sycl::marray<complex<unsigned long long>, 42>>::value ==
      false);
  static_assert(
      is_mgencomplex<sycl::marray<complex<unsigned long>, 42>>::value == false);
  static_assert(
      is_mgencomplex<sycl::marray<complex<unsigned int>, 42>>::value == false);

  static_assert(is_mgencomplex_v<sycl::marray<complex<long long>, 42>> ==
                false);
  static_assert(is_mgencomplex_v<sycl::marray<complex<long>, 42>> == false);
  static_assert(is_mgencomplex_v<sycl::marray<complex<int>, 42>> == false);
  static_assert(
      is_mgencomplex_v<sycl::marray<complex<unsigned long long>, 42>> == false);
  static_assert(is_mgencomplex_v<sycl::marray<complex<unsigned long>, 42>> ==
                false);
  static_assert(is_mgencomplex_v<sycl::marray<complex<unsigned int>, 42>> ==
                false);
}

// Check is_plus
TEST_CASE("Test is_plus", "[traits]") {
  static_assert(cplex::detail::is_plus_v<sycl::plus<>> == true);

  static_assert(cplex::detail::is_plus_v<sycl::multiplies<>> == false);
  static_assert(cplex::detail::is_plus_v<sycl::bit_and<>> == false);
  static_assert(cplex::detail::is_plus_v<sycl::logical_and<>> == false);
  static_assert(cplex::detail::is_plus_v<sycl::minimum<>> == false);
  static_assert(cplex::detail::is_plus_v<sycl::maximum<>> == false);
}

// Check is_multiplies
TEST_CASE("Test is_multiplies", "[traits]") {
  static_assert(cplex::detail::is_multiplies_v<sycl::multiplies<>> == true);

  static_assert(cplex::detail::is_multiplies_v<sycl::plus<>> == false);
  static_assert(cplex::detail::is_multiplies_v<sycl::bit_and<>> == false);
  static_assert(cplex::detail::is_multiplies_v<sycl::logical_and<>> == false);
  static_assert(cplex::detail::is_multiplies_v<sycl::minimum<>> == false);
  static_assert(cplex::detail::is_multiplies_v<sycl::maximum<>> == false);
}

// Check is_binary_op_supported
TEST_CASE("Test is_binary_op_supported", "[traits]") {
  static_assert(cplex::detail::is_binary_op_supported_v<sycl::plus<>> == true);
  static_assert(cplex::detail::is_binary_op_supported_v<sycl::multiplies<>> ==
                true);

  static_assert(cplex::detail::is_binary_op_supported_v<sycl::bit_and<>> ==
                false);
  static_assert(cplex::detail::is_binary_op_supported_v<sycl::logical_and<>> ==
                false);
  static_assert(cplex::detail::is_binary_op_supported_v<sycl::minimum<>> ==
                false);
  static_assert(cplex::detail::is_binary_op_supported_v<sycl::maximum<>> ==
                false);
}
