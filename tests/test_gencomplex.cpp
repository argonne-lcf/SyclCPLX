#include "sycl_ext_complex.hpp"
#include <sycl/sycl.hpp>

using namespace sycl::ext::cplx;

// Check is_gencomplex
void check_is_gencomplex() {
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

// Check is_mgencomplex
template <std::size_t NumElements> void check_is_mgencomplex() {
  static_assert(is_mgencomplex<sycl::marray<complex<double>, NumElements>,
                               NumElements>::value == true);
  static_assert(is_mgencomplex<sycl::marray<complex<float>, NumElements>,
                               NumElements>::value == true);
  static_assert(is_mgencomplex<sycl::marray<complex<sycl::half>, NumElements>,
                               NumElements>::value == true);

  static_assert(is_mgencomplex<sycl::marray<complex<long long>, NumElements>,
                               NumElements>::value == false);
  static_assert(is_mgencomplex<sycl::marray<complex<long>, NumElements>,
                               NumElements>::value == false);
  static_assert(is_mgencomplex<sycl::marray<complex<int>, NumElements>,
                               NumElements>::value == false);
  static_assert(
      is_mgencomplex<sycl::marray<complex<unsigned long long>, NumElements>,
                     NumElements>::value == false);
  static_assert(
      is_mgencomplex<sycl::marray<complex<unsigned long>, NumElements>,
                     NumElements>::value == false);
  static_assert(is_mgencomplex<sycl::marray<complex<unsigned int>, NumElements>,
                               NumElements>::value == false);
}

// Check is_mgencomplex
template <typename T, T... ints>
void check_is_mgencomplex_for_sizes(std::integer_sequence<T, ints...> int_seq) {
  ((check_is_mgencomplex<ints>()), ...);
}

int main() {
  check_is_gencomplex();

  check_is_mgencomplex_for_sizes(
      std::integer_sequence<int, 5, 20, 31, 64, 100, 500>{});
}
