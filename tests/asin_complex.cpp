#include "test_helper.hpp"

////////////////////////////////////////////////////////////////////////////////
// COMPLEX TESTS
////////////////////////////////////////////////////////////////////////////////

TEMPLATE_TEST_CASE("Test complex asin", "[asin]", double, float, sycl::half) {
  using T = TestType;
  using std::make_tuple;

  sycl::queue Q;

  cmplx<T> input;
  bool is_error_checking;

  std::tie(input, is_error_checking) = GENERATE(table<cmplx<T>, bool>(
      {make_tuple(cmplx<T>{4.42, 2.02}, false),
       make_tuple(cmplx<T>{inf_val<T>, 2.02}, true),
       make_tuple(cmplx<T>{4.42, inf_val<T>}, true),
       make_tuple(cmplx<T>{inf_val<T>, inf_val<T>}, true),
       make_tuple(cmplx<T>{nan_val<T>, 2.02}, true),
       make_tuple(cmplx<T>{4.42, nan_val<T>}, true),
       make_tuple(cmplx<T>{nan_val<T>, nan_val<T>}, true),
       make_tuple(cmplx<T>{nan_val<T>, inf_val<T>}, true),
       make_tuple(cmplx<T>{inf_val<T>, nan_val<T>}, true)}));

  auto std_in = init_std_complex(input);
  sycl::ext::cplx::complex<T> cplx_input{input.re, input.im};

  std::complex<T> std_out{input.re, input.im};
  sycl::ext::cplx::complex<T> h_cplx_out;
  auto d_cplx_out = sycl::malloc_device<sycl::ext::cplx::complex<T>>(1, Q);

  // Get std::complex output
  if (is_error_checking)
    std_out = std::asin(std_in);

  // Check cplx::complex output from device
  if (is_type_supported<T>(Q)) {
    if (is_error_checking) {
      Q.single_task([=]() {
         d_cplx_out[0] = sycl::ext::cplx::asin<T>(cplx_input);
       }).wait();
    } else {
      Q.single_task([=]() {
         d_cplx_out[0] =
             sycl::ext::cplx::sin<T>(sycl::ext::cplx::asin<T>(cplx_input));
       }).wait();
    }
    Q.copy(d_cplx_out, &h_cplx_out, 1).wait();

    check_results(h_cplx_out, std_out, /*tol_multiplier*/ 6);
  }

  // Check cplx::complex output from host
  if (is_error_checking)
    h_cplx_out = sycl::ext::cplx::asin<T>(cplx_input);
  else
    h_cplx_out = sycl::ext::cplx::sin<T>(sycl::ext::cplx::asin<T>(cplx_input));

  check_results(h_cplx_out, std_out, /*tol_multiplier*/ 6);

  sycl::free(d_cplx_out, Q);
}

////////////////////////////////////////////////////////////////////////////////
// MARRAY<COMPLEX> TESTS'S UTILITIES
////////////////////////////////////////////////////////////////////////////////

template <typename T, typename X, std::size_t NumElements>
auto test(
    sycl::queue &Q, const sycl::marray<std::complex<X>, NumElements> &std_in,
    const sycl::marray<sycl::ext::cplx::complex<T>, NumElements> &cplx_input,
    bool is_error_checking) {
  sycl::marray<std::complex<T>, NumElements> std_out{};
  sycl::marray<sycl::ext::cplx::complex<T>, NumElements> h_cplx_out;
  auto d_cplx_out = sycl::malloc_device<
      sycl::marray<sycl::ext::cplx::complex<T>, NumElements>>(1, Q);

  // Get std::complex output
  if (is_error_checking) {
    for (std::size_t i = 0; i < NumElements; ++i)
      std_out[i] = std::asin(std_in[i]);
  } else {
    // Need to manually copy to handle as for for halfs, std_in is of value
    // type float and std_out is of value type half
    for (std::size_t i = 0; i < NumElements; ++i)
      std_out[i] = std_in[i];
  }

  // Check cplx::complex output from device
  if (is_type_supported<T>(Q)) {
    if (is_error_checking) {
      Q.single_task([=]() {
         d_cplx_out[0] = sycl::ext::cplx::asin<T>(cplx_input);
       }).wait();
    } else {
      Q.single_task([=]() {
         d_cplx_out[0] =
             sycl::ext::cplx::sin<T>(sycl::ext::cplx::asin<T>(cplx_input));
       }).wait();
    }
    Q.copy(d_cplx_out, &h_cplx_out, 1).wait();

    check_results(h_cplx_out, std_out, /*tol_multiplier*/ 6);
  }

  // Check cplx::complex output from host
  if (is_error_checking) {
    h_cplx_out = sycl::ext::cplx::asin<T>(cplx_input);
  } else {
    h_cplx_out = sycl::ext::cplx::sin<T>(sycl::ext::cplx::asin<T>(cplx_input));
  }

  check_results(h_cplx_out, std_out, /*tol_multiplier*/ 6);

  sycl::free(d_cplx_out, Q);
}

////////////////////////////////////////////////////////////////////////////////
// MARRAY<COMPLEX> TESTS
////////////////////////////////////////////////////////////////////////////////

TEMPLATE_TEST_CASE_SIG("Test marray complex asin (check error: false)",
                       "[asin]",
                       ((typename T, std::size_t NumElements), T, NumElements),
                       (double, 4), (float, 4), (sycl::half, 4)) {
  sycl::queue Q;

  // std::complex test cases
  const auto std_in =
      GENERATE(init_std_complex(sycl::marray<std::complex<T>, NumElements>{
          std::complex<T>{1.0, 1.0},
          std::complex<T>{4.42, 2.02},
          std::complex<T>{-3, 2.5},
          std::complex<T>{4.0, -4.0},
      }));

  // sycl::complex test cases
  sycl::marray<sycl::ext::cplx::complex<T>, NumElements> cplx_input;
  for (std::size_t i = 0; i < NumElements; ++i) {
    cplx_input[i] =
        sycl::ext::cplx::complex<T>{std_in[i].real(), std_in[i].imag()};
  }

  // sycl::half cases are emulated with float for std::complex class (std_in)
  using X = typename std::conditional<std::is_same<T, sycl::half>::value, float,
                                      T>::type;
  test<T, X>(Q, std_in, cplx_input, false);
}

TEMPLATE_TEST_CASE_SIG("Test marray complex asin (check error: true)", "[asin]",
                       ((typename T, std::size_t NumElements), T, NumElements),
                       (double, 10), (float, 10), (sycl::half, 10)) {
  sycl::queue Q;

  // std::complex test cases
  const auto std_in =
      GENERATE(init_std_complex(sycl::marray<std::complex<T>, NumElements>{
          std::complex<T>{2.02, inf_val<T>},
          std::complex<T>{inf_val<T>, 4.42},
          std::complex<T>{inf_val<T>, nan_val<T>},
          std::complex<T>{2.02, 4.42},
          std::complex<T>{nan_val<T>, nan_val<T>},
          std::complex<T>{nan_val<T>, nan_val<T>},
          std::complex<T>{inf_val<T>, inf_val<T>},
          std::complex<T>{nan_val<T>, nan_val<T>},
          std::complex<T>{inf_val<T>, inf_val<T>},
          std::complex<T>{nan_val<T>, nan_val<T>},
      }));

  // sycl::complex test cases
  sycl::marray<sycl::ext::cplx::complex<T>, NumElements> cplx_input;
  for (std::size_t i = 0; i < NumElements; ++i) {
    cplx_input[i] =
        sycl::ext::cplx::complex<T>{std_in[i].real(), std_in[i].imag()};
  }

  // sycl::half cases are emulated with float for std::complex class (std_in)
  using X = typename std::conditional<std::is_same<T, sycl::half>::value, float,
                                      T>::type;
  test<T, X>(Q, std_in, cplx_input, true);
}
