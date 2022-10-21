#include "test_helper.hpp"

////////////////////////////////////////////////////////////////////////////////
// COMPLEX TESTS
////////////////////////////////////////////////////////////////////////////////

TEMPLATE_TEST_CASE("Test complex acosh", "[acosh]", double, float, sycl::half) {
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
  auto *cplx_out = sycl::malloc_shared<sycl::ext::cplx::complex<T>>(1, Q);

  // Get std::complex output
  if (is_error_checking)
    std_out = std::acosh(std_in);

  // Check cplx::complex output from device
  if (is_type_supported<T>(Q)) {
    if (is_error_checking) {
      Q.single_task(
          [=]() { cplx_out[0] = sycl::ext::cplx::acosh<T>(cplx_input); });
    } else {
      Q.single_task([=]() {
        cplx_out[0] =
            sycl::ext::cplx::cosh<T>(sycl::ext::cplx::acosh<T>(cplx_input));
      });
    }
    Q.wait();
  }

  check_results(cplx_out[0], std_out, /*tol_multiplier*/ 2);

  // Check cplx::complex output from host
  if (is_error_checking)
    cplx_out[0] = sycl::ext::cplx::acosh<T>(cplx_input);
  else
    cplx_out[0] =
        sycl::ext::cplx::cosh<T>(sycl::ext::cplx::acosh<T>(cplx_input));

  check_results(cplx_out[0], std_out, /*tol_multiplier*/ 2);

  sycl::free(cplx_out, Q);
}

////////////////////////////////////////////////////////////////////////////////
// MARRAY<COMPLEX> TESTS'S UTILITIES
////////////////////////////////////////////////////////////////////////////////

template <typename T, std::size_t NumElements>
auto test(sycl::queue &Q, test_marray<T, NumElements> init_re,
          test_marray<T, NumElements> init_im, bool is_error_checking) {
  auto std_in = init_std_complex(init_re.get(), init_im.get());
  sycl::marray<sycl::ext::cplx::complex<T>, NumElements> cplx_input =
      sycl::ext::cplx::make_complex_marray(init_re.get(), init_im.get());

  sycl::marray<std::complex<T>, NumElements> std_out{};
  auto *cplx_out = sycl::malloc_shared<
      sycl::marray<sycl::ext::cplx::complex<T>, NumElements>>(1, Q);

  // Get std::complex output
  if (is_error_checking) {
    for (std::size_t i = 0; i < NumElements; ++i)
      std_out[i] = std::acosh(std_in[i]);
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
         *cplx_out = sycl::ext::cplx::acosh<T>(cplx_input);
       }).wait();
    } else {
      Q.single_task([=]() {
         *cplx_out =
             sycl::ext::cplx::cosh<T>(sycl::ext::cplx::acosh<T>(cplx_input));
       }).wait();
    }
  }

  check_results(*cplx_out, std_out, /*tol_multiplier*/ 4);

  // Check cplx::complex output from host
  if (is_error_checking) {
    *cplx_out = sycl::ext::cplx::acosh<T>(cplx_input);
  } else {
    *cplx_out = sycl::ext::cplx::cosh<T>(sycl::ext::cplx::acosh<T>(cplx_input));
  }

  check_results(*cplx_out, std_out, /*tol_multiplier*/ 2);

  sycl::free(cplx_out, Q);
}

////////////////////////////////////////////////////////////////////////////////
// MARRAY<COMPLEX> TESTS
////////////////////////////////////////////////////////////////////////////////

TEMPLATE_TEST_CASE_SIG("Test marray complex acosh (check error: false)",
                       "[acosh]",
                       ((typename T, std::size_t NumElements), T, NumElements),
                       (double, 4), (float, 4), (sycl::half, 4)) {
  sycl::queue Q;

  // Test cases
  auto init_re = GENERATE(test_marray<T, NumElements>{1.0, 4.42, -3, 4.0});
  auto init_im = GENERATE(test_marray<T, NumElements>{1.0, 2.02, 3.5, -4.0});

  test(Q, init_re, init_im, false);
}

TEMPLATE_TEST_CASE_SIG("Test marray complex acosh (check error: true)",
                       "[acosh]",
                       ((typename T, std::size_t NumElements), T, NumElements),
                       (double, 10), (float, 10), (sycl::half, 10)) {
  sycl::queue Q;

  // Test cases
  auto init_re = GENERATE(test_marray<T, NumElements>{
      2.02, inf_val<T>, inf_val<T>, 2.02, nan_val<T>, nan_val<T>, inf_val<T>,
      nan_val<T>, inf_val<T>, nan_val<T>});
  auto init_im = GENERATE(test_marray<T, NumElements>{
      inf_val<T>, 4.42, nan_val<T>, 4.42, nan_val<T>, nan_val<T>, inf_val<T>,
      nan_val<T>, inf_val<T>, nan_val<T>});

  test(Q, init_re, init_im, true);
}
