#include "test_helper.hpp"

////////////////////////////////////////////////////////////////////////////////
// COMPLEX TESTS
////////////////////////////////////////////////////////////////////////////////

TEMPLATE_TEST_CASE("Test complex pow cplx-cplx overload", "[pow]", double,
                   float, sycl::half) {
  using T = TestType;

  sycl::queue Q;

  // Test cases
  // Values are generated as cross product of input1 and input2's GENERATE list
  cmplx<T> input1 = GENERATE(
      cmplx<T>{4.42, 2.02}, cmplx<T>{inf_val<T>, 2.02},
      cmplx<T>{4.42, inf_val<T>}, cmplx<T>{inf_val<T>, inf_val<T>},
      cmplx<T>{nan_val<T>, 2.02}, cmplx<T>{4.42, nan_val<T>},
      cmplx<T>{nan_val<T>, nan_val<T>}, cmplx<T>{nan_val<T>, inf_val<T>},
      cmplx<T>{inf_val<T>, nan_val<T>});

  cmplx<T> input2 = GENERATE(
      cmplx<T>{4.42, 2.02}, cmplx<T>{inf_val<T>, 2.02},
      cmplx<T>{4.42, inf_val<T>}, cmplx<T>{inf_val<T>, inf_val<T>},
      cmplx<T>{nan_val<T>, 2.02}, cmplx<T>{4.42, nan_val<T>},
      cmplx<T>{nan_val<T>, nan_val<T>}, cmplx<T>{nan_val<T>, inf_val<T>},
      cmplx<T>{inf_val<T>, nan_val<T>});

  auto std_in1 = init_std_complex(input1);
  auto std_in2 = init_std_complex(input2);
  sycl::ext::cplx::complex<T> cplx_input1{input1.re, input1.im};
  sycl::ext::cplx::complex<T> cplx_input2{input2.re, input2.im};

  std::complex<T> std_out{};
  auto *cplx_out = sycl::malloc_shared<sycl::ext::cplx::complex<T>>(1, Q);

  // Get std::complex output
  std_out = std::pow(std_in1, std_in2);

  // Check cplx::complex output from device
  if (is_type_supported<T>(Q)) {
    Q.single_task([=]() {
       cplx_out[0] = sycl::ext::cplx::pow<T>(cplx_input1, cplx_input2);
     }).wait();

    check_results(cplx_out[0], std_out);
  }

  // Check cplx::complex output from host
  cplx_out[0] = sycl::ext::cplx::pow<T>(cplx_input1, cplx_input2);

  check_results(cplx_out[0], std_out);

  sycl::free(cplx_out, Q);
}

TEMPLATE_TEST_CASE("Test complex pow cplx-deci overload", "[pow]", double,
                   float, sycl::half) {
  using T = TestType;

  sycl::queue Q;

  // Test cases
  // Values are generated as cross product of input1 and input2's GENERATE list
  cmplx<T> input1 = GENERATE(
      cmplx<T>{4.42, 2.02}, cmplx<T>{inf_val<T>, 2.02},
      cmplx<T>{4.42, inf_val<T>}, cmplx<T>{inf_val<T>, inf_val<T>},
      cmplx<T>{nan_val<T>, 2.02}, cmplx<T>{4.42, nan_val<T>},
      cmplx<T>{nan_val<T>, nan_val<T>}, cmplx<T>{nan_val<T>, inf_val<T>},
      cmplx<T>{inf_val<T>, nan_val<T>});
  T input2 = GENERATE(4.42, inf_val<T>, nan_val<T>);

  auto std_in = init_std_complex(input1);
  auto std_deci_in = init_deci(input2);
  sycl::ext::cplx::complex<T> cplx_input{input1.re, input1.im};
  T deci_input = input2;

  std::complex<T> std_out{};
  auto *cplx_out = sycl::malloc_shared<sycl::ext::cplx::complex<T>>(1, Q);

  // Get std::complex output
  std_out = std::pow(std_in, std_deci_in);

  // Check cplx::complex output from device
  if (is_type_supported<T>(Q)) {
    Q.single_task([=]() {
       cplx_out[0] = sycl::ext::cplx::pow<T>(cplx_input, deci_input);
     }).wait();

    check_results(cplx_out[0], std_out);
  }

  // Check cplx::complex output from host
  cplx_out[0] = sycl::ext::cplx::pow<T>(cplx_input, deci_input);

  check_results(cplx_out[0], std_out);

  sycl::free(cplx_out, Q);
}

TEMPLATE_TEST_CASE("Test complex pow deci-cplx overload", "[pow]", double,
                   float, sycl::half) {
  using T = TestType;

  sycl::queue Q;

  // Test cases
  // Values are generated as cross product of input1 and input2's GENERATE list
  cmplx<T> input1 = GENERATE(
      cmplx<T>{4.42, 2.02}, cmplx<T>{inf_val<T>, 2.02},
      cmplx<T>{4.42, inf_val<T>}, cmplx<T>{inf_val<T>, inf_val<T>},
      cmplx<T>{nan_val<T>, 2.02}, cmplx<T>{4.42, nan_val<T>},
      cmplx<T>{nan_val<T>, nan_val<T>}, cmplx<T>{nan_val<T>, inf_val<T>},
      cmplx<T>{inf_val<T>, nan_val<T>});
  T input2 = GENERATE(4.42, inf_val<T>, nan_val<T>);

  auto std_in = init_std_complex(input1);
  auto std_deci_in = init_deci(input2);
  sycl::ext::cplx::complex<T> cplx_input{input1.re, input1.im};
  T deci_input = input2;

  std::complex<T> std_out{};
  auto *cplx_out = sycl::malloc_shared<sycl::ext::cplx::complex<T>>(1, Q);

  // Get std::complex output
  std_out = std::pow(std_deci_in, std_in);

  // Check cplx::complex output from device
  if (is_type_supported<T>(Q)) {
    Q.single_task([=]() {
       cplx_out[0] = sycl::ext::cplx::pow<T>(deci_input, cplx_input);
     }).wait();

    check_results(cplx_out[0], std_out);
  }

  // Check cplx::complex output from host
  cplx_out[0] = sycl::ext::cplx::pow<T>(deci_input, cplx_input);

  check_results(cplx_out[0], std_out);

  sycl::free(cplx_out, Q);
}

////////////////////////////////////////////////////////////////////////////////
// MARRAY<COMPLEX> TESTS
////////////////////////////////////////////////////////////////////////////////

TEMPLATE_TEST_CASE_SIG("Test marray complex pow cplx-cplx overload", "[pow]",
                       ((typename T, std::size_t NumElements), T, NumElements),
                       (double, 14), (float, 14), (sycl::half, 14)) {
  sycl::queue Q;

  // Test cases
  auto init_re1 = GENERATE(test_marray<T, NumElements>{
      1.0, 4.42, -3, 4.0, 2.02, inf_val<T>, inf_val<T>, 2.02, nan_val<T>,
      nan_val<T>, inf_val<T>, nan_val<T>, inf_val<T>, nan_val<T>});
  auto init_im1 = GENERATE(test_marray<T, NumElements>{
      1.0, 2.02, 3.5, -4.0, inf_val<T>, 4.42, nan_val<T>, 4.42, nan_val<T>,
      nan_val<T>, inf_val<T>, nan_val<T>, inf_val<T>, nan_val<T>});
  auto init_re2 = GENERATE(test_marray<T, NumElements>{
      1.0, 4.42, -3, 4.0, 2.02, inf_val<T>, inf_val<T>, 2.02, nan_val<T>,
      nan_val<T>, inf_val<T>, nan_val<T>, inf_val<T>, nan_val<T>});
  auto init_im2 = GENERATE(test_marray<T, NumElements>{
      1.0, 2.02, 3.5, -4.0, inf_val<T>, 4.42, nan_val<T>, 4.42, nan_val<T>,
      nan_val<T>, inf_val<T>, nan_val<T>, inf_val<T>, nan_val<T>});

  auto std_in1 = init_std_complex(init_re1.get(), init_im1.get());
  auto std_in2 = init_std_complex(init_re2.get(), init_im2.get());
  sycl::marray<sycl::ext::cplx::complex<T>, NumElements> cplx_input1 =
      sycl::ext::cplx::make_complex_marray(init_re1.get(), init_im1.get());
  sycl::marray<sycl::ext::cplx::complex<T>, NumElements> cplx_input2 =
      sycl::ext::cplx::make_complex_marray(init_re2.get(), init_im2.get());

  sycl::marray<std::complex<T>, NumElements> std_out{};
  auto *cplx_out = sycl::malloc_shared<
      sycl::marray<sycl::ext::cplx::complex<T>, NumElements>>(1, Q);

  // Get std::complex output
  for (std::size_t i = 0; i < NumElements; ++i)
    std_out[i] = std::pow(std_in1[i], std_in2[i]);

  // Check cplx::complex output from device
  if (is_type_supported<T>(Q)) {
    Q.single_task([=]() {
       *cplx_out = sycl::ext::cplx::pow<T>(cplx_input1, cplx_input2);
     }).wait();
  }

  check_results(*cplx_out, std_out);

  // Check cplx::complex output from host
  *cplx_out = sycl::ext::cplx::pow<T>(cplx_input1, cplx_input2);

  check_results(*cplx_out, std_out);

  sycl::free(cplx_out, Q);
}

TEMPLATE_TEST_CASE_SIG("Test marray complex pow cplx-deci overload", "[pow]",
                       ((typename T, std::size_t NumElements), T, NumElements),
                       (double, 14), (float, 14), (sycl::half, 14)) {
  sycl::queue Q;

  // Test cases
  auto init_re1 = GENERATE(test_marray<T, NumElements>{
      1.0, 4.42, -3, 4.0, 2.02, inf_val<T>, inf_val<T>, 2.02, nan_val<T>,
      nan_val<T>, inf_val<T>, nan_val<T>, inf_val<T>, nan_val<T>});
  auto init_im1 = GENERATE(test_marray<T, NumElements>{
      1.0, 2.02, 3.5, -4.0, inf_val<T>, 4.42, nan_val<T>, 4.42, nan_val<T>,
      nan_val<T>, inf_val<T>, nan_val<T>, inf_val<T>, nan_val<T>});
  auto init_re2 = GENERATE(test_marray<T, NumElements>{
      1.0, 4.42, -3, 4.0, 2.02, inf_val<T>, inf_val<T>, 2.02, nan_val<T>,
      nan_val<T>, inf_val<T>, nan_val<T>, inf_val<T>, nan_val<T>});

  auto std_in1 = init_std_complex(init_re1.get(), init_im1.get());
  auto std_in2 = init_deci(init_re2.get());
  sycl::marray<sycl::ext::cplx::complex<T>, NumElements> cplx_input1 =
      sycl::ext::cplx::make_complex_marray(init_re1.get(), init_im1.get());
  sycl::marray<T, NumElements> cplx_input2 = init_re2.get();

  sycl::marray<std::complex<T>, NumElements> std_out{};
  auto *cplx_out = sycl::malloc_shared<
      sycl::marray<sycl::ext::cplx::complex<T>, NumElements>>(1, Q);

  // Get std::complex output
  for (std::size_t i = 0; i < NumElements; ++i)
    std_out[i] = std::pow(std_in1[i], std_in2[i]);

  // Check cplx::complex output from device
  if (is_type_supported<T>(Q)) {
    Q.single_task([=]() {
       *cplx_out = sycl::ext::cplx::pow<T>(cplx_input1, cplx_input2);
     }).wait();
  }

  check_results(*cplx_out, std_out);

  // Check cplx::complex output from host
  *cplx_out = sycl::ext::cplx::pow<T>(cplx_input1, cplx_input2);

  check_results(*cplx_out, std_out);

  sycl::free(cplx_out, Q);
}

TEMPLATE_TEST_CASE_SIG("Test marray complex pow deci-cplx overload", "[pow]",
                       ((typename T, std::size_t NumElements), T, NumElements),
                       (double, 14), (float, 14), (sycl::half, 14)) {
  sycl::queue Q;

  // Test cases
  auto init_re1 = GENERATE(test_marray<T, NumElements>{
      1.0, 4.42, -3, 4.0, 2.02, inf_val<T>, inf_val<T>, 2.02, nan_val<T>,
      nan_val<T>, inf_val<T>, nan_val<T>, inf_val<T>, nan_val<T>});
  auto init_re2 = GENERATE(test_marray<T, NumElements>{
      1.0, 4.42, -3, 4.0, 2.02, inf_val<T>, inf_val<T>, 2.02, nan_val<T>,
      nan_val<T>, inf_val<T>, nan_val<T>, inf_val<T>, nan_val<T>});
  auto init_im2 = GENERATE(test_marray<T, NumElements>{
      1.0, 2.02, 3.5, -4.0, inf_val<T>, 4.42, nan_val<T>, 4.42, nan_val<T>,
      nan_val<T>, inf_val<T>, nan_val<T>, inf_val<T>, nan_val<T>});

  auto std_in1 = init_deci(init_re1.get());
  auto std_in2 = init_std_complex(init_re2.get(), init_im2.get());
  sycl::marray<T, NumElements> cplx_input1 = init_re1.get();
  sycl::marray<sycl::ext::cplx::complex<T>, NumElements> cplx_input2 =
      sycl::ext::cplx::make_complex_marray(init_re2.get(), init_im2.get());

  sycl::marray<std::complex<T>, NumElements> std_out{};
  auto *cplx_out = sycl::malloc_shared<
      sycl::marray<sycl::ext::cplx::complex<T>, NumElements>>(1, Q);

  // Get std::complex output
  for (std::size_t i = 0; i < NumElements; ++i)
    std_out[i] = std::pow(std_in1[i], std_in2[i]);

  // Check cplx::complex output from device
  if (is_type_supported<T>(Q)) {
    Q.single_task([=]() {
       *cplx_out = sycl::ext::cplx::pow<T>(cplx_input1, cplx_input2);
     }).wait();
  }

  check_results(*cplx_out, std_out);

  // Check cplx::complex output from host
  *cplx_out = sycl::ext::cplx::pow<T>(cplx_input1, cplx_input2);

  check_results(*cplx_out, std_out);

  sycl::free(cplx_out, Q);
}
