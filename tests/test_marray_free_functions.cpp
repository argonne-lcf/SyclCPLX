#include "test_helper.hpp"

using namespace sycl::ext::cplx;

////////////////////////////////////////////////////////////////////////////////
// MARRAY<COMPLEX> TESTS
////////////////////////////////////////////////////////////////////////////////

TEMPLATE_TEST_CASE_SIG("Test marray complex make marray-marray",
                       "[free functions]",
                       ((typename T, std::size_t NumElements), T, NumElements),
                       (double, 14), (float, 14), (sycl::half, 14)) {
  sycl::queue Q;

  // Test cases
  auto init_re = GENERATE(test_marray<T, NumElements>{
      1.0, 4.42, -3, 4.0, 2.02, inf_val<T>, inf_val<T>, 2.02, nan_val<T>,
      nan_val<T>, inf_val<T>, nan_val<T>, inf_val<T>, nan_val<T>});
  auto init_im = GENERATE(test_marray<T, NumElements>{
      1.0, 2.02, 3.5, -4.0, inf_val<T>, 4.42, nan_val<T>, 4.42, nan_val<T>,
      nan_val<T>, inf_val<T>, nan_val<T>, inf_val<T>, nan_val<T>});

  auto *cplx = sycl::malloc_shared<
      sycl::marray<sycl::ext::cplx::complex<T>, NumElements>>(1, Q);

  // Check on device
  if (is_type_supported<T>(Q)) {
    Q.single_task([=]() {
       *cplx =
           sycl::ext::cplx::make_complex_marray(init_re.get(), init_im.get());
     }).wait();
  }

  for (std::size_t i = 0; i < NumElements; ++i) {
    check_results(get_real(*cplx), init_re.get());
    check_results(get_imag(*cplx), init_im.get());
  }

  // Check on host
  *cplx = sycl::ext::cplx::complex<T>(0, 0);
  *cplx = sycl::ext::cplx::make_complex_marray(init_re.get(), init_im.get());

  for (std::size_t i = 0; i < NumElements; ++i) {
    check_results(get_real(*cplx), init_re.get());
    check_results(get_imag(*cplx), init_im.get());
  }

  sycl::free(cplx, Q);
}

TEMPLATE_TEST_CASE_SIG("Test marray complex make marray-deci",
                       "[free functions]",
                       ((typename T, std::size_t NumElements), T, NumElements),
                       (double, 14), (float, 14), (sycl::half, 14)) {
  sycl::queue Q;

  // Test cases
  auto init_re = GENERATE(test_marray<T, NumElements>{
      1.0, 4.42, -3, 4.0, 2.02, inf_val<T>, inf_val<T>, 2.02, nan_val<T>,
      nan_val<T>, inf_val<T>, nan_val<T>, inf_val<T>, nan_val<T>});
  auto init_im = GENERATE(T{1.0});

  auto *cplx = sycl::malloc_shared<
      sycl::marray<sycl::ext::cplx::complex<T>, NumElements>>(1, Q);

  // Check on device
  if (is_type_supported<T>(Q)) {
    Q.single_task([=]() {
       *cplx = sycl::ext::cplx::make_complex_marray(init_re.get(), init_im);
     }).wait();
  }

  for (std::size_t i = 0; i < NumElements; ++i) {
    check_results(get_real(*cplx), init_re.get());
    check_results(get_imag(*cplx), init_im);
  }

  // Check on host
  *cplx = sycl::ext::cplx::complex<T>(0, 0);
  *cplx = sycl::ext::cplx::make_complex_marray(init_re.get(), init_im);

  for (std::size_t i = 0; i < NumElements; ++i) {
    check_results(get_real(*cplx), init_re.get());
    check_results(get_imag(*cplx), init_im);
  }

  sycl::free(cplx, Q);
}

TEMPLATE_TEST_CASE_SIG("Test marray complex make deci-marray",
                       "[free functions]",
                       ((typename T, std::size_t NumElements), T, NumElements),
                       (double, 14), (float, 14), (sycl::half, 14)) {
  sycl::queue Q;

  // Test cases
  auto init_re = GENERATE(T{1.0});
  auto init_im = GENERATE(test_marray<T, NumElements>{
      1.0, 2.02, 3.5, -4.0, inf_val<T>, 4.42, nan_val<T>, 4.42, nan_val<T>,
      nan_val<T>, inf_val<T>, nan_val<T>, inf_val<T>, nan_val<T>});

  auto *cplx = sycl::malloc_shared<
      sycl::marray<sycl::ext::cplx::complex<T>, NumElements>>(1, Q);

  // Check on device
  if (is_type_supported<T>(Q)) {
    Q.single_task([=]() {
       *cplx = sycl::ext::cplx::make_complex_marray(init_re, init_im.get());
     }).wait();
  }

  for (std::size_t i = 0; i < NumElements; ++i) {
    check_results(get_real(*cplx), init_re);
    check_results(get_imag(*cplx), init_im.get());
  }

  // Check on host
  *cplx = sycl::ext::cplx::make_complex_marray(init_re, init_im.get());

  for (std::size_t i = 0; i < NumElements; ++i) {
    check_results(get_real(*cplx), init_re);
    check_results(get_imag(*cplx), init_im.get());
  }

  sycl::free(cplx, Q);
}

TEMPLATE_TEST_CASE_SIG("Test marray complex get component marray",
                       "[free functions]",
                       ((typename T, std::size_t NumElements), T, NumElements),
                       (double, 14), (float, 14), (sycl::half, 14)) {
  sycl::queue Q;

  // Test cases
  auto init_re = GENERATE(test_marray<T, NumElements>{
      1.0, 4.42, -3, 4.0, 2.02, inf_val<T>, inf_val<T>, 2.02, nan_val<T>,
      nan_val<T>, inf_val<T>, nan_val<T>, inf_val<T>, nan_val<T>});
  auto init_im = GENERATE(test_marray<T, NumElements>{
      1.0, 2.02, 3.5, -4.0, inf_val<T>, 4.42, nan_val<T>, 4.42, nan_val<T>,
      nan_val<T>, inf_val<T>, nan_val<T>, inf_val<T>, nan_val<T>});

  // Check on device
  auto *cplx = sycl::malloc_shared<
      sycl::marray<sycl::ext::cplx::complex<T>, NumElements>>(1, Q);
  auto *re = sycl::malloc_shared<sycl::marray<T, NumElements>>(1, Q);
  auto *im = sycl::malloc_shared<sycl::marray<T, NumElements>>(1, Q);

  *cplx = sycl::ext::cplx::make_complex_marray(init_re.get(), init_im.get());

  // Check get_X overload that retrieves the entire marray-complex real or
  // imaginary component
  if (is_type_supported<T>(Q)) {
    Q.single_task([=]() {
       *re = get_real(*cplx);
       *im = get_imag(*cplx);
     }).wait();
  }

  check_results(*re, init_re.get());
  check_results(*im, init_im.get());

  // Check get_X overload that retrieves a single marray-complex real or
  // imaginary value
  *re = 0;
  *im = 0;

  if (is_type_supported<T>(Q)) {
    Q.parallel_for(sycl::range(NumElements), [=](sycl::id<1> i) {
       (*re)[i] = get_real(*cplx, i);
       (*im)[i] = get_imag(*cplx, i);
     }).wait();
  }

  for (std::size_t i = 0; i < NumElements; ++i) {
    check_results((*re)[i], init_re[i]);
    check_results((*im)[i], init_im[i]);
  }

  // Check on host
  // Check get_X overload that retrieves the entire marray-complex real or
  // imaginary component
  check_results(get_real(*cplx), init_re.get());
  check_results(get_imag(*cplx), init_im.get());

  // Check get_X overload that retrieves a single marray-complex real or
  // imaginary value
  for (std::size_t i = 0; i < NumElements; ++i) {
    check_results(get_real(*cplx, i), init_re[i]);
    check_results(get_imag(*cplx, i), init_im[i]);
  }

  sycl::free(cplx, Q);
  sycl::free(re, Q);
  sycl::free(im, Q);
}

TEMPLATE_TEST_CASE_SIG("Test marray complex set component vec",
                       "[free functions]",
                       ((typename T, std::size_t NumElements), T, NumElements),
                       (double, 14), (float, 14), (sycl::half, 14)) {
  sycl::queue Q;

  // Test cases
  auto init_re = GENERATE(test_marray<T, NumElements>{
      1.0, 4.42, -3, 4.0, 2.02, inf_val<T>, inf_val<T>, 2.02, nan_val<T>,
      nan_val<T>, inf_val<T>, nan_val<T>, inf_val<T>, nan_val<T>});
  auto init_im = GENERATE(test_marray<T, NumElements>{
      1.0, 2.02, 3.5, -4.0, inf_val<T>, 4.42, nan_val<T>, 4.42, nan_val<T>,
      nan_val<T>, inf_val<T>, nan_val<T>, inf_val<T>, nan_val<T>});

  auto *cplx = sycl::malloc_shared<
      sycl::marray<sycl::ext::cplx::complex<T>, NumElements>>(1, Q);

  // Check on device
  // set the real/imag component with an marray and check it is correct
  if (is_type_supported<T>(Q)) {
    Q.single_task([=]() {
       set_real(*cplx, init_re.get());
       set_imag(*cplx, init_im.get());
     }).wait();
  }

  check_results(get_real(*cplx), init_re.get());
  check_results(get_imag(*cplx), init_im.get());

  if (is_type_supported<T>(Q)) {
    Q.single_task([=]() {
       set_real(*cplx, init_re[0]);
       set_imag(*cplx, init_im[0]);
     }).wait();
  }

  check_results(get_real(*cplx), init_re[0]);
  check_results(get_imag(*cplx), init_im[0]);

  // Check on host
  // set the real/imag component with an marray and check it is correct
  set_real(*cplx, init_re.get());
  set_imag(*cplx, init_im.get());

  check_results(get_real(*cplx), init_re.get());
  check_results(get_imag(*cplx), init_im.get());

  // set the real/imag component with a single value and check it is correct
  set_real(*cplx, init_re[0]);
  set_imag(*cplx, init_im[0]);

  check_results(get_real(*cplx), init_re[0]);
  check_results(get_imag(*cplx), init_im[0]);

  sycl::free(cplx, Q);
}

TEMPLATE_TEST_CASE_SIG("Test marray complex set component elem",
                       "[free functions]",
                       ((typename T, std::size_t NumElements), T, NumElements),
                       (double, 14), (float, 14), (sycl::half, 14)) {
  sycl::queue Q;

  // Test cases
  auto init_re = GENERATE(test_marray<T, NumElements>{
      1.0, 4.42, -3, 4.0, 2.02, inf_val<T>, inf_val<T>, 2.02, nan_val<T>,
      nan_val<T>, inf_val<T>, nan_val<T>, inf_val<T>, nan_val<T>});
  auto init_im = GENERATE(test_marray<T, NumElements>{
      1.0, 2.02, 3.5, -4.0, inf_val<T>, 4.42, nan_val<T>, 4.42, nan_val<T>,
      nan_val<T>, inf_val<T>, nan_val<T>, inf_val<T>, nan_val<T>});

  auto *cplx = sycl::malloc_shared<
      sycl::marray<sycl::ext::cplx::complex<T>, NumElements>>(1, Q);

  // Check on device
  if (is_type_supported<T>(Q)) {
    Q.parallel_for(sycl::range(NumElements), [=](sycl::id<1> i) {
       set_real(*cplx, i, init_re[i]);
       set_imag(*cplx, i, init_im[i]);
     }).wait();
  }

  for (std::size_t i = 0; i < NumElements; ++i) {
    check_results((*cplx)[i].real(), init_re[i]);
    check_results((*cplx)[i].imag(), init_im[i]);
  }

  // Check on host
  *cplx = sycl::ext::cplx::complex<T>(0, 0);

  for (std::size_t i = 0; i < NumElements; ++i) {
    set_real(*cplx, i, init_re[i]);
    check_results((*cplx)[i].real(), init_re[i]);

    set_imag(*cplx, i, init_im[i]);
    check_results((*cplx)[i].imag(), init_im[i]);
  }

  sycl::free(cplx, Q);
}

TEMPLATE_TEST_CASE_SIG("Test marray complex set integer sequence",
                       "[free functions]",
                       ((typename T, std::size_t NumElements, std::size_t... I),
                        T, NumElements, I...),
                       (double, 14), (float, 14), (sycl::half, 14)) {
  sycl::queue Q;

  // Test cases
  auto init_re = GENERATE(test_marray<T, NumElements>{
      1.0, 4.42, -3, 4.0, 2.02, inf_val<T>, inf_val<T>, 2.02, nan_val<T>,
      nan_val<T>, inf_val<T>, nan_val<T>, inf_val<T>, nan_val<T>});
  auto init_im = GENERATE(test_marray<T, NumElements>{
      1.0, 2.02, 3.5, -4.0, inf_val<T>, 4.42, nan_val<T>, 4.42, nan_val<T>,
      nan_val<T>, inf_val<T>, nan_val<T>, inf_val<T>, nan_val<T>});
  auto int_seq = GENERATE(std::integer_sequence<std::size_t, 11, 2, 7, 12, 5, 2,
                                                9, 10, 7, 0, 7, 4, 5, 3>{});

  bool pass = true;
  int i = 0;

  auto cplx =
      sycl::ext::cplx::make_complex_marray(init_re.get(), init_im.get());
  auto *re_dev = sycl::malloc_shared<sycl::marray<T, int_seq.size()>>(1, Q);
  auto *im_dev = sycl::malloc_shared<sycl::marray<T, int_seq.size()>>(1, Q);

  // Check on device
  if (is_type_supported<T>(Q)) {
    Q.single_task([=]() {
       *re_dev = get_real(cplx, int_seq);
       *im_dev = get_imag(cplx, int_seq);
     }).wait();
  }

  i = 0;
  ((check_results((*re_dev)[i++], init_re[I])), ...);
  i = 0;
  ((check_results((*im_dev)[i++], init_im[I])), ...);

  // Check on host (redeclare components to test marray size)
  auto re_host = get_real(cplx, int_seq);
  auto im_host = get_imag(cplx, int_seq);

  if (re_host.size() != int_seq.size())
    pass = false;
  if (im_host.size() != int_seq.size())
    pass = false;

  i = 0;
  ((check_results(re_host[i++], init_re[I])), ...);
  i = 0;
  ((check_results(im_host[i++], init_im[I])), ...);

  sycl::free(re_dev, Q);
  sycl::free(im_dev, Q);

  CHECK(pass);
}

TEMPLATE_TEST_CASE_SIG("Test marray complex make integer sequence",
                       "[free functions]",
                       ((typename T, std::size_t NumElements, std::size_t... I),
                        T, NumElements, I...),
                       (double, 14), (float, 14), (sycl::half, 14)) {
  sycl::queue Q;

  // Test cases
  auto init_re = GENERATE(test_marray<T, NumElements>{
      1.0, 4.42, -3, 4.0, 2.02, inf_val<T>, inf_val<T>, 2.02, nan_val<T>,
      nan_val<T>, inf_val<T>, nan_val<T>, inf_val<T>, nan_val<T>});
  auto init_im = GENERATE(test_marray<T, NumElements>{
      1.0, 2.02, 3.5, -4.0, inf_val<T>, 4.42, nan_val<T>, 4.42, nan_val<T>,
      nan_val<T>, inf_val<T>, nan_val<T>, inf_val<T>, nan_val<T>});
  auto int_seq = GENERATE(std::integer_sequence<std::size_t, 11, 2, 7, 12, 5, 2,
                                                9, 10, 7, 0, 7, 4, 5, 3>{});

  bool pass = true;
  int i = 0;

  auto cplx =
      sycl::ext::cplx::make_complex_marray(init_re.get(), init_im.get());
  auto *cplx_dev = sycl::malloc_shared<
      sycl::marray<sycl::ext::cplx::complex<T>, int_seq.size()>>(1, Q);

  // Check on device

  // Check re, im, int_seq overload
  {
    if (is_type_supported<T>(Q)) {
      Q.single_task([=]() {
         *cplx_dev = sycl::ext::cplx::make_complex_marray(
             init_re.get(), init_im.get(), int_seq);
       }).wait();
    }

    i = 0;
    ((check_results((*cplx_dev)[i++], std::complex<T>(init_re[I], init_im[I]))),
     ...);
  }

  // Check complex, int_seq overload
  {
    if (is_type_supported<T>(Q)) {
      Q.single_task([=]() {
         *cplx_dev = sycl::ext::cplx::make_complex_marray(cplx, int_seq);
       }).wait();
    }

    i = 0;
    ((check_results((*cplx_dev)[i++], std::complex<T>(init_re[I], init_im[I]))),
     ...);
  }

  // Check on host (redeclare components to test marray size)

  // Check re, im, int_seq overload
  {
    auto cplx_host = sycl::ext::cplx::make_complex_marray(
        init_re.get(), init_im.get(), int_seq);

    if (cplx_host.size() != int_seq.size())
      pass = false;

    i = 0;
    ((check_results(cplx_host[i++], std::complex<T>(init_re[I], init_im[I]))),
     ...);
  }

  // Check complex, int_seq overload
  {
    auto cplx_host = sycl::ext::cplx::make_complex_marray(cplx, int_seq);

    if (cplx_host.size() != int_seq.size())
      pass = false;

    i = 0;
    ((check_results(cplx_host[i++], std::complex<T>(init_re[I], init_im[I]))),
     ...);
  }

  sycl::free(cplx_dev, Q);

  CHECK(pass);
}
