#include "test_helper.hpp"

using namespace sycl::ext::cplx;

template <typename T, std::size_t NumElements> struct test_make_complex_marray {

  bool test_make_complex_marray_marray(sycl::queue &Q,
                                       test_marray<T, NumElements> init_re,
                                       test_marray<T, NumElements> init_im) {
    bool pass = true;

    auto *cplx = sycl::malloc_shared<
        sycl::marray<sycl::ext::cplx::complex<T>, NumElements>>(1, Q);

    // Check on device
    Q.single_task([=]() {
       *cplx =
           sycl::ext::cplx::make_complex_marray(init_re.get(), init_im.get());
     }).wait();

    for (std::size_t i = 0; i < NumElements; ++i) {
      pass &= check_results(get_real(*cplx), init_re.get(),
                            /* is_device */ true);
      pass &= check_results(get_imag(*cplx), init_im.get(),
                            /* is_device */ true);
    }

    // Check on host
    *cplx = sycl::ext::cplx::complex<T>(0, 0);
    *cplx = sycl::ext::cplx::make_complex_marray(init_re.get(), init_im.get());

    for (std::size_t i = 0; i < NumElements; ++i) {
      pass &= check_results(get_real(*cplx), init_re.get(),
                            /* is_device */ false);
      pass &= check_results(get_imag(*cplx), init_im.get(),
                            /* is_device */ false);
    }

    sycl::free(cplx, Q);

    return pass;
  }

  bool test_make_complex_marray_deci(sycl::queue &Q,
                                     test_marray<T, NumElements> init_re,
                                     T init_im) {
    bool pass = true;

    auto *cplx = sycl::malloc_shared<
        sycl::marray<sycl::ext::cplx::complex<T>, NumElements>>(1, Q);

    // Check on device
    Q.single_task([=]() {
       *cplx = sycl::ext::cplx::make_complex_marray(init_re.get(), init_im);
     }).wait();

    for (std::size_t i = 0; i < NumElements; ++i) {
      pass &= check_results(get_real(*cplx), init_re.get(),
                            /* is_device */ true);
      pass &= check_results(get_imag(*cplx), init_im,
                            /* is_device */ true);
    }

    // Check on host
    *cplx = sycl::ext::cplx::complex<T>(0, 0);
    *cplx = sycl::ext::cplx::make_complex_marray(init_re.get(), init_im);

    for (std::size_t i = 0; i < NumElements; ++i) {
      pass &= check_results(get_real(*cplx), init_re.get(),
                            /* is_device */ false);
      pass &= check_results(get_imag(*cplx), init_im,
                            /* is_device */ false);
    }

    sycl::free(cplx, Q);

    return pass;
  }

  bool test_make_complex_deci_marray(sycl::queue &Q, T init_re,
                                     test_marray<T, NumElements> init_im) {
    bool pass = true;

    auto *cplx = sycl::malloc_shared<
        sycl::marray<sycl::ext::cplx::complex<T>, NumElements>>(1, Q);

    // Check on device
    Q.single_task([=]() {
       *cplx = sycl::ext::cplx::make_complex_marray(init_re, init_im.get());
     }).wait();

    for (std::size_t i = 0; i < NumElements; ++i) {
      pass &= check_results(get_real(*cplx), init_re,
                            /* is_device */ true);
      pass &= check_results(get_imag(*cplx), init_im.get(),
                            /* is_device */ true);
    }

    // Check on host
    *cplx = sycl::ext::cplx::make_complex_marray(init_re, init_im.get());

    for (std::size_t i = 0; i < NumElements; ++i) {
      pass &= check_results(get_real(*cplx), init_re,
                            /* is_device */ false);
      pass &= check_results(get_imag(*cplx), init_im.get(),
                            /* is_device */ false);
    }

    return pass;
  }

  bool operator()(sycl::queue &Q, test_marray<T, NumElements> init_re,
                  test_marray<T, NumElements> init_im) {
    bool test_passes = true;

    test_passes &= test_make_complex_marray_marray(Q, init_re, init_im);
    test_passes &= test_make_complex_marray_deci(Q, init_re, init_im[0]);
    test_passes &= test_make_complex_deci_marray(Q, init_re[0], init_im);

    return test_passes;
  }
};

template <typename T, std::size_t NumElements>
struct test_get_component_marray {

  bool test_get_component(sycl::queue &Q, test_marray<T, NumElements> init_re,
                          test_marray<T, NumElements> init_im) {
    bool pass = true;

    // Check on device
    auto *cplx = sycl::malloc_shared<
        sycl::marray<sycl::ext::cplx::complex<T>, NumElements>>(1, Q);
    auto *re = sycl::malloc_shared<sycl::marray<T, NumElements>>(1, Q);
    auto *im = sycl::malloc_shared<sycl::marray<T, NumElements>>(1, Q);

    *cplx = sycl::ext::cplx::make_complex_marray(init_re.get(), init_im.get());

    // Check get_X overload that retrieves the entire marray-complex real or
    // imaginary component
    Q.single_task([=]() {
       *re = get_real(*cplx);
       *im = get_imag(*cplx);
     }).wait();

    pass &= check_results(*re, init_re.get(),
                          /* is_device */ true);
    pass &= check_results(*im, init_im.get(),
                          /* is_device */ true);

    // Check get_X overload that retrieves a single marray-complex real or
    // imaginary value
    *re = 0;
    *im = 0;

    Q.parallel_for(sycl::range(NumElements), [=](sycl::id<1> i) {
       (*re)[i] = get_real(*cplx, i);
       (*im)[i] = get_imag(*cplx, i);
     }).wait();

    for (std::size_t i = 0; i < NumElements; ++i) {
      pass &= check_results((*re)[i], init_re[i],
                            /* is_device */ false);
      pass &= check_results((*im)[i], init_im[i],
                            /* is_device */ false);
    }

    // Check on host
    // Check get_X overload that retrieves the entire marray-complex real or
    // imaginary component
    pass &= check_results(get_real(*cplx), init_re.get(),
                          /* is_device */ false);
    pass &= check_results(get_imag(*cplx), init_im.get(),
                          /* is_device */ false);

    // Check get_X overload that retrieves a single marray-complex real or
    // imaginary value
    for (std::size_t i = 0; i < NumElements; ++i) {
      pass &= check_results(get_real(*cplx, i), init_re[i],
                            /* is_device */ false);
      pass &= check_results(get_imag(*cplx, i), init_im[i],
                            /* is_device */ false);
    }

    return pass;
  }

  bool operator()(sycl::queue &Q, test_marray<T, NumElements> init_re,
                  test_marray<T, NumElements> init_im) {
    bool test_passes = true;

    test_passes &= test_get_component(Q, init_re, init_im);

    return test_passes;
  }
};

template <typename T, std::size_t NumElements>
struct test_set_component_marray {

  bool test_set_vec(sycl::queue &Q, test_marray<T, NumElements> init_re,
                    test_marray<T, NumElements> init_im) {
    bool pass = true;

    auto *cplx = sycl::malloc_shared<
        sycl::marray<sycl::ext::cplx::complex<T>, NumElements>>(1, Q);

    // Check on device
    // set the real/imag component with an marray and check it is correct
    Q.single_task([=]() {
       set_real(*cplx, init_re.get());
       set_imag(*cplx, init_im.get());
     }).wait();

    pass &= check_results(get_real(*cplx), init_re.get(),
                          /* is_device */ false);
    pass &= check_results(get_imag(*cplx), init_im.get(),
                          /* is_device */ false);

    Q.single_task([=]() {
       set_real(*cplx, init_re[0]);
       set_imag(*cplx, init_im[0]);
     }).wait();

    pass &= check_results(get_real(*cplx), init_re[0],
                          /* is_device */ false);
    pass &= check_results(get_imag(*cplx), init_im[0],
                          /* is_device */ false);

    // Check on host
    // set the real/imag component with an marray and check it is correct
    set_real(*cplx, init_re.get());
    set_imag(*cplx, init_im.get());

    pass &= check_results(get_real(*cplx), init_re.get(),
                          /* is_device */ false);
    pass &= check_results(get_imag(*cplx), init_im.get(),
                          /* is_device */ false);

    // set the real/imag component with a single value and check it is correct
    set_real(*cplx, init_re[0]);
    set_imag(*cplx, init_im[0]);

    pass &= check_results(get_real(*cplx), init_re[0],
                          /* is_device */ false);
    pass &= check_results(get_imag(*cplx), init_im[0],
                          /* is_device */ false);

    return pass;
  }

  bool test_set_elem(sycl::queue &Q, test_marray<T, NumElements> init_re,
                     test_marray<T, NumElements> init_im) {
    bool pass = true;

    auto *cplx = sycl::malloc_shared<
        sycl::marray<sycl::ext::cplx::complex<T>, NumElements>>(1, Q);

    // Check on device
    Q.parallel_for(sycl::range(NumElements), [=](sycl::id<1> i) {
       set_real(*cplx, i, init_re[i]);
       set_imag(*cplx, i, init_im[i]);
     }).wait();

    for (std::size_t i = 0; i < NumElements; ++i) {
      pass &= check_results((*cplx)[i].real(), init_re[i],
                            /* is_device */ false);
      pass &= check_results((*cplx)[i].imag(), init_im[i],
                            /* is_device */ false);
    }

    // Check on host
    *cplx = sycl::ext::cplx::complex<T>(0, 0);

    for (std::size_t i = 0; i < NumElements; ++i) {
      set_real(*cplx, i, init_re[i]);
      pass &= check_results((*cplx)[i].real(), init_re[i],
                            /* is_device */ false);

      set_imag(*cplx, i, init_im[i]);
      pass &= check_results((*cplx)[i].imag(), init_im[i],
                            /* is_device */ false);
    }

    return pass;
  }

  bool operator()(sycl::queue &Q, test_marray<T, NumElements> init_re,
                  test_marray<T, NumElements> init_im) {
    bool test_passes = true;

    test_passes &= test_set_vec(Q, init_re, init_im);
    test_passes &= test_set_elem(Q, init_re, init_im);

    return test_passes;
  }
};

int main() {
  sycl::queue Q;

  bool test_failed = false;

  {
    bool test_passes = true;

    constexpr size_t m_size = 4;
    test_marray<double, m_size> A = {1, 4.42, -3, 4};
    test_marray<double, m_size> B = {1, 2.02, 3.5, -4};
    test_passes &= test_valid_types<test_make_complex_marray, m_size>(Q, A, B);

    if (!test_passes)
      std::cerr << "make_complex_marray free function test fails\n";
  }

  {
    bool test_passes = true;

    constexpr size_t m_size = 4;
    test_marray<double, m_size> A = {1, 4.42, -3, 4};
    test_marray<double, m_size> B = {1, 2.02, 3.5, -4};
    test_passes &= test_valid_types<test_get_component_marray, m_size>(Q, A, B);

    if (!test_passes)
      std::cerr << "test get_X free function test fails\n";
  }

  {
    bool test_passes = true;

    constexpr size_t m_size = 4;
    test_marray<double, m_size> A = {1, 4.42, -3, 4};
    test_marray<double, m_size> B = {1, 2.02, 3.5, -4};
    test_passes &= test_valid_types<test_set_component_marray, m_size>(Q, A, B);

    if (!test_passes)
      std::cerr << "test set_X free function test fails\n";
  }

  return test_failed;
}
