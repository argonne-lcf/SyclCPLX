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

    sycl::free(cplx, Q);

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

    sycl::free(cplx, Q);
    sycl::free(re, Q);
    sycl::free(im, Q);

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

    sycl::free(cplx, Q);

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

    sycl::free(cplx, Q);

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

template <typename T, std::size_t NumElements, std::size_t... I>
struct test_integer_sequence_overloads {

  bool
  test_make_complex_int_seq(sycl::queue &Q,
                            std::integer_sequence<std::size_t, I...> int_seq,
                            test_marray<T, NumElements> init_re,
                            test_marray<T, NumElements> init_im) {
    bool pass = true;
    int i = 0;

    auto cplx =
        sycl::ext::cplx::make_complex_marray(init_re.get(), init_im.get());
    auto *cplx_dev = sycl::malloc_shared<
        sycl::marray<sycl::ext::cplx::complex<T>, int_seq.size()>>(1, Q);

    // Check on device

    // Check re, im, int_seq overload
    {
      Q.single_task([=]() {
         *cplx_dev = sycl::ext::cplx::make_complex_marray(
             init_re.get(), init_im.get(), int_seq);
       }).wait();

      i = 0;
      ((pass &=
        check_results((*cplx_dev)[i++], std::complex<T>(init_re[I], init_im[I]),
                      /* is_device */ true)),
       ...);
    }

    // Check complex, int_seq overload
    {
      Q.single_task([=]() {
         *cplx_dev = sycl::ext::cplx::make_complex_marray(cplx, int_seq);
       }).wait();

      i = 0;
      ((pass &=
        check_results((*cplx_dev)[i++], std::complex<T>(init_re[I], init_im[I]),
                      /* is_device */ true)),
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
      ((pass &=
        check_results(cplx_host[i++], std::complex<T>(init_re[I], init_im[I]),
                      /* is_device */ true)),
       ...);
    }

    // Check complex, int_seq overload
    {
      auto cplx_host = sycl::ext::cplx::make_complex_marray(cplx, int_seq);

      if (cplx_host.size() != int_seq.size())
        pass = false;

      i = 0;
      ((pass &=
        check_results(cplx_host[i++], std::complex<T>(init_re[I], init_im[I]),
                      /* is_device */ true)),
       ...);
    }

    sycl::free(cplx_dev, Q);

    return pass;
  }

  bool test_set_int_seq(sycl::queue &Q,
                        std::integer_sequence<std::size_t, I...> int_seq,
                        test_marray<T, NumElements> init_re,
                        test_marray<T, NumElements> init_im) {
    bool pass = true;
    int i = 0;

    auto cplx =
        sycl::ext::cplx::make_complex_marray(init_re.get(), init_im.get());
    auto *re_dev = sycl::malloc_shared<sycl::marray<T, int_seq.size()>>(1, Q);
    auto *im_dev = sycl::malloc_shared<sycl::marray<T, int_seq.size()>>(1, Q);

    // Check on device
    Q.single_task([=]() {
       *re_dev = get_real(cplx, int_seq);
       *im_dev = get_imag(cplx, int_seq);
     }).wait();

    i = 0;
    ((pass &= check_results((*re_dev)[i++], init_re[I], /* is_device */ true)),
     ...);
    i = 0;
    ((pass &= check_results((*im_dev)[i++], init_im[I], /* is_device */ true)),
     ...);

    // Check on host (redeclare components to test marray size)
    auto re_host = get_real(cplx, int_seq);
    auto im_host = get_imag(cplx, int_seq);

    if (re_host.size() != int_seq.size())
      pass = false;
    if (im_host.size() != int_seq.size())
      pass = false;

    i = 0;
    ((pass &= check_results(re_host[i++], init_re[I], /* is_device */ false)),
     ...);
    i = 0;
    ((pass &= check_results(im_host[i++], init_im[I], /* is_device */ false)),
     ...);

    sycl::free(re_dev, Q);
    sycl::free(im_dev, Q);

    return pass;
  }

  bool operator()(sycl::queue &Q,
                  std::integer_sequence<std::size_t, I...> int_seq,
                  test_marray<T, NumElements> init_re,
                  test_marray<T, NumElements> init_im) {
    bool test_passes = true;

    test_passes &= test_set_int_seq(Q, int_seq, init_re, init_im);
    test_passes &= test_make_complex_int_seq(Q, int_seq, init_re, init_im);

    return test_passes;
  }
};

int main() {
  sycl::queue Q;

  bool test_failed = false;

  {
    bool test_passes = true;

    constexpr size_t m_size = 14;
    test_marray<double, m_size> re = {
        1.0,  4.42, -3,   4.0,       2.02, INFINITYd, INFINITYd,
        2.02, NANd, NANd, INFINITYd, NANd, INFINITYd, NANd};
    test_marray<double, m_size> im = {
        1.0,  2.02, 3.5,  -4.0,      INFINITYd, 4.42,      NANd,
        4.42, NANd, NANd, INFINITYd, NANd,      INFINITYd, NANd};
    test_passes &=
        test_valid_types<test_make_complex_marray, m_size>(Q, re, im);

    if (!test_passes)
      std::cerr << "make_complex_marray free function test fails\n";
  }

  {
    bool test_passes = true;

    constexpr size_t m_size = 14;
    test_marray<double, m_size> re = {
        1.0,  4.42, -3,   4.0,       2.02, INFINITYd, INFINITYd,
        2.02, NANd, NANd, INFINITYd, NANd, INFINITYd, NANd};
    test_marray<double, m_size> im = {
        1.0,  2.02, 3.5,  -4.0,      INFINITYd, 4.42,      NANd,
        4.42, NANd, NANd, INFINITYd, NANd,      INFINITYd, NANd};
    test_passes &=
        test_valid_types<test_get_component_marray, m_size>(Q, re, im);

    if (!test_passes)
      std::cerr << "get_X free function test fails\n";
  }

  {
    bool test_passes = true;

    constexpr size_t m_size = 14;
    test_marray<double, m_size> re = {
        1.0,  4.42, -3,   4.0,       2.02, INFINITYd, INFINITYd,
        2.02, NANd, NANd, INFINITYd, NANd, INFINITYd, NANd};
    test_marray<double, m_size> im = {
        1.0,  2.02, 3.5,  -4.0,      INFINITYd, 4.42,      NANd,
        4.42, NANd, NANd, INFINITYd, NANd,      INFINITYd, NANd};
    test_passes &=
        test_valid_types<test_set_component_marray, m_size>(Q, re, im);

    if (!test_passes)
      std::cerr << "set_X free function test fails\n";
  }

  {
    bool test_passes = true;

    constexpr size_t m_size = 14;
    test_marray<double, m_size> re = {
        1.0,  4.42, -3,   4.0,       2.02, INFINITYd, INFINITYd,
        2.02, NANd, NANd, INFINITYd, NANd, INFINITYd, NANd};
    test_marray<double, m_size> im = {
        1.0,  2.02, 3.5,  -4.0,      INFINITYd, 4.42,      NANd,
        4.42, NANd, NANd, INFINITYd, NANd,      INFINITYd, NANd};
    std::integer_sequence<std::size_t, 11, 2, 7, 12, 5, 2, 9, 10, 7, 0, 7, 4, 5,
                          3>
        int_seq{};

    test_passes &= test_valid_types<test_integer_sequence_overloads, m_size>(
        Q, int_seq, re, im);

    if (!test_passes)
      std::cerr << "integer sequence overloads test fails\n";
  }

  return test_failed;
}
