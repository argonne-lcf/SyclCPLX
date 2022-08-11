#include "test_helper.hpp"

template <typename T> struct test_pow {
  bool test_pow_cplx_cplx(sycl::queue &Q, T init_re1, T init_im1, T init_re2,
                          T init_im2) {
    bool pass = true;

    auto std_in1 = init_std_complex(init_re1, init_im1);
    auto std_in2 = init_std_complex(init_re2, init_im2);
    sycl::ext::cplx::complex<T> cplx_input1{init_re1, init_im1};
    sycl::ext::cplx::complex<T> cplx_input2{init_re2, init_im2};

    std::complex<T> std_out{};
    auto *cplx_out = sycl::malloc_shared<sycl::ext::cplx::complex<T>>(1, Q);

    // Get std::complex output
    std_out = std::pow(std_in1, std_in2);

    // Check cplx::complex output from device
    Q.single_task([=]() {
       cplx_out[0] = sycl::ext::cplx::pow<T>(cplx_input1, cplx_input2);
     }).wait();

    pass &= check_results(cplx_out[0], std_out, /*is_device*/ true);

    // Check cplx::complex output from host
    cplx_out[0] = sycl::ext::cplx::pow<T>(cplx_input1, cplx_input2);

    pass &= check_results(cplx_out[0], std_out, /*is_device*/ false);

    sycl::free(cplx_out, Q);

    return pass;
  }

  bool test_pow_cplx_deci(sycl::queue &Q, T init_re1, T init_im1, T init_re2) {
    bool pass = true;

    auto std_in = init_std_complex(init_re1, init_im1);
    auto std_deci_in = init_deci(init_re2);
    sycl::ext::cplx::complex<T> cplx_input{init_re1, init_im1};
    T deci_input = init_re2;

    std::complex<T> std_out{};
    auto *cplx_out = sycl::malloc_shared<sycl::ext::cplx::complex<T>>(1, Q);

    // Get std::complex output
    std_out = std::pow(std_in, std_deci_in);

    // Check cplx::complex output from device
    Q.single_task([=]() {
       cplx_out[0] = sycl::ext::cplx::pow(cplx_input, deci_input);
     }).wait();

    pass &= check_results(cplx_out[0], std_out, /*is_device*/ true);

    // Check cplx::complex output from host
    cplx_out[0] = sycl::ext::cplx::pow(cplx_input, deci_input);

    pass &= check_results(cplx_out[0], std_out, /*is_device*/ false);

    return pass;
  }

  bool test_pow_deci_cplx(sycl::queue &Q, T init_re1, T init_re2, T init_im2) {
    bool pass = true;

    auto std_in = init_std_complex(init_re2, init_im2);
    auto std_deci_in = init_deci(init_re1);
    sycl::ext::cplx::complex<T> cplx_input{init_re2, init_im2};
    T deci_input = init_re1;

    std::complex<T> std_out{};
    auto *cplx_out = sycl::malloc_shared<sycl::ext::cplx::complex<T>>(1, Q);

    // Get std::complex output
    std_out = std::pow(std_deci_in, std_in);

    // Check cplx::complex output from device
    Q.single_task([=]() {
       cplx_out[0] = sycl::ext::cplx::pow(deci_input, cplx_input);
     }).wait();

    pass &= check_results(cplx_out[0], std_out, /*is_device*/ true);

    // Check cplx::complex output from host
    cplx_out[0] = sycl::ext::cplx::pow(deci_input, cplx_input);

    pass &= check_results(cplx_out[0], std_out, /*is_device*/ false);

    return pass;
  }

  bool operator()(sycl::queue &Q, T init_re1, T init_im1, T init_re2,
                  T init_im2) {
    bool pass = true;
    pass &= test_pow_cplx_cplx(Q, init_re1, init_im1, init_re2, init_im2);
    pass &= test_pow_cplx_deci(Q, init_re1, init_im1, init_re2);
    pass &= test_pow_deci_cplx(Q, init_re1, init_re2, init_im2);
    return pass;
  }
};

template <typename T, std::size_t NumElements> struct test_pow_marray {
  bool test_pow_cplx_cplx_marray(sycl::queue &Q,
                                 test_marray<T, NumElements> init_re1,
                                 test_marray<T, NumElements> init_im1,
                                 test_marray<T, NumElements> init_re2,
                                 test_marray<T, NumElements> init_im2) {
    bool pass = true;

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
    Q.single_task([=]() {
       *cplx_out = sycl::ext::cplx::pow<T>(cplx_input1, cplx_input2);
     }).wait();

    pass &= check_results(*cplx_out, std_out, /*is_device*/ true);

    // Check cplx::complex output from host
    *cplx_out = sycl::ext::cplx::pow<T>(cplx_input1, cplx_input2);

    pass &= check_results(*cplx_out, std_out, /*is_device*/ false);

    sycl::free(cplx_out, Q);

    return pass;
  }

  bool test_pow_cplx_deci_marray(sycl::queue &Q,
                                 test_marray<T, NumElements> init_re1,
                                 test_marray<T, NumElements> init_im1,
                                 test_marray<T, NumElements> init_re2) {
    bool pass = true;

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
    Q.single_task([=]() {
       *cplx_out = sycl::ext::cplx::pow<T>(cplx_input1, cplx_input2);
     }).wait();

    pass &= check_results(*cplx_out, std_out, /*is_device*/ true);

    // Check cplx::complex output from host
    *cplx_out = sycl::ext::cplx::pow<T>(cplx_input1, cplx_input2);

    pass &= check_results(*cplx_out, std_out, /*is_device*/ false);

    sycl::free(cplx_out, Q);

    return pass;
  }

  bool test_pow_deci_cplx_marray(sycl::queue &Q,
                                 test_marray<T, NumElements> init_re1,
                                 test_marray<T, NumElements> init_re2,
                                 test_marray<T, NumElements> init_im2) {
    bool pass = true;

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
    Q.single_task([=]() {
       *cplx_out = sycl::ext::cplx::pow<T>(cplx_input1, cplx_input2);
     }).wait();

    pass &= check_results(*cplx_out, std_out, /*is_device*/ true);

    // Check cplx::complex output from host
    *cplx_out = sycl::ext::cplx::pow<T>(cplx_input1, cplx_input2);

    pass &= check_results(*cplx_out, std_out, /*is_device*/ false);

    sycl::free(cplx_out, Q);

    return pass;
  }

  bool operator()(sycl::queue &Q, test_marray<T, NumElements> init_re1,
                  test_marray<T, NumElements> init_im1,
                  test_marray<T, NumElements> init_re2,
                  test_marray<T, NumElements> init_im2) {
    bool pass = true;
    pass &=
        test_pow_cplx_cplx_marray(Q, init_re1, init_im1, init_re2, init_im2);
    pass &= test_pow_cplx_deci_marray(Q, init_re1, init_im1, init_re2);
    pass &= test_pow_deci_cplx_marray(Q, init_re1, init_re2, init_im2);
    return pass;
  }
};

int main() {
  sycl::queue Q;

  bool test_passes = true;
  {
    test_passes &= test_valid_types<test_pow>(Q, 4.42, 2.02, 4.42, 2.02);

    test_passes &=
        test_valid_types<test_pow>(Q, INFINITY, 2.02, INFINITY, 2.02);
    test_passes &=
        test_valid_types<test_pow>(Q, 4.42, INFINITY, 4.42, INFINITY);
    test_passes &=
        test_valid_types<test_pow>(Q, INFINITY, INFINITY, INFINITY, INFINITY);

    test_passes &= test_valid_types<test_pow>(Q, NAN, 2.02, NAN, 2.02);
    test_passes &= test_valid_types<test_pow>(Q, 4.42, NAN, 4.42, NAN);
    test_passes &= test_valid_types<test_pow>(Q, NAN, NAN, NAN, NAN);

    test_passes &= test_valid_types<test_pow>(Q, NAN, INFINITY, NAN, INFINITY);
    test_passes &= test_valid_types<test_pow>(Q, INFINITY, NAN, INFINITY, NAN);
    test_passes &= test_valid_types<test_pow>(Q, NAN, INFINITY, NAN, INFINITY);
    test_passes &= test_valid_types<test_pow>(Q, INFINITY, NAN, INFINITY, NAN);
  }

  // marray tests
  {
    constexpr size_t m_size = 14;
    test_marray<double, m_size> re1 = {
        1.0,  4.42, -3,   4.0,       2.02, INFINITYd, INFINITYd,
        2.02, NANd, NANd, INFINITYd, NANd, INFINITYd, NANd};
    test_marray<double, m_size> im1 = {
        1.0,  2.02, 3.5,  -4.0,      INFINITYd, 4.42,      NANd,
        4.42, NANd, NANd, INFINITYd, NANd,      INFINITYd, NANd};

    test_marray<double, m_size> re2 = {
        1.0,  4.42, -3,   4.0,       2.02, INFINITYd, INFINITYd,
        2.02, NANd, NANd, INFINITYd, NANd, INFINITYd, NANd};
    test_marray<double, m_size> im2 = {
        1.0,  2.02, 3.5,  -4.0,      INFINITYd, 4.42,      NANd,
        4.42, NANd, NANd, INFINITYd, NANd,      INFINITYd, NANd};
    test_passes &=
        test_valid_types<test_pow_marray, m_size>(Q, re1, im1, re2, im2);
  }

  if (!test_passes)
    std::cerr << "pow complex test fails\n";

  return !test_passes;
}
