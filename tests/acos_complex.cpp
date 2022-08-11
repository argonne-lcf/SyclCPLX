#include "test_helper.hpp"

template <typename T> struct test_acos {
  bool operator()(sycl::queue &Q, T init_re, T init_im,
                  bool is_error_checking = false) {
    bool pass = true;

    auto std_in = init_std_complex(init_re, init_im);
    sycl::ext::cplx::complex<T> cplx_input{init_re, init_im};

    std::complex<T> std_out{init_re, init_im};
    auto *cplx_out = sycl::malloc_shared<sycl::ext::cplx::complex<T>>(1, Q);

    // Get std::complex output
    if (is_error_checking)
      std_out = std::acos(std_in);

    // Check cplx::complex output from device
    if (is_error_checking) {
      Q.single_task(
          [=]() { cplx_out[0] = sycl::ext::cplx::acos<T>(cplx_input); });
    } else {
      Q.single_task([=]() {
        cplx_out[0] =
            sycl::ext::cplx::cos<T>(sycl::ext::cplx::acos<T>(cplx_input));
      });
    }
    Q.wait();

    pass &= check_results(cplx_out[0], std_out, /*is_device*/ true,
                          /*tol_multiplier*/ 2);

    // Check cplx::complex output from host
    if (is_error_checking)
      cplx_out[0] = sycl::ext::cplx::acos<T>(cplx_input);
    else
      cplx_out[0] =
          sycl::ext::cplx::cos<T>(sycl::ext::cplx::acos<T>(cplx_input));

    pass &= check_results(cplx_out[0], std_out, /*is_device*/ false,
                          /*tol_multiplier*/ 2);

    sycl::free(cplx_out, Q);

    return pass;
  }
};

template <typename T, std::size_t NumElements> struct test_acos_marray {
  bool operator()(sycl::queue &Q, test_marray<T, NumElements> init_re,
                  test_marray<T, NumElements> init_im,
                  bool is_error_checking = false) {
    bool pass = true;

    auto std_in = init_std_complex(init_re.get(), init_im.get());
    sycl::marray<sycl::ext::cplx::complex<T>, NumElements> cplx_input =
        sycl::ext::cplx::make_complex_marray(init_re.get(), init_im.get());

    sycl::marray<std::complex<T>, NumElements> std_out{};
    auto *cplx_out = sycl::malloc_shared<
        sycl::marray<sycl::ext::cplx::complex<T>, NumElements>>(1, Q);

    // Get std::complex output
    if (is_error_checking) {
      for (std::size_t i = 0; i < NumElements; ++i)
        std_out[i] = std::acos(std_in[i]);
    } else {
      // Need to manually copy to handle as for for halfs, std_in is of value
      // type float and std_out is of value type half
      for (std::size_t i = 0; i < NumElements; ++i)
        std_out[i] = std_in[i];
    }

    // Check cplx::complex output from device
    if (is_error_checking) {
      Q.single_task([=]() {
         *cplx_out = sycl::ext::cplx::acos<T>(cplx_input);
       }).wait();
    } else {
      Q.single_task([=]() {
         *cplx_out =
             sycl::ext::cplx::cos<T>(sycl::ext::cplx::acos<T>(cplx_input));
       }).wait();
    }

    pass &= check_results(*cplx_out, std_out, /*is_device*/ true,
                          /*tol_multiplier*/ 2);

    // Check cplx::complex output from host
    if (is_error_checking) {
      *cplx_out = sycl::ext::cplx::acos<T>(cplx_input);
    } else {
      *cplx_out = sycl::ext::cplx::cos<T>(sycl::ext::cplx::acos<T>(cplx_input));
    }

    pass &= check_results(*cplx_out, std_out, /*is_device*/ false,
                          /*tol_multiplier*/ 2);

    sycl::free(cplx_out, Q);

    return pass;
  }
};

int main() {
  sycl::queue Q;

  bool test_passes = true;
  {
    test_passes &= test_valid_types<test_acos>(Q, 4.42, 2.02);

    test_passes &= test_valid_types<test_acos>(Q, INFINITY, 2.02, true);
    test_passes &= test_valid_types<test_acos>(Q, 4.42, INFINITY, true);
    test_passes &= test_valid_types<test_acos>(Q, INFINITY, INFINITY, true);

    test_passes &= test_valid_types<test_acos>(Q, NAN, 2.02, true);
    test_passes &= test_valid_types<test_acos>(Q, 4.42, NAN, true);
    test_passes &= test_valid_types<test_acos>(Q, NAN, NAN, true);

    test_passes &= test_valid_types<test_acos>(Q, NAN, INFINITY, true);
    test_passes &= test_valid_types<test_acos>(Q, INFINITY, NAN, true);
    test_passes &= test_valid_types<test_acos>(Q, NAN, INFINITY, true);
    test_passes &= test_valid_types<test_acos>(Q, INFINITY, NAN, true);
  }

  // marray tests
  {
    // Check output values
    constexpr size_t value_m_size = 4;
    test_marray<double, value_m_size> re = {1.0, 4.42, -3, 4.0};
    test_marray<double, value_m_size> im = {1.0, 2.02, 3.5, -4.0};

    test_passes &= test_valid_types<test_acos_marray, value_m_size>(Q, re, im);

    // Check error codes
    constexpr size_t error_m_size = 10;
    test_marray<double, error_m_size> error_re = {
        2.02, INFINITYd, INFINITYd, 2.02,      NANd,
        NANd, INFINITYd, NANd,      INFINITYd, NANd};
    test_marray<double, error_m_size> error_im = {
        INFINITYd, 4.42,      NANd, 4.42,      NANd,
        NANd,      INFINITYd, NANd, INFINITYd, NANd};

    test_passes &= test_valid_types<test_acos_marray, error_m_size>(
        Q, error_re, error_im, /* is_error_checking */ true);
  }

  if (!test_passes)
    std::cerr << "acos complex test fails\n";

  return !test_passes;
}
