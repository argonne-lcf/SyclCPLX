/**
 * Example using sycl::ext::cplx::complex with oneMKL.
 *
 * Build with:
 *
 *  icpx -fsycl -qmkl -I./include -o hello_mkl hello_mkl.cpp
 */
#include <cassert>
#include <complex>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
// support oneAPI 2022.X and earlier
#include <CL/sycl.hpp>
#else
#error "SYCL header not found"
#endif

#include "sycl_ext_complex.hpp"

#include <oneapi/mkl.hpp>

bool almost_equal(std::complex<double> x, std::complex<double> y, int ulp) {
  return std::abs(x - y) <=
             std::numeric_limits<double>::epsilon() * std::abs(x + y) * ulp ||
         std::abs(x - y) < std::numeric_limits<double>::min();
}

int main() {

  sycl::queue Q(sycl::gpu_selector_v);
  std::cout << "Running on "
            << Q.get_device().get_info<sycl::info::device::name>() << "\n";

  using complex_std_t = std::complex<double>;
  using complex_ext_t = sycl::ext::cplx::complex<double>;

  // For demonstration purposes, host vectors are declared using std::complex.
  // This comes up for example when working with existing code base with host
  // side computations.
  std::vector<complex_std_t> x{{1.0, 1.0}, {1.0, -1.0}};
  std::vector<complex_std_t> y{{0.0, -2.0}, {0.0, 2.0}};
  std::vector<complex_std_t> result{{0.0, 0.0}, {0.0, 0.0}};

  complex_ext_t a = {2.0, 0.0};

  for (int i = 0; i < result.size(); i++) {
    // Note: mixed type operations require conversion
    result[i] = complex_std_t{a} * x[i] + y[i];
  }

  // Device arrays declared as sycl::ext::cplx::complex, to ensure
  // standard device compatible complex operations.
  auto *d_x = sycl::malloc_shared<complex_ext_t>(x.size(), Q);
  auto *d_y = sycl::malloc_shared<complex_ext_t>(x.size(), Q);

  // Note: cast is necessary here because source and test types must
  // be the same, and we used std::complex for the host vectors.
  Q.copy(reinterpret_cast<const complex_ext_t *>(x.data()), d_x, x.size());
  Q.copy(reinterpret_cast<const complex_ext_t *>(y.data()), d_y, y.size());
  Q.copy(reinterpret_cast<const complex_ext_t *>(y.data()), d_y, y.size());
  Q.wait();

  // Note: implicit conversion cannot happen with pointers to complex arrays,
  // so reinterpret cast is necessary. Implicit conversion DOES work when
  // passing values, e.g. for a.
  auto e = oneapi::mkl::blas::axpy(
      Q, x.size(), a, reinterpret_cast<const complex_std_t *>(d_x), 1,
      reinterpret_cast<complex_std_t *>(d_y), 1);
  Q.copy(d_y, reinterpret_cast<complex_ext_t *>(y.data()), y.size(), e).wait();

  for (int i = 0; i < y.size(); i++) {
    std::cout << "result [" << i << "] cpu " << result[i] << " gpu " << y[i]
              << std::endl;
    almost_equal(y[i], result[i], 1);
  }
}
