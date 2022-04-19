#include "sycl_ext_complex.hpp"
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdlib.h>
#include <sycl/sycl.hpp>
using namespace std;
bool almost_equal(float x, float y, int ulp) { return std::abs(x - y) <= std::numeric_limits<float>::epsilon() * std::abs(x + y) * ulp || std::abs(x - y) < std::numeric_limits<float>::min(); }
bool almost_equal(complex<float> x, complex<float> y, int ulp) { return almost_equal(x.real(), y.real(), ulp) && almost_equal(x.imag(), y.imag(), ulp); }
void test_tan() {
  sycl::queue Q(sycl::gpu_selector{});
  const char *usr_precision = getenv("SYCL_TOL_ULP");
  const int precision = usr_precision ? atoi(usr_precision) : 1;
  complex<float> in0_host{0.42, 0.0};
  complex<float> out1_host{};
  auto *out1_device = sycl::malloc_shared<sycl::ext::cplx::complex<float>>(1, Q);
  { out1_host = tan(in0_host); }
  Q.single_task([=]() { out1_device[0] = sycl::ext::cplx::tan<float>(in0_host); }).wait();
  {
    if (!almost_equal(out1_host, out1_device[0], precision)) {
      std::cerr << std::setprecision(std::numeric_limits<float>::max_digits10) << "Host: " << out1_host << " GPU: " << out1_device[0] << std::endl;
      std::exit(112);
    }
  }
}
int main() { test_tan(); }
