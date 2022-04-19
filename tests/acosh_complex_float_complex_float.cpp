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
void test_acosh() {
  sycl::queue Q(sycl::gpu_selector{});
  const char *usr_precision = getenv("SYCL_TOL_ULP");
  const int precision = usr_precision ? atoi(usr_precision) : 1;
  complex<float> x_host{4.42, 0.0};
  complex<float> o_host{};
  auto *o_device = sycl::malloc_shared<sycl::ext::cplx::complex<float>>(1, Q);
  { o_host = acosh(x_host); }
  Q.single_task([=]() { o_device[0] = sycl::ext::cplx::acosh<float>(x_host); }).wait();
  {
    if (!almost_equal(o_host, o_device[0], precision)) {
      std::cerr << std::setprecision(std::numeric_limits<float>::max_digits10) << "Host: " << o_host << " GPU: " << o_device[0] << std::endl;
      std::exit(112);
    }
  }
}
int main() { test_acosh(); }
