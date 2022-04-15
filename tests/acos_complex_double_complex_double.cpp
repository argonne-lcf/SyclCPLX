#include <complex>
#include <cmath>
#include <iomanip>
#include <stdlib.h>
#include <limits>
#include <iostream>
#include <cstdlib>
#include <sycl/sycl.hpp>
#include "sycl_ext_complex.hpp"
using namespace std;
bool almost_equal(complex<double> x, complex<double> y, int ulp) {
   return std::abs(x-y) <= std::numeric_limits<double>::epsilon() * std::abs(x+y) * ulp || std::abs(x-y) < std::numeric_limits<double>::min();
}
void test_acos(){
   sycl::queue Q(sycl::gpu_selector{});
   const char* usr_precision = getenv("SYCL_TOL_ULP");
   const int precision = usr_precision ? atoi(usr_precision) : 4;
   complex<double> x_host { 4.42, 0.0 };
   sycl::ext::cplx::complex<double> x_device { 4.42, 0.0 };
   auto* o_device = sycl::malloc_shared<sycl::ext::cplx::complex<double>>(1,Q);;
  Q.single_task([=]() {
   o_device[0] = sycl::ext::cplx::acos(x_device);
   }).wait();
   std::complex<double> o_device_std = { o_device[0].real(), o_device[0].imag() };
   {
     if ( !almost_equal(cos(o_device_std), x_host, 2*precision) ) {
          std::cerr << std::setprecision (std::numeric_limits<double>::max_digits10 )
                    << "Expected:" << x_host << " Got: " << cos(o_device_std) << std::endl;
          std::exit(112);
     }
   }
}
int main()
{
    test_acos();
}
