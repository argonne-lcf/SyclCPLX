#include "sycl_ext_complex.hpp"
#include <cassert>
#include <complex>
#include <sycl/sycl.hpp>

bool almost_equal(std::complex<double> x, std::complex<double> y, int ulp) {
  return std::abs(x - y) <=
             std::numeric_limits<double>::epsilon() * std::abs(x + y) * ulp ||
         std::abs(x - y) < std::numeric_limits<double>::min();
}

int main() {

  std::cout
      << "Is sycl::ext::cplx::complex<double> trivially_copyable? "
      << std::boolalpha
      << std::is_trivially_copyable<sycl::ext::cplx::complex<double>>::value
      << '\n';

  sycl::queue Q(sycl::gpu_selector_v);
  std::cout << "Running on "
            << Q.get_device().get_info<sycl::info::device::name>() << "\n";

  std::complex<double> i00{0.2, 0.5};
  std::complex<double> i01{0.2, 0.3};
  std::complex<double> cpu_result = std::pow(i00, i01);
  auto *gpu_result =
      sycl::malloc_shared<sycl::ext::cplx::complex<double>>(1, Q);
  Q.single_task([=]() {
     // Using implicit cast from std::complex -> sycl::ext::cplx::complex
     sycl::ext::cplx::complex<double> i11{i01};
     gpu_result[0] = sycl::ext::cplx::pow<double>(i00, i11);
   }).wait();

  std::cout << "cpu_result " << cpu_result << std::endl;
  std::cout << "gpu_result " << gpu_result[0] << std::endl;
  // Using implicit cast from sycl::ext::cplx::complex -> std::complex
  assert(almost_equal(cpu_result, gpu_result[0], 1));
  return 0;
}
