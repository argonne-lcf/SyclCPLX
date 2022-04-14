#include <sycl/sycl.hpp>
#include <complex>
#include "sycl_ext_complex.hpp"

bool almost_equal(std::complex<double> x, std::complex<double> y, int ulp) {
   return std::abs(x-y) <= std::numeric_limits<double>::epsilon() * std::abs(x+y) * ulp || std::abs(x-y) < std::numeric_limits<double>::min();
}

int main() {

  sycl::queue Q(sycl::gpu_selector{});
  
  std::cout << "Running on "
            << Q.get_device().get_info<sycl::info::device::name>()
            << "\n";

  std::complex<double> i00{0.5,0.5};
  std::complex<double> i01{0.2,0.3};
  sycl::ext::cplx::complex<double> i10{0.5,0.5};
  sycl::ext::cplx::complex<double> i11{0.2,0.3};
  std::complex<double> cpu_result = std::pow(i00,i01);

  auto* gpu_result = sycl::malloc_shared<sycl::ext::cplx::complex<double>>(1,Q);

  Q.single_task([=]() {
    gpu_result[0]  = sycl::ext::cplx::pow(i10,i11); 
  }).wait();  

  std::cout << "cpu_result " << cpu_result << std::endl;
  std::cout << "gpu_result" << gpu_result[0] << std::endl;
  std::cout << almost_equal(cpu_result, std::complex<double>{gpu_result[0].real(), gpu_result[0].imag()}, 1) << std::endl;
  return 0;
}
