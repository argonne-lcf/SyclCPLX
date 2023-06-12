#include <array>
#include <numeric>

#include "test_helper.hpp"

////////////////////////////////////////////////////////////////////////////////
// UTILITY FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

template <typename T, typename BinaryOperation>
void test_joint_exclusive_scan(sycl::queue q, T input,
                               BinaryOperation binary_op) {
  using V = typename T::value_type;

  constexpr size_t N = input.size();

  auto init = sycl::ext::cplx::cplex::detail::get_init<V, BinaryOperation>();

  auto *in = sycl::malloc_shared<V>(N, q);
  auto *output_with_init = sycl::malloc_shared<V>(N, q);
  auto *output_without_init = sycl::malloc_shared<V>(N, q);

  for (std::size_t i = 0; i < N; i++) {
    in[i] = input[i];
  }

  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(N, N), [=](sycl::nd_item<1> it) {
      auto gid = it.get_global_id(0);
      auto lid = it.get_local_id(0);
      auto g = it.get_group();

      sycl::ext::cplx::joint_exclusive_scan(g, in, in + N, output_with_init,
                                            init, binary_op);
      sycl::ext::cplx::joint_exclusive_scan(g, in, in + N, output_without_init,
                                            binary_op);
    });
  });

  q.wait();

  std::array<V, N> expected;
  std::exclusive_scan(input.begin(), input.end(), expected.begin(), init,
                      binary_op);

  std::array<V, N> result;
  for (std::size_t i = 0; i < N; i++) {
    result[i] = output_with_init[i];
  }

  check_results(result, expected);

  for (std::size_t i = 0; i < N; i++) {
    result[i] = output_without_init[i];
  }

  check_results(result, expected);

  sycl::free(in, q);
  sycl::free(output_with_init, q);
  sycl::free(output_without_init, q);
}

////////////////////////////////////////////////////////////////////////////////
// COMPLEX TESTS
////////////////////////////////////////////////////////////////////////////////

TEMPLATE_TEST_CASE_SIG("Test complex joint_exclusive_scan", "[scan]",
                       ((typename T, std::size_t N, typename BinaryOperation),
                        T, N, BinaryOperation),
                       (double, 4, sycl::plus<>), (float, 4, sycl::plus<>),
                       (sycl::half, 4, sycl::plus<>)) {

  using Complex = typename sycl::ext::cplx::complex<T>;
  using Array = typename std::array<Complex, N>;

  sycl::queue q;

  const auto test_cases = GENERATE(
      // Basic value test
      Array{Complex{1, 0}, Complex{2, 0}, Complex{3, 0}, Complex{4, 0}},
      // Random value test
      Array{Complex{0.5, 0.5}, Complex{1.2, 1.2}, Complex{-2.8, -2.8},
            Complex{3.7, 3.7}},
      // Repeated value test
      Array{Complex{1, 1}, Complex{1, 1}, Complex{1, 1}, Complex{1, 1}},
      // Negative value test
      Array{Complex{-3.0, -3.0}, Complex{2.5, 2.5}, Complex{-1.2, -1.2},
            Complex{0, 0}},
      // Large value test
      Array{Complex{1000000.0, 1000000.0}, Complex{2000000.0, 2000000.0},
            Complex{3000000.0, 3000000.0}, Complex{4000000.0, 4000000.0}},
      // Small value test
      Array{Complex{0.0001, 0.0001}, Complex{0.0002, 0.0002},
            Complex{0.0003, 0.0003}, Complex{0.0004, 0.0004}},
      // Edge case value test
      Array{Complex{nan_val<T>, nan_val<T>}, Complex{inf_val<T>, inf_val<T>},
            Complex{nan_val<T>, inf_val<T>}, Complex{inf_val<T>, nan_val<T>}});
  const auto binary_op = BinaryOperation{};

  if (is_type_supported<T>(q)) {
    test_joint_exclusive_scan(q, test_cases, binary_op);
  }
}

////////////////////////////////////////////////////////////////////////////////
// MARRAY<COMPLEX> TESTS
////////////////////////////////////////////////////////////////////////////////

TEMPLATE_TEST_CASE_SIG("Test marray<complex> joint_exclusive_scan", "[scan]",
                       ((typename T, std::size_t N, typename BinaryOperation),
                        T, N, BinaryOperation),
                       (double, 4, sycl::plus<>), (float, 4, sycl::plus<>),
                       (sycl::half, 4, sycl::plus<>)) {

  using Complex = typename sycl::ext::cplx::complex<T>;
  using Marray = typename sycl::marray<Complex, N>;
  using Array = typename std::array<Marray, N>;

  sycl::queue q;

  const auto binary_op = BinaryOperation{};

  const auto test_cases = GENERATE(
      // Basic value test
      Array{Marray{Complex{1, 0}, Complex{1, 0}, Complex{1, 0}, Complex{1, 0}},
            Marray{Complex{2, 0}, Complex{2, 0}, Complex{2, 0}, Complex{2, 0}},
            Marray{Complex{3, 0}, Complex{3, 0}, Complex{3, 0}, Complex{3, 0}},
            Marray{Complex{4, 0}, Complex{4, 0}, Complex{4, 0}, Complex{4, 0}}},
      // Random value test
      Array{Marray{Complex{0.5, 0.5}, Complex{1.2, 1.2}, Complex{-2.8, -2.8},
                   Complex{3.7, 3.7}},
            Marray{Complex{0.5, 0.5}, Complex{1.2, 1.2}, Complex{-2.8, -2.8},
                   Complex{3.7, 3.7}},
            Marray{Complex{0.5, 0.5}, Complex{1.2, 1.2}, Complex{-2.8, -2.8},
                   Complex{3.7, 3.7}},
            Marray{Complex{0.5, 0.5}, Complex{1.2, 1.2}, Complex{-2.8, -2.8},
                   Complex{3.7, 3.7}}},
      // Repeated value test
      Array{Marray{Complex{1, 1}, Complex{1, 1}, Complex{1, 1}, Complex{1, 1}},
            Marray{Complex{1, 1}, Complex{1, 1}, Complex{1, 1}, Complex{1, 1}},
            Marray{Complex{1, 1}, Complex{1, 1}, Complex{1, 1}, Complex{1, 1}},
            Marray{Complex{1, 1}, Complex{1, 1}, Complex{1, 1}, Complex{1, 1}}},
      // Negative value test
      Array{Marray{Complex{-3.0, -3.0}, Complex{2.5, 2.5}, Complex{-1.2, -1.2},
                   Complex{0, 0}},
            Marray{Complex{-3.0, -3.0}, Complex{2.5, 2.5}, Complex{-1.2, -1.2},
                   Complex{0, 0}},
            Marray{Complex{-3.0, -3.0}, Complex{2.5, 2.5}, Complex{-1.2, -1.2},
                   Complex{0, 0}},
            Marray{Complex{-3.0, -3.0}, Complex{2.5, 2.5}, Complex{-1.2, -1.2},
                   Complex{0, 0}}},
      // Large value test
      Array{
          Marray{Complex{1000000.0, 1000000.0}, Complex{2000000.0, 2000000.0},
                 Complex{3000000.0, 3000000.0}, Complex{4000000.0, 4000000.0}},
          Marray{Complex{1000000.0, 1000000.0}, Complex{2000000.0, 2000000.0},
                 Complex{3000000.0, 3000000.0}, Complex{4000000.0, 4000000.0}},
          Marray{Complex{1000000.0, 1000000.0}, Complex{2000000.0, 2000000.0},
                 Complex{3000000.0, 3000000.0}, Complex{4000000.0, 4000000.0}},
          Marray{Complex{1000000.0, 1000000.0}, Complex{2000000.0, 2000000.0},
                 Complex{3000000.0, 3000000.0}, Complex{4000000.0, 4000000.0}}},
      // Small value test
      Array{Marray{Complex{0.0001, 0.0001}, Complex{0.0002, 0.0002},
                   Complex{0.0003, 0.0003}, Complex{0.0004, 0.0004}},
            Marray{Complex{0.0001, 0.0001}, Complex{0.0002, 0.0002},
                   Complex{0.0003, 0.0003}, Complex{0.0004, 0.0004}},
            Marray{Complex{0.0001, 0.0001}, Complex{0.0002, 0.0002},
                   Complex{0.0003, 0.0003}, Complex{0.0004, 0.0004}},
            Marray{Complex{0.0001, 0.0001}, Complex{0.0002, 0.0002},
                   Complex{0.0003, 0.0003}, Complex{0.0004, 0.0004}}},
      // Edge case value test
      Array{
          Marray{
              Complex{nan_val<T>, nan_val<T>}, Complex{inf_val<T>, inf_val<T>},
              Complex{nan_val<T>, inf_val<T>}, Complex{inf_val<T>, nan_val<T>}},
          Marray{
              Complex{nan_val<T>, nan_val<T>}, Complex{inf_val<T>, inf_val<T>},
              Complex{nan_val<T>, inf_val<T>}, Complex{inf_val<T>, nan_val<T>}},
          Marray{
              Complex{nan_val<T>, nan_val<T>}, Complex{inf_val<T>, inf_val<T>},
              Complex{nan_val<T>, inf_val<T>}, Complex{inf_val<T>, nan_val<T>}},
          Marray{Complex{nan_val<T>, nan_val<T>},
                 Complex{inf_val<T>, inf_val<T>},
                 Complex{nan_val<T>, inf_val<T>},
                 Complex{inf_val<T>, nan_val<T>}}});
  if (is_type_supported<T>(q)) {
    test_joint_exclusive_scan(q, test_cases, binary_op);
  }
}
