#include "test_helper.hpp"

using namespace sycl::ext::cplx;

constexpr std::size_t marray_test_size = 10;

// Define math operations tests
#define TEST_MATH_OP_TYPE(op_name, op)                                         \
  template <typename T, std::size_t NumElements>                               \
  struct test##_##op_name##_##types {                                          \
    bool operator()(sycl::queue &Q) {                                          \
      /* Check host code */                                                    \
      static_assert(                                                           \
          std::is_same_v<                                                      \
              sycl::marray<complex<T>, NumElements>,                           \
              decltype(std::declval<sycl::marray<complex<T>, NumElements>>()   \
                           op std::declval<                                    \
                               sycl::marray<complex<T>, NumElements>>())>);    \
                                                                               \
      static_assert(                                                           \
          std::is_same_v<                                                      \
              sycl::marray<complex<T>, NumElements>,                           \
              decltype(std::declval<sycl::marray<complex<T>, NumElements>>()   \
                           op std::declval<sycl::marray<T, NumElements>>())>); \
                                                                               \
      static_assert(                                                           \
          std::is_same_v<sycl::marray<complex<T>, NumElements>,                \
                         decltype(std::declval<sycl::marray<T, NumElements>>() \
                                      op std::declval<sycl::marray<            \
                                          complex<T>, NumElements>>())>);      \
                                                                               \
      /* Check device code */                                                  \
      Q.single_task([=]() {                                                    \
        static_assert(                                                         \
            std::is_same_v<                                                    \
                sycl::marray<complex<T>, NumElements>,                         \
                decltype(std::declval<sycl::marray<complex<T>, NumElements>>() \
                             op std::declval<                                  \
                                 sycl::marray<complex<T>, NumElements>>())>);  \
                                                                               \
        static_assert(                                                         \
            std::is_same_v<                                                    \
                sycl::marray<complex<T>, NumElements>,                         \
                decltype(std::declval<sycl::marray<complex<T>, NumElements>>() \
                             op std::declval<                                  \
                                 sycl::marray<T, NumElements>>())>);           \
                                                                               \
        static_assert(                                                         \
            std::is_same_v<                                                    \
                sycl::marray<complex<T>, NumElements>,                         \
                decltype(std::declval<sycl::marray<T, NumElements>>()          \
                             op std::declval<                                  \
                                 sycl::marray<complex<T>, NumElements>>())>);  \
      });                                                                      \
                                                                               \
      return true;                                                             \
    }                                                                          \
  };

TEST_MATH_OP_TYPE(add, +)
TEST_MATH_OP_TYPE(sub, -)
TEST_MATH_OP_TYPE(mul, *)
TEST_MATH_OP_TYPE(div, /)
#undef TEST_MATH_FUNC_TYPE

// Define math operations tests
#define TEST_LOGIC_OP_TYPE(op_name, op)                                        \
  template <typename T, std::size_t NumElements>                               \
  struct test##_##op_name##_##types {                                          \
    bool operator()(sycl::queue &Q) {                                          \
      /* Check host code */                                                    \
      static_assert(                                                           \
          std::is_same_v<                                                      \
              sycl::marray<bool, NumElements>,                                 \
              decltype(std::declval<sycl::marray<complex<T>, NumElements>>()   \
                           op std::declval<                                    \
                               sycl::marray<complex<T>, NumElements>>())>);    \
                                                                               \
      static_assert(                                                           \
          std::is_same_v<                                                      \
              sycl::marray<bool, NumElements>,                                 \
              decltype(std::declval<sycl::marray<complex<T>, NumElements>>()   \
                           op std::declval<sycl::marray<T, NumElements>>())>); \
                                                                               \
      static_assert(                                                           \
          std::is_same_v<sycl::marray<bool, NumElements>,                      \
                         decltype(std::declval<sycl::marray<T, NumElements>>() \
                                      op std::declval<sycl::marray<            \
                                          complex<T>, NumElements>>())>);      \
                                                                               \
      /* Check device code */                                                  \
      Q.single_task([=]() {                                                    \
        static_assert(                                                         \
            std::is_same_v<                                                    \
                sycl::marray<bool, NumElements>,                               \
                decltype(std::declval<sycl::marray<complex<T>, NumElements>>() \
                             op std::declval<                                  \
                                 sycl::marray<complex<T>, NumElements>>())>);  \
                                                                               \
        static_assert(                                                         \
            std::is_same_v<                                                    \
                sycl::marray<bool, NumElements>,                               \
                decltype(std::declval<sycl::marray<complex<T>, NumElements>>() \
                             op std::declval<                                  \
                                 sycl::marray<T, NumElements>>())>);           \
                                                                               \
        static_assert(                                                         \
            std::is_same_v<                                                    \
                sycl::marray<bool, NumElements>,                               \
                decltype(std::declval<sycl::marray<T, NumElements>>()          \
                             op std::declval<                                  \
                                 sycl::marray<complex<T>, NumElements>>())>);  \
      });                                                                      \
                                                                               \
      return true;                                                             \
    }                                                                          \
  };

TEST_LOGIC_OP_TYPE(eq, ==)
TEST_LOGIC_OP_TYPE(inv_eq, !=)
#undef TEST_LOGIC_OP_TYPE

// Check operations return correct types
void check_math_operator_types(sycl::queue &Q) {
  test_valid_types<test_add_types, marray_test_size>(Q);
  test_valid_types<test_sub_types, marray_test_size>(Q);
  test_valid_types<test_mul_types, marray_test_size>(Q);
  test_valid_types<test_div_types, marray_test_size>(Q);

  test_valid_types<test_eq_types, marray_test_size>(Q);
  test_valid_types<test_inv_eq_types, marray_test_size>(Q);
}

// Define math function tests
#define TEST_MATH_FUNC_TYPE(func)                                              \
  template <typename T, std::size_t NumElements>                               \
  struct test##_##func##_##types {                                             \
    bool operator()(sycl::queue &Q) {                                          \
      /* Check host code */                                                    \
      static_assert(std::is_same_v<                                            \
                    sycl::marray<complex<T>, NumElements>,                     \
                    decltype(func(sycl::marray<complex<T>, NumElements>()))>); \
                                                                               \
      /* Check device code */                                                  \
      Q.single_task([=]() {                                                    \
        static_assert(                                                         \
            std::is_same_v<sycl::marray<complex<T>, NumElements>,              \
                           decltype(func(                                      \
                               sycl::marray<complex<T>, NumElements>()))>);    \
      });                                                                      \
      return true;                                                             \
    }                                                                          \
  };

TEST_MATH_FUNC_TYPE(acos)
TEST_MATH_FUNC_TYPE(asin)
TEST_MATH_FUNC_TYPE(atan)
TEST_MATH_FUNC_TYPE(acosh)
TEST_MATH_FUNC_TYPE(asinh)
TEST_MATH_FUNC_TYPE(atanh)
TEST_MATH_FUNC_TYPE(conj)
TEST_MATH_FUNC_TYPE(cos)
TEST_MATH_FUNC_TYPE(cosh)
TEST_MATH_FUNC_TYPE(exp)
TEST_MATH_FUNC_TYPE(log)
TEST_MATH_FUNC_TYPE(log10)
TEST_MATH_FUNC_TYPE(proj)
TEST_MATH_FUNC_TYPE(sin)
TEST_MATH_FUNC_TYPE(sinh)
TEST_MATH_FUNC_TYPE(sqrt)
TEST_MATH_FUNC_TYPE(tan)
TEST_MATH_FUNC_TYPE(tanh)
#undef TEST_MATH_FUNC_TYPE

template <typename T, std::size_t NumElements> struct test_abs_types {
  bool operator()(sycl::queue &Q) {
    /* Check host code */
    static_assert(
        std::is_same_v<sycl::marray<T, NumElements>,
                       decltype(abs(sycl::marray<complex<T>, NumElements>()))>);

    /* Check device code */
    Q.single_task([=]() {
      static_assert(std::is_same_v<
                    sycl::marray<T, NumElements>,
                    decltype(abs(sycl::marray<complex<T>, NumElements>()))>);
    });

    return true;
  }
};

template <typename T, std::size_t NumElements> struct test_polar_types {
  bool operator()(sycl::queue &Q) {
    /* Check host code */
    static_assert(
        std::is_same_v<sycl::marray<complex<T>, NumElements>,
                       decltype(polar(sycl::marray<T, NumElements>()))>);
    static_assert(
        std::is_same_v<sycl::marray<complex<T>, NumElements>,
                       decltype(polar(sycl::marray<T, NumElements>(),
                                      sycl::marray<T, NumElements>()))>);
    static_assert(
        std::is_same_v<sycl::marray<complex<T>, NumElements>,
                       decltype(polar(sycl::marray<T, NumElements>(), T()))>);
    static_assert(
        std::is_same_v<sycl::marray<complex<T>, NumElements>,
                       decltype(polar(T(), sycl::marray<T, NumElements>()))>);

    /* Check device code */
    Q.single_task([=]() {
      static_assert(
          std::is_same_v<sycl::marray<complex<T>, NumElements>,
                         decltype(polar(sycl::marray<T, NumElements>()))>);
      static_assert(
          std::is_same_v<sycl::marray<complex<T>, NumElements>,
                         decltype(polar(sycl::marray<T, NumElements>(),
                                        sycl::marray<T, NumElements>()))>);
      static_assert(
          std::is_same_v<sycl::marray<complex<T>, NumElements>,
                         decltype(polar(sycl::marray<T, NumElements>(), T()))>);
      static_assert(
          std::is_same_v<sycl::marray<complex<T>, NumElements>,
                         decltype(polar(T(), sycl::marray<T, NumElements>()))>);
    });

    return true;
  }
};

template <typename T, std::size_t NumElements> struct test_pow_types {
  bool operator()(sycl::queue &Q) {
    /* Check host code */
    // complex-deci
    static_assert(
        std::is_same_v<sycl::marray<complex<T>, NumElements>,
                       decltype(pow(sycl::marray<complex<T>, NumElements>(),
                                    sycl::marray<T, NumElements>()))>);
    static_assert(std::is_same_v<
                  sycl::marray<complex<T>, NumElements>,
                  decltype(pow(sycl::marray<complex<T>, NumElements>(), T()))>);
    static_assert(std::is_same_v<
                  sycl::marray<complex<T>, NumElements>,
                  decltype(pow(complex<T>(), sycl::marray<T, NumElements>()))>);

    // complex-complex
    static_assert(
        std::is_same_v<sycl::marray<complex<T>, NumElements>,
                       decltype(pow(sycl::marray<complex<T>, NumElements>(),
                                    sycl::marray<complex<T>, NumElements>()))>);
    static_assert(
        std::is_same_v<sycl::marray<complex<T>, NumElements>,
                       decltype(pow(sycl::marray<complex<T>, NumElements>(),
                                    complex<T>()))>);
    static_assert(
        std::is_same_v<sycl::marray<complex<T>, NumElements>,
                       decltype(pow(complex<T>(),
                                    sycl::marray<complex<T>, NumElements>()))>);

    // deci-complx
    static_assert(
        std::is_same_v<sycl::marray<complex<T>, NumElements>,
                       decltype(pow(sycl::marray<T, NumElements>(),
                                    sycl::marray<complex<T>, NumElements>()))>);
    static_assert(std::is_same_v<sycl::marray<complex<T>, NumElements>,
                                 decltype(pow(sycl::marray<T, NumElements>(),
                                              complex<T>()))>);
    static_assert(std::is_same_v<
                  sycl::marray<complex<T>, NumElements>,
                  decltype(pow(T(), sycl::marray<complex<T>, NumElements>()))>);

    /* Check device code */
    Q.single_task([=]() {
      // complex-deci
      static_assert(
          std::is_same_v<sycl::marray<complex<T>, NumElements>,
                         decltype(pow(sycl::marray<complex<T>, NumElements>(),
                                      sycl::marray<T, NumElements>()))>);
      static_assert(
          std::is_same_v<sycl::marray<complex<T>, NumElements>,
                         decltype(pow(sycl::marray<complex<T>, NumElements>(),
                                      T()))>);
      static_assert(
          std::is_same_v<sycl::marray<complex<T>, NumElements>,
                         decltype(pow(complex<T>(),
                                      sycl::marray<T, NumElements>()))>);

      // complex-complex
      static_assert(std::is_same_v<
                    sycl::marray<complex<T>, NumElements>,
                    decltype(pow(sycl::marray<complex<T>, NumElements>(),
                                 sycl::marray<complex<T>, NumElements>()))>);
      static_assert(
          std::is_same_v<sycl::marray<complex<T>, NumElements>,
                         decltype(pow(sycl::marray<complex<T>, NumElements>(),
                                      complex<T>()))>);
      static_assert(std::is_same_v<
                    sycl::marray<complex<T>, NumElements>,
                    decltype(pow(complex<T>(),
                                 sycl::marray<complex<T>, NumElements>()))>);

      // deci-complx
      static_assert(std::is_same_v<
                    sycl::marray<complex<T>, NumElements>,
                    decltype(pow(sycl::marray<T, NumElements>(),
                                 sycl::marray<complex<T>, NumElements>()))>);
      static_assert(std::is_same_v<sycl::marray<complex<T>, NumElements>,
                                   decltype(pow(sycl::marray<T, NumElements>(),
                                                complex<T>()))>);
      static_assert(
          std::is_same_v<sycl::marray<complex<T>, NumElements>,
                         decltype(pow(
                             T(), sycl::marray<complex<T>, NumElements>()))>);
    });

    return true;
  }
};

// Check functions return correct types
void check_math_function_types(sycl::queue &Q) {

  test_valid_types<test_abs_types, marray_test_size>(Q);
  test_valid_types<test_acos_types, marray_test_size>(Q);
  test_valid_types<test_asin_types, marray_test_size>(Q);
  test_valid_types<test_atan_types, marray_test_size>(Q);
  test_valid_types<test_acosh_types, marray_test_size>(Q);
  test_valid_types<test_asinh_types, marray_test_size>(Q);
  test_valid_types<test_atanh_types, marray_test_size>(Q);
  test_valid_types<test_conj_types, marray_test_size>(Q);
  test_valid_types<test_cos_types, marray_test_size>(Q);
  test_valid_types<test_cosh_types, marray_test_size>(Q);
  test_valid_types<test_exp_types, marray_test_size>(Q);
  test_valid_types<test_log_types, marray_test_size>(Q);
  test_valid_types<test_log10_types, marray_test_size>(Q);
  test_valid_types<test_polar_types, marray_test_size>(Q);
  test_valid_types<test_pow_types, marray_test_size>(Q);
  test_valid_types<test_proj_types, marray_test_size>(Q);
  test_valid_types<test_sin_types, marray_test_size>(Q);
  test_valid_types<test_sinh_types, marray_test_size>(Q);
  test_valid_types<test_sqrt_types, marray_test_size>(Q);
  test_valid_types<test_tan_types, marray_test_size>(Q);
  test_valid_types<test_tanh_types, marray_test_size>(Q);
}

int main() {
  sycl::queue Q;

  check_math_function_types(Q);
  check_math_operator_types(Q);

  return 0;
}
