#include "test_helper.hpp"

using namespace sycl::ext::cplx;

////////////////////////////////////////////////////////////////////////////////
// MARRAY<COMPLEX> TESTS'S UTILITIES
////////////////////////////////////////////////////////////////////////////////

// Define math function tests
#define TEST_MATH_FUNC_TYPE(func)                                              \
  template <typename T, std::size_t NumElements>                               \
  auto test##_##func##_##types(sycl::queue &Q) {                               \
    /* Check host code */                                                      \
    static_assert(std::is_same_v<                                              \
                  sycl::marray<complex<T>, NumElements>,                       \
                  decltype(func(sycl::marray<complex<T>, NumElements>()))>);   \
                                                                               \
    /* Check device code */                                                    \
    if (is_type_supported<T>(Q)) {                                             \
      Q.single_task([=]() {                                                    \
         static_assert(                                                        \
             std::is_same_v<sycl::marray<complex<T>, NumElements>,             \
                            decltype(func(                                     \
                                sycl::marray<complex<T>, NumElements>()))>);   \
       }).wait();                                                              \
    }                                                                          \
  }

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

// Define math operations tests
#define TEST_MATH_OP_TYPE(op_name, op)                                         \
  template <typename T, std::size_t NumElements>                               \
  auto test##_##op_name##_##types(sycl::queue &Q) {                            \
    /* Check host code */                                                      \
    static_assert(                                                             \
        std::is_same_v<                                                        \
            sycl::marray<complex<T>, NumElements>,                             \
            decltype(                                                          \
                std::declval<sycl::marray<complex<T>, NumElements>>() op       \
                    std::declval<sycl::marray<complex<T>, NumElements>>())>);  \
                                                                               \
    static_assert(                                                             \
        std::is_same_v<                                                        \
            sycl::marray<complex<T>, NumElements>,                             \
            decltype(std::declval<sycl::marray<complex<T>, NumElements>>()     \
                         op std::declval<sycl::marray<T, NumElements>>())>);   \
                                                                               \
    static_assert(                                                             \
        std::is_same_v<                                                        \
            sycl::marray<complex<T>, NumElements>,                             \
            decltype(std::declval<sycl::marray<T, NumElements>>() op std::     \
                         declval<sycl::marray<complex<T>, NumElements>>())>);  \
                                                                               \
    /* Check device code */                                                    \
    if (is_type_supported<T>(Q)) {                                             \
      Q.single_task([=]() {                                                    \
         static_assert(                                                        \
             std::is_same_v<                                                   \
                 sycl::marray<complex<T>, NumElements>,                        \
                 decltype(                                                     \
                     std::declval<sycl::marray<complex<T>, NumElements>>()     \
                         op std::declval<                                      \
                             sycl::marray<complex<T>, NumElements>>())>);      \
                                                                               \
         static_assert(                                                        \
             std::is_same_v<                                                   \
                 sycl::marray<complex<T>, NumElements>,                        \
                 decltype(                                                     \
                     std::declval<sycl::marray<complex<T>, NumElements>>()     \
                         op std::declval<sycl::marray<T, NumElements>>())>);   \
                                                                               \
         static_assert(                                                        \
             std::is_same_v<                                                   \
                 sycl::marray<complex<T>, NumElements>,                        \
                 decltype(std::declval<sycl::marray<T, NumElements>>()         \
                              op std::declval<                                 \
                                  sycl::marray<complex<T>, NumElements>>())>); \
       }).wait();                                                              \
    }                                                                          \
  }

TEST_MATH_OP_TYPE(add, +)
TEST_MATH_OP_TYPE(sub, -)
TEST_MATH_OP_TYPE(mul, *)
TEST_MATH_OP_TYPE(div, /)
#undef TEST_MATH_FUNC_TYPE

// Define math operations tests
#define TEST_LOGIC_OP_TYPE(op_name, op)                                        \
  template <typename T, std::size_t NumElements>                               \
  auto test##_##op_name##_##types(sycl::queue &Q) {                            \
    /* Check host code */                                                      \
    static_assert(                                                             \
        std::is_same_v<                                                        \
            sycl::marray<bool, NumElements>,                                   \
            decltype(                                                          \
                std::declval<sycl::marray<complex<T>, NumElements>>() op       \
                    std::declval<sycl::marray<complex<T>, NumElements>>())>);  \
                                                                               \
    static_assert(                                                             \
        std::is_same_v<                                                        \
            sycl::marray<bool, NumElements>,                                   \
            decltype(std::declval<sycl::marray<complex<T>, NumElements>>()     \
                         op std::declval<sycl::marray<T, NumElements>>())>);   \
                                                                               \
    static_assert(                                                             \
        std::is_same_v<                                                        \
            sycl::marray<bool, NumElements>,                                   \
            decltype(std::declval<sycl::marray<T, NumElements>>() op std::     \
                         declval<sycl::marray<complex<T>, NumElements>>())>);  \
                                                                               \
    /* Check device code */                                                    \
    if (is_type_supported<T>(Q)) {                                             \
      Q.single_task([=]() {                                                    \
         static_assert(                                                        \
             std::is_same_v<                                                   \
                 sycl::marray<bool, NumElements>,                              \
                 decltype(                                                     \
                     std::declval<sycl::marray<complex<T>, NumElements>>()     \
                         op std::declval<                                      \
                             sycl::marray<complex<T>, NumElements>>())>);      \
                                                                               \
         static_assert(                                                        \
             std::is_same_v<                                                   \
                 sycl::marray<bool, NumElements>,                              \
                 decltype(                                                     \
                     std::declval<sycl::marray<complex<T>, NumElements>>()     \
                         op std::declval<sycl::marray<T, NumElements>>())>);   \
                                                                               \
         static_assert(                                                        \
             std::is_same_v<                                                   \
                 sycl::marray<bool, NumElements>,                              \
                 decltype(std::declval<sycl::marray<T, NumElements>>()         \
                              op std::declval<                                 \
                                  sycl::marray<complex<T>, NumElements>>())>); \
       }).wait();                                                              \
    }                                                                          \
  }

TEST_LOGIC_OP_TYPE(eq, ==)
TEST_LOGIC_OP_TYPE(inv_eq, !=)
#undef TEST_LOGIC_OP_TYPE

////////////////////////////////////////////////////////////////////////////////
// MARRAY<COMPLEX> TESTS
////////////////////////////////////////////////////////////////////////////////

TEMPLATE_TEST_CASE_SIG("Test marray complex abs function return types",
                       "[marray types]",
                       ((typename T, std::size_t NumElements), T, NumElements),
                       (double, 10), (float, 10), (sycl::half, 10)) {
  sycl::queue Q;

  static_assert(
      std::is_same_v<sycl::marray<T, NumElements>,
                     decltype(abs(sycl::marray<complex<T>, NumElements>()))>);

  /* Check device code */
  if (is_type_supported<T>(Q)) {
    Q.single_task([=]() {
       static_assert(std::is_same_v<
                     sycl::marray<T, NumElements>,
                     decltype(abs(sycl::marray<complex<T>, NumElements>()))>);
     }).wait();
  }
}

TEMPLATE_TEST_CASE_SIG("Test marray complex polar function return types",
                       "[marray types]",
                       ((typename T, std::size_t NumElements), T, NumElements),
                       (double, 10), (float, 10), (sycl::half, 10)) {
  sycl::queue Q;

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
  if (is_type_supported<T>(Q)) {
    Q.single_task([=]() {
       static_assert(
           std::is_same_v<sycl::marray<complex<T>, NumElements>,
                          decltype(polar(sycl::marray<T, NumElements>()))>);
       static_assert(
           std::is_same_v<sycl::marray<complex<T>, NumElements>,
                          decltype(polar(sycl::marray<T, NumElements>(),
                                         sycl::marray<T, NumElements>()))>);
       static_assert(std::is_same_v<sycl::marray<complex<T>, NumElements>,
                                    decltype(polar(
                                        sycl::marray<T, NumElements>(), T()))>);
       static_assert(std::is_same_v<sycl::marray<complex<T>, NumElements>,
                                    decltype(polar(
                                        T(), sycl::marray<T, NumElements>()))>);
     }).wait();
  }
}

TEMPLATE_TEST_CASE_SIG("Test marray complex pow function return types",
                       "[marray types]",
                       ((typename T, std::size_t NumElements), T, NumElements),
                       (double, 10), (float, 10), (sycl::half, 10)) {
  sycl::queue Q;

  /* Check host code */
  // complex-deci
  static_assert(
      std::is_same_v<sycl::marray<complex<T>, NumElements>,
                     decltype(pow(sycl::marray<complex<T>, NumElements>(),
                                  sycl::marray<T, NumElements>()))>);
  static_assert(std::is_same_v<
                sycl::marray<complex<T>, NumElements>,
                decltype(pow(sycl::marray<complex<T>, NumElements>(), T()))>);
  static_assert(std::is_same_v<sycl::marray<complex<T>, NumElements>,
                               decltype(pow(complex<T>(),
                                            sycl::marray<T, NumElements>()))>);

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
  if (is_type_supported<T>(Q)) {
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
     }).wait();
  }
}

#define TEST(func)                                                             \
  TEMPLATE_TEST_CASE_SIG(                                                      \
      "Test marray complex " #func " function return types", "[marray types]", \
      ((typename T, std::size_t NumElements), T, NumElements), (double, 10),   \
      (float, 10), (sycl::half, 10)) {                                         \
    sycl::queue Q;                                                             \
    test##_##func##_##types<T, NumElements>(Q);                                \
  }

TEST(acos)
TEST(asin)
TEST(atan)
TEST(acosh)
TEST(asinh)
TEST(atanh)
TEST(conj)
TEST(cos)
TEST(cosh)
TEST(exp)
TEST(log)
TEST(log10)
TEST(proj)
TEST(sin)
TEST(sinh)
TEST(sqrt)
TEST(tan)
TEST(tanh)

TEST(add)
TEST(sub)
TEST(mul)
TEST(div)

TEST(eq)
TEST(inv_eq)
#undef TEST
