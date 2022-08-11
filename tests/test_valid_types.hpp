// Helper for testing each decimal type

// Generic template and overload
template <template <typename> typename action, typename... argsT>
bool test_valid_types(sycl::queue &Q, argsT... args) {
  bool test_passes = true;

  if (Q.get_device().has(sycl::aspect::fp64)) {
    action<double> test;
    test_passes &= test(Q, args...);
  }

  {
    action<float> test;
    test_passes &= test(Q, args...);
  }

  if (Q.get_device().has(sycl::aspect::fp16)) {
    action<sycl::half> test;
    test_passes &= test(Q, args...);
  }

  return test_passes;
}

// Overload for static_assert checks
template <template <typename> typename action>
bool test_valid_types(sycl::queue &Q) {
  bool test_passes = true;

  if (Q.get_device().has(sycl::aspect::fp64)) {
    action<double> test;
    test_passes &= test(Q);
  }

  {
    action<float> test;
    test_passes &= test(Q);
  }

  if (Q.get_device().has(sycl::aspect::fp16)) {
    action<sycl::half> test;
    test_passes &= test(Q);
  }

  return test_passes;
}

// Generic host only template and overload
template <template <typename> typename action, typename... argsT>
bool test_valid_types(argsT... args) {
  bool test_passes = true;

  {
    action<double> test;
    test_passes &= test(args...);
  }

  {
    action<float> test;
    test_passes &= test(args...);
  }

  {
    action<sycl::half> test;
    test_passes &= test(args...);
  }

  return test_passes;
}

// Generic template and overload for marray tests
template <template <typename, std::size_t> typename action,
          std::size_t NumElements, typename... argsT>
bool test_valid_types(sycl::queue &Q, argsT... args) {
  bool test_passes = true;

  if (Q.get_device().has(sycl::aspect::fp64)) {
    action<double, NumElements> test;
    test_passes &= test(Q, args...);
  }

  {
    action<float, NumElements> test;
    test_passes &= test(Q, args...);
  }

  if (Q.get_device().has(sycl::aspect::fp16)) {
    action<sycl::half, NumElements> test;
    test_passes &= test(Q, args...);
  }

  return test_passes;
}

// Overload for static_assert checks for marray tests
template <template <typename, std::size_t> typename action,
          std::size_t NumElements>
bool test_valid_types(sycl::queue &Q) {
  bool test_passes = true;

  if (Q.get_device().has(sycl::aspect::fp64)) {
    action<double, NumElements> test;
    test_passes &= test(Q);
  }

  {
    action<float, NumElements> test;
    test_passes &= test(Q);
  }

  if (Q.get_device().has(sycl::aspect::fp16)) {
    action<sycl::half, NumElements> test;
    test_passes &= test(Q);
  }

  return test_passes;
}

// Overload for marray tests with integer sequences
template <template <typename, std::size_t, std::size_t...> typename action,
          std::size_t NumElements, std::size_t... I, typename... argsT>
bool test_valid_types(sycl::queue &Q,
                      std::integer_sequence<std::size_t, I...> int_seq,
                      argsT... args) {
  bool test_passes = true;

  if (Q.get_device().has(sycl::aspect::fp64)) {
    action<double, NumElements, I...> test;
    test_passes &= test(Q, int_seq, args...);
  }

  {
    action<float, NumElements, I...> test;
    test_passes &= test(Q, int_seq, args...);
  }

  if (Q.get_device().has(sycl::aspect::fp16)) {
    action<sycl::half, NumElements, I...> test;
    test_passes &= test(Q, int_seq, args...);
  }

  return test_passes;
}
