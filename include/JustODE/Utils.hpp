#pragma once

#include <type_traits>
#include <iterator>
#include <limits>

namespace JustODE {

namespace detail {

    /** @brief Discarded statement dependent of the template parameters */
    template <typename T>
    struct always_false : std::false_type {};

    // ---------------------------------------------------------
    // Checks whether the parameters pack contains a certain type
    // ---------------------------------------------------------
    template<class T, class ... Types>
    using IsContainsType = std::enable_if_t<std::disjunction_v< always_false<T>, std::is_same<T, Types> ... >, bool>;

    template<class T, class ... Types>
    using IsNotContainsType = std::enable_if_t<!std::disjunction_v< always_false<T>, std::is_same<T, Types> ... >, bool>;

    // ---------------------------------------------------------
    // SFINAE to verify that the type is integral or floating point
    // ---------------------------------------------------------
    template<typename T>
    using IsFloatingPoint = std::enable_if_t<std::is_floating_point_v<T>, bool>;

    template<typename T>
    using IsIntegral = std::enable_if_t<std::is_integral_v<T>, bool>;

    // ---------------------------------------------------------
    // Promotes the given type to a floating point type
    // ---------------------------------------------------------
    template<class T, bool = std::is_integral<T>::value>
    struct promote_fp { typedef double type; };

    template<class T>
    struct promote_fp<T, false> {};

    template<>
    struct promote_fp<long double> { typedef long double type; };

    template<>
    struct promote_fp<double> { typedef double type; };

    template<>
    struct promote_fp<float> { typedef float type; };

    template<
        class T1, class T2, 
        class TP1 = typename promote_fp<T1>::type,
        class TP2 = typename promote_fp<T2>::type
    >
    struct promote_fp_2 { typedef std::remove_reference_t<decltype(TP1() + TP2())> type; };

    template<
        class T1, class T2, class T3,
        class TP1 = typename promote_fp<T1>::type,
        class TP2 = typename promote_fp<T2>::type,
        class TP3 = typename promote_fp<T3>::type
    >
    struct promote_fp_3 { typedef std::remove_reference_t<decltype(TP1() + TP2() + TP3())> type; };

    // ---------------------------------------------------------
    // Detection idiom
    // C++17 compatible implementation of std::experimental::is_detected
    // @see https://people.eecs.berkeley.edu/~brock/blog/detection_idiom.php
    // @see https://blog.tartanllama.xyz/detection-idiom/
    // ---------------------------------------------------------
    namespace detect_detail {
        template<template <class...> class Trait, class Enabler, class... Args>
        struct is_detected : std::false_type{};

        template<template <class...> class Trait, class... Args>
        struct is_detected<Trait, std::void_t<Trait<Args...>>, Args...> : std::true_type{};
    }
    template<template <class...> class Trait, class... Args>
    using is_detected = typename detect_detail::is_detected<Trait, void, Args...>::type;

    // checks for std::size() method support
    template<class T>
    using method_size_t = decltype(std::size(std::declval<T>()));
    template<class T>
    using supports_size = is_detected<method_size_t, T>;

    // checks for std::begin() method support
    template<class T>
    using method_begin_t = decltype(std::begin(std::declval<T>()));
    template<class T>
    using supports_begin = is_detected<method_begin_t, T>;

    // checks for std::end() method support
    template<class T>
    using method_end_t = decltype(std::end(std::declval<T>()));
    template<class T>
    using supports_end = is_detected<method_end_t, T>;

    // obtains element type of iterable container
    template<class Container>
    using elem_type_t = std::decay_t<decltype(*std::begin(std::declval<Container>()))>;

    // checks are container elements real
    template<class Container, class Real>
    using is_real_type_data = std::is_same<Real, elem_type_t<Container>>;

    // chacks for real container
    template<class Container, class Real>
    using is_real_container = std::conjunction<
        std::is_floating_point<Real>,
        supports_begin<Container>,
        supports_end<Container>,
        supports_size<Container>,
        is_real_type_data<Container, Real>
    >;

    // ---------------------------------------------------------
    // Default parameters for embedded solvers
    // ---------------------------------------------------------
    template<class T, detail::IsFloatingPoint<T> = true>
    struct DefaultParams {
        constexpr static T atol = T(1e-6);
        constexpr static T rtol = T(1e-3);
        constexpr static T hmax = std::numeric_limits<T>::max();
    };
}

}