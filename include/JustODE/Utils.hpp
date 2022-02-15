#pragma once

#include <type_traits>

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
}

}