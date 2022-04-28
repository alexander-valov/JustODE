#pragma once

#include <type_traits>
#include <iterator>

namespace JustODE {

namespace detail {

namespace type_traits {

    // --------------------------------------------------------------------
    // Instantiated false type.
    // @see https://artificial-mind.net/blog/2020/10/03/always-false
    // --------------------------------------------------------------------
    template<class... T>
    constexpr bool always_false = false;

    /// Detection idiom details
    namespace detection_idiom {
        // --------------------------------------------------------------------
        // C++17 compatible implementation of std::experimental::is_detected.
        // @see https://en.cppreference.com/w/cpp/experimental/is_detected
        // @see https://blog.tartanllama.xyz/detection-idiom/
        // @see https://people.eecs.berkeley.edu/~brock/blog/detection_idiom.php
        // --------------------------------------------------------------------
        template<template <class...> class Trait, class Enabler, class... Args>
        struct is_detected : std::false_type{};

        template<template <class...> class Trait, class... Args>
        struct is_detected<Trait, std::void_t<Trait<Args...>>, Args...> : std::true_type{};

        template<template <class...> class Trait, class... Args>
        using is_detected_t = typename is_detected<Trait, void, Args...>::type;
        template<template <class...> class Trait, class... Args>
        inline constexpr bool is_detected_v = is_detected<Trait, void, Args...>::value;


        // --------------------------------------------------------------------
        // Generalization of the detection idiom. In addition to detection,
        // checks the ability to convert return type to the specified type `RetType`.
        // --------------------------------------------------------------------
        template<class RetType, template <class...> class Trait, class Enabler, class... Args>
        struct is_detected_and_convertible : std::false_type{};

        template<class RetType, template <class...> class Trait, class... Args>
        struct is_detected_and_convertible<RetType, Trait, std::void_t<Trait<Args...>>, Args...> : std::is_convertible<Trait<Args...>, RetType> {};

        template<class RetType, template <class...> class Trait, class... Args>
        using is_detected_and_convertible_t = typename is_detected_and_convertible<RetType, Trait, void, Args...>::type;
        template<class RetType, template <class...> class Trait, class... Args>
        inline constexpr bool is_detected_and_convertible_v = is_detected_and_convertible<RetType, Trait, void, Args...>::value;
    }

    // --------------------------------------------------------------------
    // Deduces type of the MatrixType element
    // --------------------------------------------------------------------
    template<class MatrixType>
    using MatrixElemType = std::decay_t<decltype(std::declval<MatrixType>()(0, 0))>;

    // --------------------------------------------------------------------
    // Checks for specific method
    // --------------------------------------------------------------------
    // checks for std::size() method support
    template<class T>
    using method_size_t = decltype(std::size(std::declval<T>()));
    template<class T>
    using supports_size = detection_idiom::is_detected_t<method_size_t, T>;

    // checks for std::begin() method support
    template<class T>
    using method_begin_t = decltype(std::begin(std::declval<T>()));
    template<class T>
    using supports_begin = detection_idiom::is_detected_t<method_begin_t, T>;

    // checks for std::end() method support
    template<class T>
    using method_end_t = decltype(std::end(std::declval<T>()));
    template<class T>
    using supports_end = detection_idiom::is_detected_t<method_end_t, T>;

    // --------------------------------------------------------------------
    // Deduces element type of the iterable container
    // --------------------------------------------------------------------
    template<class Container>
    using elem_type_t = std::decay_t<decltype(*std::begin(std::declval<Container>()))>;

    // --------------------------------------------------------------------
    // Checks for real data container
    // --------------------------------------------------------------------
    // checks are container elements real
    template<class Container>
    using is_real_type_data = std::is_floating_point<elem_type_t<Container>>;

    // checks for real container
    template<class Container>
    using is_real_container = std::conjunction<
        supports_begin<Container>,
        supports_end<Container>,
        supports_size<Container>,
        is_real_type_data<Container>
    >;

    // --------------------------------------------------------------------
    // SFINAE alias to check for floating point data iterable container
    // --------------------------------------------------------------------
    template<typename T>
    using IsRealContainer = std::enable_if_t<is_real_container<T>::value, bool>;

    // --------------------------------------------------------------------
    // SFINAE alias to verify that the type is supports parethesis operator
    // --------------------------------------------------------------------
    template<class T>
    using method_parenthesis_t = decltype(std::declval<T>()(std::declval<std::size_t>(), std::declval<std::size_t>()));
    template<class T>
    inline constexpr bool has_mehod_parenthesis_v = detection_idiom::is_detected_v<method_parenthesis_t, T>;
    template<typename T>
    using IsSupportsParenthesis = std::enable_if_t<has_mehod_parenthesis_v<T>, bool>;

    // ---------------------------------------------------------
    // SFINAE alias to verify that the type is integral or floating point
    // ---------------------------------------------------------
    template<typename T>
    using IsFloatingPoint = std::enable_if_t<std::is_floating_point_v<T>, bool>;
    template<typename T>
    using IsIntegral = std::enable_if_t<std::is_integral_v<T>, bool>;

    // ---------------------------------------------------------
    // SFINAE alias to check whether the parameters pack
    // contains or not a certain type
    // ---------------------------------------------------------
    template<class T, class ... Types>
    using IsContainsType = std::enable_if_t<std::disjunction_v< always_false<T>, std::is_same<T, Types> ... >, bool>;
    template<class T, class ... Types>
    using IsNotContainsType = std::enable_if_t<!std::disjunction_v< always_false<T>, std::is_same<T, Types> ... >, bool>;


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


    // --------------------------------------------------------------------
    // Detects the relevant method to get the number of rows for the `InputType`.
    // Detection Idioms that are used for the following purposes:
    // 1. Detecting a specific method or attribute.
    // 2. Checking the possibility of converting the return value
    //    to the specified type.
    // --------------------------------------------------------------------
    template<class T>
    using method_rows_t = decltype(std::declval<T>().rows());
    template<class T, class RetType>
    inline constexpr bool has_relevant_method_rows_v = detection_idiom::is_detected_and_convertible_v<RetType, method_rows_t, T>;

    template<class T>
    using attribute_n_rows_t = decltype(std::declval<T>().n_rows);
    template<class T, class RetType>
    inline constexpr bool has_relevant_attribute_n_rows_v = detection_idiom::is_detected_and_convertible_v<RetType, attribute_n_rows_t, T>;

    template<class InputType>
    int rows(const InputType& matrix) {
        if constexpr (has_relevant_method_rows_v<InputType, int>) {
            return matrix.rows();
        } else if constexpr (has_relevant_attribute_n_rows_v<InputType, int>) {
            return matrix.n_rows;
        } else {
            static_assert(
                always_false<InputType>,
                R"MESSAGE(
                There are no supported methods of the `InputType` for accessing the number of matrix rows.
                Supported methods:
                    1. rows()
                Supported attributes:
                    1. n_rows
                Note: return type must be convertible to `int` in the sense of std::is_convertible.
                )MESSAGE"
            );
        }
    }

    // --------------------------------------------------------------------
    // Detects the relevant method to get the number of columns for the `InputType`.
    // Detection Idioms that are used for the following purposes:
    // 1. Detecting a specific method or attribute.
    // 2. Checking the possibility of converting the return value
    //    to the specified type.
    // --------------------------------------------------------------------
    template<class T>
    using method_cols_t = decltype(std::declval<T>().cols());
    template<class T, class RetType>
    inline constexpr bool has_relevant_method_cols_v = detection_idiom::is_detected_and_convertible_v<RetType, method_cols_t, T>;

    template<class T>
    using method_columns_t = decltype(std::declval<T>().columns());
    template<class T, class RetType>
    inline constexpr bool has_relevant_method_columns_v = detection_idiom::is_detected_and_convertible_v<RetType, method_columns_t, T>;

    template<class T>
    using attribute_n_cols_t = decltype(std::declval<T>().n_cols);
    template<class T, class RetType>
    inline constexpr bool has_relevant_attribute_n_cols_v = detection_idiom::is_detected_and_convertible_v<RetType, attribute_n_cols_t, T>;

    template<class InputType>
    int cols(const InputType& matrix) {
        if constexpr (has_relevant_method_cols_v<InputType, int>) {
            return matrix.cols();
        } else if constexpr (has_relevant_method_columns_v<InputType, int>) {
            return matrix.columns();
        } else if constexpr (has_relevant_attribute_n_cols_v<InputType, int>) {
            return matrix.n_cols;
        } else {
            static_assert(
                always_false<InputType>,
                R"MESSAGE(
                There are no supported methods of the `InputType` for accessing the number of matrix columns.
                Supported methods:
                    1. cols()
                    2. columns()
                Supported attributes:
                    1. n_cols
                Note: return type must be convertible to `int` in the sense of std::is_convertible.
                )MESSAGE"
            );
        }
    }


    // --------------------------------------------------------------------
    // Detects the relevant method to transpose matrix of `InputType`.
    // Detection Idioms that are used for the following purposes:
    // 1. Detecting a specific method or attribute.
    // 2. Checking the possibility of converting the return value
    //    to the specified type.
    // --------------------------------------------------------------------
    template<class T>
    using method_transpose_t = decltype(std::declval<T>().transpose());
    template<class T, class RetType>
    inline constexpr bool has_relevant_method_transpose_v = detection_idiom::is_detected_and_convertible_v<RetType, method_transpose_t, T>;

    template<class T>
    using method_t_t = decltype(std::declval<T>().t());
    template<class T, class RetType>
    inline constexpr bool has_relevant_method_t_v = detection_idiom::is_detected_and_convertible_v<RetType, method_t_t, T>;

    template<class InputType>
    InputType transpose(const InputType& matrix) {
        if constexpr (has_relevant_method_transpose_v<InputType, InputType>) {
            return matrix.transpose();
        } else if constexpr (has_relevant_method_t_v<InputType, InputType>) {
            return matrix.t();
        } else if constexpr (std::is_copy_assignable_v<InputType>) {
            InputType transposed = matrix;
            assert(rows(matrix) == cols(matrix) && "Default transpose supports only square matrix");
            auto n_rows = rows(matrix);
            for (int ir = 0; ir < n_rows; ir++) {
                for (int ic = ir + 1; ic < n_rows; ic++) {
                    std::swap(transposed(ir, ic), transposed(ic, ir));
                }
            }
            return transposed;
        } else {
            static_assert(
                always_false<InputType>,
                R"MESSAGE(
                There are no supported methods of the `InputType` to transpose matrix.
                Supported methods:
                    1. transpose()
                    2. t()
                    3. Copy assignment operator and operator()(size_t, size_t)
                Note:
                    1. The presence of `Copy assignment operator and operator()(size_t, size_t)` is necessary
                       for the default transposition algorithm, that uses std::swap().
                    2. Return type must be convertible to `InputType` in the sense of std::is_convertible.
                )MESSAGE"
            );
        }
    }
}

}

}