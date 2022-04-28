/*
 * JustODE https://github.com/alexander-valov/JustODE
 *
 * MIT License
 *
 * Copyright (c) 2022 Alexander Valov <https://github.com/alexander-valov>
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

#define JUST_ODE_VERSION_MAJOR 0
#define JUST_ODE_VERSION_MINOR 8
#define JUST_ODE_VERSION_PATCH 0

// #include "EmbeddedSolvers.hpp"


// #include "Utils/TraitsHelper.hpp"


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
// #include "RungeKuttaBase.hpp"


#include <map>
#include <cmath>
#include <array>
#include <vector>
#include <limits>
#include <utility>
#include <numeric>
#include <algorithm>
#include <optional>

// #include "Utils/TraitsHelper.hpp"

// #include "Utils/Operations.hpp"


#include <cmath>
#include <numeric>
#include <algorithm>

// #include "TraitsHelper.hpp"


namespace JustODE {

namespace detail {

    // Coefficient-wise 
    template<class Container, type_traits::IsRealContainer<Container> = true>
    Container AbsCwise(const Container& container) {
        Container result = container;
        for (auto& item : result) {
            item = std::abs(item);
        }
        return result;
    }

    /// Coefficient-wise maximum
    template<class Container, type_traits::IsRealContainer<Container> = true>
    Container MaxCwise(const Container& left, const Container& right) {
        Container res = left;
        auto first1 = std::begin(res);
        auto last1  = std::end(res);
        auto first2 = std::begin(right);
        for (; first1 != last1; ++first1, (void)++first2) {
            if (*first2 > *first1) {
                *first1 = *first2;
            }
        }
        return res;
    }

    /// Coefficient-wise summation
    template<class Container, type_traits::IsRealContainer<Container> = true>
    Container PlusCwise(const Container& left, const Container& right) {
        Container res = left;
        auto first1 = std::begin(res);
        auto last1  = std::end(res);
        auto first2 = std::begin(right);
        for (; first1 != last1; ++first1, (void)++first2) {
            *first1 += *first2;
        }
        return res;
    }
    /// Coefficient-wise summation
    template<class Container, type_traits::IsRealContainer<Container> = true>
    Container PlusCwise(Container&& left, const Container& right) {
        auto first1 = std::begin(left);
        auto last1  = std::end(left);
        auto first2 = std::begin(right);
        for (; first1 != last1; ++first1, (void)++first2) {
            *first1 += *first2;
        }
        return left;
    }

    /// Coefficient-wise subtraction
    template<class Container, type_traits::IsRealContainer<Container> = true>
    Container MinusCwise(const Container& left, const Container& right) {
        Container res = left;
        auto first1 = std::begin(res);
        auto last1  = std::end(res);
        auto first2 = std::begin(right);
        for (; first1 != last1; ++first1, (void)++first2) {
            *first1 -= *first2;
        }
        return res;
    }
    /// Coefficient-wise subtraction
    template<class Container, type_traits::IsRealContainer<Container> = true>
    Container MinusCwise(Container&& left, const Container& right) {
        auto first1 = std::begin(left);
        auto last1  = std::end(left);
        auto first2 = std::begin(right);
        for (; first1 != last1; ++first1, (void)++first2) {
            *first1 -= *first2;
        }
        return left;
    }

    /// Coefficient-wise division
    template<class Container, type_traits::IsRealContainer<Container> = true>
    Container DivCwise(const Container& left, const Container& right) {
        Container res = left;
        auto first1 = std::begin(res);
        auto last1  = std::end(res);
        auto first2 = std::begin(right);
        for (; first1 != last1; ++first1, (void)++first2) {
            *first1 /= *first2;
        }
        return res;
    }
    /// Coefficient-wise division
    template<class Container, type_traits::IsRealContainer<Container> = true>
    Container DivCwise(Container&& left, const Container& right) {
        auto first1 = std::begin(left);
        auto last1  = std::end(left);
        auto first2 = std::begin(right);
        for (; first1 != last1; ++first1, (void)++first2) {
            *first1 /= *first2;
        }
        return left;
    }

    /// Multiplies all coefficients by the given scalar
    template<class Container, class T, type_traits::IsRealContainer<Container> = true>
    Container MultScalar(const Container& container, const T& scalar) {
        Container res = container;
        for (auto& elem : res) { elem *= scalar; }
        return res;
    }
    /// Multiplies all coefficients by the given scalar
    template<class Container, class T, type_traits::IsRealContainer<Container> = true>
    Container MultScalar(const T& scalar, const Container& container) {
        return MultScalar(container, scalar);
    }
    /// Multiplies all coefficients by the given scalar
    template<class Container, class T, type_traits::IsRealContainer<Container> = true>
    Container MultScalar(Container&& container, const T& scalar) {
        for (auto& elem : container) { elem *= scalar; }
        return container;
    }
    /// Multiplies all coefficients by the given scalar
    template<class Container, class T, type_traits::IsRealContainer<Container> = true>
    Container MultScalar(const T& scalar, Container&& container) {
        return MultScalar(container, scalar);
    }

    /// Add the given scalar to all coefficients
    template<class Container, class T, type_traits::IsRealContainer<Container> = true>
    Container PlusScalar(const Container& container, const T& scalar) {
        Container res = container;
        for (auto& elem : res) { elem += scalar; }
        return res;
    }
    /// Add the given scalar to all coefficients
    template<class Container, class T, type_traits::IsRealContainer<Container> = true>
    Container PlusScalar(const T& scalar, const Container& container) {
        return PlusScalar(container, scalar);
    }
    /// Add the given scalar to all coefficients
    template<class Container, class T, type_traits::IsRealContainer<Container> = true>
    Container PlusScalar(Container&& container, const T& scalar) {
        for (auto& elem : container) { elem += scalar; }
        return container;
    }
    /// Add the given scalar to all coefficients
    template<class Container, class T, type_traits::IsRealContainer<Container> = true>
    Container PlusScalar(const T& scalar, Container&& container) {
        return PlusScalar(container, scalar);
    }

    /// Matrix-vector multiplication
    template<
        class MatrixType,
        class Container,
        type_traits::IsSupportsParenthesis<MatrixType> = true,
        type_traits::IsRealContainer<Container> = true
    >
    Container MVProd(const MatrixType& matrix, const Container& vec) {
        Container result = vec;
        for (auto& item : result) { item = 0; }
        for (int ic = 0; ic < type_traits::cols(matrix); ic++) {
            auto vec_it = std::next(std::begin(vec), ic);
            for (int ir = 0; ir < type_traits::rows(matrix); ir++) {
                auto res_it = std::next(std::begin(result), ir);
                *res_it += matrix(ir, ic) * (*vec_it);
            }
        }
        return result;
    }

    /// Computes Root Mean Square norm on the given container
    template<class Container, class T, type_traits::IsRealContainer<Container> = true>
    T RMSNorm(const Container& container, T init) {
        return std::sqrt(
            std::transform_reduce(
                std::begin(container), std::end(container), std::begin(container), init
            ) / std::size(container)
        );
    }

    /// Squared 2-norm of the vector
    template<class Container, class T, type_traits::IsRealContainer<Container> = true>
    T SquaredNorm(const Container& container, T init) {
        return std::transform_reduce(
            std::begin(container), std::end(container), std::begin(container), init
        );
    }

}

}

namespace JustODE {

/********************************************************************
 * @brief Stores ODE solution *y(t)* and a solver iformation.
 *********************************************************************/
template<class T>
struct ODEResult {
    std::vector<T> t;               ///< The vector of t
    std::vector<std::vector<T>> y;  ///< The vector of y
    int status = 0;                 ///< The solver termination status
    std::string message;            ///< Termination cause description
    std::size_t nfev = 0;           ///< Number of RHS evaluations
};

namespace detail {

/********************************************************************
 * @brief Adaptive explicit Runge-Kutta algorithm base class.
 * 
 * This class implements an embedded Runge-Kutta method with
 * automatic step-size control and initial step-size selection.
 * 
 * To implement a specific explicit RK method, you should:
 *   1. Specify method's error order *ErrOrder* and number of stages
 *      *NStages* as template parameters.
 *   2. Define the following Butcher Tableau submatrices:
 *      *C*, *A*, *B*, and *E*, where submatrix *E* is used for
 *      error estimation and defined as *E = \\hat{B} - B*.
 *      The *C* and *B* must be std::array<T, NStages>. The submatrix
 *      *A* must be std::array<std::array<T, NStages>, NStages>.
 *      The error estimation submatrix *E* can be std::array<T, NStages>
 *      or std::array<T, NStages + 1>. The extended array *E* 
 *      is applicable for methods like Dormand-Prince 4(5), there
 *      the last stage can be eliminated since the last
 *      stage coincides with the approximation computation. That
 *      technique is used to reduce the number of RHS evaluations.
 *   3. A constructor must accept parameters presented in the base
 *      class (listed below).
 * 
 * This class implements the Curiously Recurring Template Pattern
 * (CRTP). CRTP is used to implement static polymorphism. That allows
 * a user to define specific Butcher Tableau in a derived class.
 * 
 * References:
 * - [1] Colin Barr Macdonald "The predicted sequential regularization 
 *       method for differential-algebraic equations"
 * - [2] J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta
 *       formulae", Journal of Computational and Applied Mathematics
 * - [3] Butcher J. C. "Numerical methods for ordinary differential
 *       equations", John Wiley & Sons, 2016
 * 
 * @tparam Derived The type of derived class
 * @tparam T Floating point type
 * @tparam Container Floating point data container
 * @tparam ErrOrder Error control order
 * @tparam NStages The number of stages of the Runge-Kutta method
 * 
 * @see https://www.math.ubc.ca/~cbm/bscthesis/cbm-bscthesis.pdf
 * @see https://core.ac.uk/download/pdf/81989096.pdf
 * @see https://onlinelibrary.wiley.com/doi/book/10.1002/9781119121534
 * @see https://www.math.auckland.ac.nz/~butcher/ODE-book-2008/Tutorials/RK-methods.pdf
 * @see https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
 * @see https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
 *********************************************************************/
template<
    class Derived,
    class T,
    class Container,
    std::size_t ErrOrder,
    std::size_t NStages,
    type_traits::IsFloatingPoint<T> = true,
    type_traits::IsRealContainer<Container> = true
>
class RungeKuttaBase {

private:

    /********************************************************************
     * This class provides access to protected fields of a Derived class.
     * @see https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern
     * @see https://stackoverflow.com/a/55928800
     *********************************************************************/
    struct Accessor : Derived {
        constexpr static auto& A = Derived::A;
        constexpr static auto& B = Derived::B;
        constexpr static auto& C = Derived::C;
        constexpr static auto& E = Derived::E;
    };

public:

    /********************************************************************
     * @brief Constructs from tolerances and step sizes.
     * @param[in] atol Absolute tolerance.
     * @param[in] rtol Relative tolerance.
     * @param[in] hmax Max step size.
     * @param[in] h_start Start step size. If not specified or std::nullopt
     *                    then initial step selected automatic.
     *********************************************************************/
    RungeKuttaBase(
        std::optional<T> atol = std::nullopt,
        std::optional<T> rtol = std::nullopt,
        std::optional<T> hmax = std::nullopt,
        std::optional<T> h_start = std::nullopt
    ) {
        static_assert(
            (Accessor::A.size() == NStages && Accessor::A.front().size() == NStages) &&
            (Accessor::E.size() == NStages || Accessor::E.size() == NStages + 1) &&
            (Accessor::B.size() == NStages) &&
            (Accessor::C.size() == NStages),
            R"MESSAGE(
            Check dimensions of the Butcher Tableau submatrices:
                'A' must be NStages x NStages array (std::array<std::array<T, NStages>, NStages>),
                'B' must be NStages array (std::array<T, NStages>),
                'C' must be NStages array (std::array<T, NStages>),
                'E' must be NStages     array (std::array<T, NStages>)     for a regular methods or
                            NStages + 1 array (std::array<T, NStages + 1>) for an extended error estimation.
            )MESSAGE"
        );

        SetAtol(atol.value_or(T(1e-6)));
        SetRtol(rtol.value_or(T(1e-3)));
        SetHmax(hmax.value_or(std::numeric_limits<T>::max()));
        SetHStart(h_start);
    }

    /// Set absolute tolerance
    void SetAtol(const T& atol) { atol_ = atol; }
    /// Set relative tolerance
    void SetRtol(const T& rtol) { rtol_ = rtol; }
    /// Set max step size
    void SetHmax(const T& hmax) { hmax_ = hmax; }
    /// Set start step size
    void SetHStart(std::optional<T> h_start = std::nullopt) { h_start_ = h_start; }

    /// Get absolute tolerance
    const T& GetAtol() { return atol_; }
    /// Get relative tolerance
    const T& GetRtol() { return rtol_; }
    /// Get max step size
    const T& GetHmax() { return hmax_; }
    /// Get user defined start step size (if std::nullopt then initial step selected automatic)
    std::optional<T> GetUserHStart() { return h_start_; }

    /********************************************************************
     * @brief Solves initial value problem for first-order ODE
     * 
     *     dy / dt = rhs(t, y),
     *     y(t0) = y0.
     * 
     * @param[in] rhs ODE right-hand-side
     * @param[in] interval Solution interval [t0, t_final]
     * @param[in] y0 Initial data
     * @return ODEResult<T> object
     *********************************************************************/
    template<class Callable>
    ODEResult<T> Solve(
        Callable&& rhs, const std::array<T, 2>& interval, const Container& y0
    ) {
        static_assert(
            std::is_invocable_r_v<Container, Callable&&, const T&, const Container&>,
            "Invalid signature or return type of the ODE right-hand-side!"
        );

        // Number of differential equations
        std::size_t n_eq = std::size(y0);

        int flag = -1;
        std::size_t nfev = 0;
        std::vector<T> tvals;
        std::vector<std::vector<T>> yvals(n_eq);

        // Right-hand side wrapper with nfev calculation support
        auto rhs_wrapper = [&](const T& t, const Container& y) {
            nfev++;
            return rhs(t, y);
        };

        // Initialization
        t_ = interval[0];
        y_ = y0;
        f_ = rhs_wrapper(t_, y_);
        if (!h_start_.has_value()) {
            h_ = CalcInitialStep(rhs_wrapper, t_, y_, f_);
        } else {
            h_ = h_start_.value();
        }
        tvals.push_back(t_);
        for (std::size_t i = 0; i < n_eq; i++) {
            auto it = std::next(std::begin(y_), i);
            yvals[i].push_back(*it);
        }

        // Main integration loop
        while (flag == -1) {
            bool step_state = Step(rhs_wrapper, interval[1]);

            if (step_state) {
                // current step accepted
                tvals.push_back(t_);
                for (std::size_t i = 0; i < n_eq; i++) {
                    auto it = std::next(std::begin(y_), i);
                    yvals[i].push_back(*it);
                }
            } else {
                // current step rejected: step size h_ less than hmin
                flag = 1;
            }

            if (t_ >= interval[1]) {
                // calculation finished
                flag = 0;
            }
        }

        // Save results
        ODEResult<T> result;
        result.status = flag;
        result.message = messages_.at(flag);
        result.nfev = nfev;
        result.t = std::move(tvals);
        result.y = std::move(yvals);
        return result;
    }

protected:

    /********************************************************************
     * @brief Calculates the initial step size for Runge-Kutta method.
     * @param[in] rhs ODE right-hand-side
     * @param[in] t0 Initial time
     * @param[in] y0 Initial solution
     * @param[in] f0 Initial value of RHS
     * @return Initial step size
     *********************************************************************/
    template<class Callable>
    T CalcInitialStep(Callable&& rhs, const T& t0, const Container& y0, const Container& f0) {
        // calculate step for second derivative approximation
        Container scale = PlusScalar(atol_, MultScalar(AbsCwise(y0), rtol_));
        T d0 = RMSNorm(DivCwise(y0, scale), T(0));
        T d1 = RMSNorm(DivCwise(f0, scale), T(0));
        T h0;
        if (d0 < T(1e-5) || d1 < T(1e-5)) { h0 = T(1e-6); }
        else { h0 = T(0.01) * d0 / d1; }

        // second derivative approximation
        Container y1 = PlusCwise(y0, MultScalar(h0, f0));
        Container f1 = rhs(t0 + h0, y1);
        T d2 = RMSNorm(DivCwise(MinusCwise(f1, f0), scale), T(0)) / h0;

        T h1;
        if (d1 <= T(1e-15) && d2 <= T(1e-15)) {
            h1 = std::max(T(1e-6), T(1e-3) * h0);
        } else {
            h1 = std::pow(T(0.01) / std::max(d1, d2), T(1) / (ErrOrder + T(1)));
        }

        return std::min(T(100) * h0, h1);
    }

    /********************************************************************
     * @brief This method computes a single Runge-Kutta step.
     * 
     * Finds step size h_ and solution y(t + h_) such that
     * this solution meets the requirements of accuracy (atol and rtol).
     * If solution error for the current step size exceeds the tolerance
     * the step size is decreased until convergence is achieved.
     * If actual step size h_ less than hmin then returns false
     * (too small step size).
     * 
     * @param[in] rhs ODE right-hand-side
     * @param[in] t_final Final time of the given solution interval
     * @return Step status (success or fail if step size is too small)
     *********************************************************************/
    template<class Callable>
    bool Step(Callable&& rhs, const T& t_final) {
        T hmin = T(10) * std::abs(
            std::nextafter(t_, std::numeric_limits<T>::max()) - t_
        );

        bool is_accepted = false;
        bool is_rejected = false;
        while (!is_accepted) {
            // if the step size is too small, then the step is rejected 
            // and abort calculations
            if (h_ < hmin) { return false; }

            // trim time step
            T t_new = t_ + h_;
            if (t_new - t_final > T(0)) {
                t_new = t_final;
            }
            h_ = t_new - t_;

            // perform solving RK-step for current step-size and estimate error
            auto [y_new, f_new] = RKStep(rhs);
            Container delta = PlusScalar(atol_, MultScalar(MaxCwise(AbsCwise(y_), AbsCwise(y_new)), rtol_));
            T xi = ErrorEstimation(delta);

            if (xi <= 1) {
                T scale = max_factor_;
                if (xi != 0) {
                    scale = std::min(max_factor_, safety_factor_ * std::pow(xi, -error_exponent_));
                }
                // if step rejected then 1 is used as max_factor 
                if (is_rejected) { scale = std::min(T(1), scale); }
                h_ = std::min(scale * h_, hmax_);

                // step is accepted
                is_accepted = true;
                t_ = t_new;
                f_ = f_new;
                y_ = y_new;
            } else {
                T scale = std::max(min_factor_, safety_factor_ * std::pow(xi, -error_exponent_));
                h_ = std::min(scale * h_, hmax_);
                is_rejected = true;
            }
        }
        return true;
    }

    /********************************************************************
     * @brief Runge-Kutta single step.
     * @param[in] rhs ODE right-hand-side
     * @return Solution for the current step
     *********************************************************************/
    template<class Callable>
    std::pair<Container, Container> RKStep(Callable&& rhs) {
        K_[0] = f_;
        for (std::size_t i = 1; i < NStages; i++) {
            Container dy = MultScalar(h_, KTDot(Accessor::A[i], i));
            K_[i] = rhs(t_ + Accessor::C[i] * h_, PlusCwise(y_, dy));
        }

        Container KTDotB = KTDot(Accessor::B, Accessor::B.size());
        Container y_new = PlusCwise(y_, MultScalar(h_, KTDotB));
        Container f_new = rhs(t_ + h_, y_new);

        // Last stage calculation for the extended error estimation submatrix E
        K_.back() = f_new;

        return {y_new, f_new};
    }

    /********************************************************************
     * @brief Estimates the error of the given step.
     * @param[in] delta Error scaling
     * @return Error estimation
     *********************************************************************/
    T ErrorEstimation(const Container& delta) {
        Container KTDotE = KTDot(Accessor::E, Accessor::E.size());
        return RMSNorm(DivCwise(MultScalar(h_, KTDotE), delta), T(0));
    }

    /********************************************************************
     * @brief Compute dot product dot(K_.transpose(), array[:n_stages]).
     * @param[in] array Butcher tableau submatrix std::array
     * @param[in] n_stages Number of RK stages for dot product
     * @return dot(K_.transpose(), array[:n_stages])
     *********************************************************************/
    template<class Array>
    Container KTDot(const Array& array, std::size_t n_stages) {
        Container result = y_;
        auto res_it = result.begin();
        // loop over K_ columns (ODE equations)
        for (std::size_t ic = 0; ic < std::size(result); ic++) {
            T init = T(0);
            // loop over K_ rows (RK stages)
            for (std::size_t ir = 0; ir < n_stages; ir++) {
                auto iter = std::next(K_[ir].begin(), ic);
                init += array[ir] * (*iter);
            }
            *res_it = init;
            ++res_it;
        }
        return result;
    }

protected:
    T atol_;                   ///< absolute tolerance
    T rtol_;                   ///< relative tolerance
    T hmax_;                   ///< max step size
    std::optional<T> h_start_; ///< optional start step size

    T h_;                                    ///< current step size
    T t_;                                    ///< current time
    Container f_;                            ///< current rhs
    Container y_;                            ///< current sulution
    std::array<Container, NStages + 1> K_{}; ///< current coefficients for stages

    const T safety_factor_ = T(0.9);                    ///< safety factor
    const T max_factor_ = T(10);                        ///< max step increasing factor
    const T min_factor_ = T(0.2);                       ///< max step decreasing factor
    const T error_exponent_ = T(1) / (T(1) + ErrOrder); ///< error estimation exponent

    const std::map<int, std::string> messages_{
        {0, "Success"},
        {1, "Terminated. Too small time step"}
    }; ///< The array of possible solver messages
};

}

}

namespace JustODE {

namespace detail {

/********************************************************************
 * @brief Bogacki-Shampine 3(2) method.
 * 
 * Scheme order: 3
 * Error order: 2
 * Stages: 3 (applying FSAL - First Same As Last)
 * 
 * - [1] P. Bogacki, L.F. Shampine, "A 3(2) Pair of Runge-Kutta Formulas",
         Appl. Math. Lett. Vol. 2, No. 4. pp. 321-325, 1989.
 * 
 * @tparam T Floating point type
 * 
 * @see https://www.sciencedirect.com/science/article/pii/0893965989900797
 * @see https://en.wikipedia.org/wiki/Bogacki%E2%80%93Shampine_method
 *********************************************************************/
template<class T, class Container, type_traits::IsFloatingPoint<T> = true, type_traits::IsRealContainer<Container> = true>
class RK32: public RungeKuttaBase<RK32<T, Container>, T, Container, 2, 3> {

public:

    /********************************************************************
     * @brief Constructor of the RK32 method.
     * @param[in] atol Absolute tolerance. Default is 1e-6.
     * @param[in] rtol Relative tolerance. Default is 1e-3.
     * @param[in] hmax Max step size. Default is numerical infinity.
     *********************************************************************/
    RK32(
        std::optional<T> atol = std::nullopt,
        std::optional<T> rtol = std::nullopt,
        std::optional<T> hmax = std::nullopt,
        std::optional<T> h_start = std::nullopt
    ) : RungeKuttaBase<RK32<T, Container>, T, Container, 2, 3>(atol, rtol, hmax, h_start) {}

protected:

    constexpr static std::array<T, 3> C{
        T(0), T(1)/T(2), T(3)/T(4)
    };
    constexpr static std::array<std::array<T, 3>, 3> A{{
        {T(0)     , T(0)     , T(0)},
        {T(1)/T(2), T(0)     , T(0)},
        {T(0)     , T(3)/T(4), T(0)}
    }};
    constexpr static std::array<T, 3> B{
        T(2)/T(9), T(1)/T(3), T(4)/T(9)
    };
    constexpr static std::array<T, 4> E{
        T(5)/T(72), T(-1)/T(12), T(-1)/T(9), T(1)/T(8)
    };
};

/********************************************************************
 * @brief Runge-Kutta-Fehlberg 4(5) method.
 * 
 * Scheme order: 4
 * Error order: 5
 * Stages: 6
 * 
 * - [1] Fehlberg E. "Some experimental results concerning the error
 *       propagation in Runge-Kutta type integration formulas",
 *       National Aeronautics and Space Administration, 1970
 * 
 * @tparam T Floating point type
 * 
 * @see https://ntrs.nasa.gov/api/citations/19700031412/downloads/19700031412.pdf
 * @see https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method
 * @see https://www.johndcook.com/blog/2020/02/19/fehlberg/
 *********************************************************************/
template<class T, class Container, type_traits::IsFloatingPoint<T> = true, type_traits::IsRealContainer<Container> = true>
class RKF45: public RungeKuttaBase<RKF45<T, Container>, T, Container, 4, 6> {

public:

    /********************************************************************
     * @brief Constructor of the RKF45 method.
     * @param[in] atol Absolute tolerance. Default is 1e-6.
     * @param[in] rtol Relative tolerance. Default is 1e-3.
     * @param[in] hmax Max step size. Default is numerical infinity.
     *********************************************************************/
    RKF45(
        std::optional<T> atol = std::nullopt,
        std::optional<T> rtol = std::nullopt,
        std::optional<T> hmax = std::nullopt,
        std::optional<T> h_start = std::nullopt
    ) : RungeKuttaBase<RKF45<T, Container>, T, Container, 4, 6>(atol, rtol, hmax, h_start) {}

protected:

    constexpr static std::array<T, 6> C{
        T(0), T(1)/T(4), T(3)/T(8), T(12)/T(13), T(1), T(1)/T(2)
    };
    constexpr static std::array<std::array<T, 6>, 6> A{{
        {T(0)           , T(0)            , T(0)            , T(0)           , T(0)        , T(0)},
        {T(1)/T(4)      , T(0)            , T(0)            , T(0)           , T(0)        , T(0)},
        {T(3)/T(32)     , T(9)/T(32)      , T(0)            , T(0)           , T(0)        , T(0)},
        {T(1932)/T(2197), T(-7200)/T(2197), T(7296)/T(2197) , T(0)           , T(0)        , T(0)},
        {T(439)/T(216)  , T(-8)           , T(3680)/T(513)  , T(-845)/T(4104), T(0)        , T(0)},
        {T(-8)/T(27)    , T(2)            , T(-3544)/T(2565), T(1859)/T(4104), T(-11)/T(40), T(0)}
    }};
    constexpr static std::array<T, 6> B{
        T(25)/T(216), T(0), T(1408)/T(2565), T(2197)/T(4104), T(-1)/T(5), T(0)
    };
    constexpr static std::array<T, 6> E{
        T(1)/T(360), T(0), T(-128)/T(4275), T(-2197)/T(75240), T(1)/T(50), T(2)/T(55)
    };
};


/********************************************************************
 * @brief Dormand-Prince 5(4) method.
 * 
 * Scheme order: 5
 * Error order: 4
 * Stages: 6 (applying FSAL - First Same As Last)
 * 
 * - [1] J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta
 *       formulae", Journal of Computational and Applied Mathematics
 * 
 * @tparam T Floating point type
 * 
 * @see https://www.sciencedirect.com/science/article/pii/0771050X80900133
 * @see https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method
 * @see https://www.johndcook.com/blog/2020/02/19/dormand-prince/
 *********************************************************************/
template<class T, class Container, type_traits::IsFloatingPoint<T> = true, type_traits::IsRealContainer<Container> = true>
class DOPRI54: public RungeKuttaBase<DOPRI54<T, Container>, T, Container, 4, 6> {

public:

    /********************************************************************
     * @brief Constructor of the DOPRI54 method.
     * @param[in] atol Absolute tolerance. Default is 1e-6.
     * @param[in] rtol Relative tolerance. Default is 1e-3.
     * @param[in] hmax Max step size. Default is numerical infinity.
     *********************************************************************/
    DOPRI54(
        std::optional<T> atol = std::nullopt,
        std::optional<T> rtol = std::nullopt,
        std::optional<T> hmax = std::nullopt,
        std::optional<T> h_start = std::nullopt
    ) : RungeKuttaBase<DOPRI54<T, Container>, T, Container, 4, 6>(atol, rtol, hmax, h_start) {}

protected:

    constexpr static std::array<T, 6> C{
        T(0), T(1)/T(5), T(3)/T(10), T(4)/T(5), T(8)/T(9), T(1)
    };
    constexpr static std::array<std::array<T, 6>, 6> A{{
        {T(0)            , T(0)             , T(0)            , T(0)           , T(0)             , T(0)},
        {T(1)/T(5)       , T(0)             , T(0)            , T(0)           , T(0)             , T(0)},
        {T(3)/T(40)      , T(9)/T(40)       , T(0)            , T(0)           , T(0)             , T(0)},
        {T(44)/T(45)     , T(-56)/T(15)     , T(32)/T(9)      , T(0)           , T(0)             , T(0)},
        {T(19372)/T(6561), T(-25360)/T(2187), T(64448)/T(6561), T(-212)/T(729) , T(0)             , T(0)},
        {T(9017)/T(3168) , T(-355)/T(33)    , T(46732)/T(5247), T(49)/T(176)   , T(-5103)/T(18656), T(0)}
    }};
    constexpr static std::array<T, 6> B{
        T(35)/T(384), T(0), T(500)/T(1113), T(125)/T(192), T(-2187)/T(6784), T(11)/T(84)
    };
    constexpr static std::array<T, 7> E{
        T(-71)/T(57600), T(0), T(71)/T(16695), T(-71)/T(1920), T(17253)/T(339200), T(-22)/T(525), T(1)/T(40)
    };
};

}

}

namespace JustODE {

/********************************************************************
 * @brief List of available methods of ODE numerical integration.
 *********************************************************************/
enum class Methods {
    RK32,       ///< Explicit Bogacki-Shampine 3(2)
    RKF45,      ///< Explicit Runge-Kutta-Fehlberf 4(5)
    DOPRI54     ///< Explicit Dormand-Prince 5(4)
};

/********************************************************************
 * @brief Solves an initial value problem for a first-order ODE.
 * 
 * This method provides the numerical integration of a ordinary 
 * differential equation with prescribed initial data:
 * 
 *     dy / dt = rhs(t, y),
 *     y(t0) = y0.
 * 
 * For the numerical integration provided the following methods:
 * 
 * - *DOPRI54* (default): Explicit Dormand-Prince method of order 5(4), [1].
 * - *RK32* : Explicit Bogacki-Shampine method of order 3(2), [2].
 * - *RKF45*: Explicit Runge-Kutta-Fehlberg method of order 4(5), [3].
 * 
 * References:
 * 
 * - [1] J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta
 *       formulae", Journal of Computational and Applied Mathematics
 * - [2] P. Bogacki, L.F. Shampine, "A 3(2) Pair of Runge-Kutta Formulas",
 *       Appl. Math. Lett. Vol. 2, No. 4. pp. 321-325, 1989.
 * - [3] Fehlberg E. "Some experimental results concerning the error
 *       propagation in Runge-Kutta type integration formulas",
 *       National Aeronautics and Space Administration, 1970
 * 
 * @tparam T Floating point type.
 * @tparam method Which method to use. Default is JustODE::Methods::DOPRI54.
 * @tparam Callable Callable type (the invoke operation is applicable).
 * @tparam Args Types of additional arguments to pass to the RHS.
 * 
 * @param[in] rhs Right-hand side of ODE.
 * @param[in] interval The interval for solving the problem [t0, t_final].
 * @param[in] y0 Initial data y(t0).
 * @param[in] atol (optional) Absolute tolerance. Default is 1e-6.
 * @param[in] rtol (optional) Relative tolerance. Default is 1e-3.
 * @param[in] hmax (optional) Max step size. Default is numerical infinity.
 * @param[in] h_start (optional) Start step size. If not specified or 
 *                    std::nullopt then initial step selected automatic.
 * @return ODEResult<T> object
 *********************************************************************/
template<
    Methods method = Methods::DOPRI54,
    class T,
    class Container,
    class Callable,
    detail::type_traits::IsFloatingPoint<T> = true,
    detail::type_traits::IsRealContainer<Container> = true
>
ODEResult<T> SolveIVP(
    Callable&& rhs,
    const std::array<T, 2>& interval,
    const Container& y0,
    std::optional<detail::type_traits::elem_type_t<decltype(interval)>> atol    = std::nullopt,
    std::optional<detail::type_traits::elem_type_t<decltype(interval)>> rtol    = std::nullopt,
    std::optional<detail::type_traits::elem_type_t<decltype(interval)>> hmax    = std::nullopt,
    std::optional<detail::type_traits::elem_type_t<decltype(interval)>> h_start = std::nullopt
) {
    if constexpr (method == Methods::RK32) {
        auto solver = detail::RK32<T, Container>(atol, rtol, hmax, h_start);
        return solver.Solve(rhs, interval, y0);
    } else if constexpr (method == Methods::RKF45) {
        auto solver = detail::RKF45<T, Container>(atol, rtol, hmax, h_start);
        return solver.Solve(rhs, interval, y0);
    } else if constexpr (method == Methods::DOPRI54) {
        auto solver = detail::DOPRI54<T, Container>(atol, rtol, hmax, h_start);
        return solver.Solve(rhs, interval, y0);
    } else {
        static_assert(
            detail::type_traits::always_false<T>,
            "Unknown method. See avaliable methods in JustODE::Methods"
        );
    }
}

}