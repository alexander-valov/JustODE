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

// #include "EmbeddedSolvers.hpp"


#include <limits>

// #include "Utils.hpp"


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

    template<typename T>
    using IsRealContainer = std::enable_if_t<is_real_container<T>::value, bool>;

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
// #include "RungeKuttaBase.hpp"


#include <map>
#include <cmath>
#include <array>
#include <vector>
#include <utility>
#include <numeric>
#include <algorithm>
#include <optional>

// #include "Utils.hpp"


namespace JustODE {

/********************************************************************
 * @brief Stores ODE solution *y(t)* and a solver iformation.
 *********************************************************************/
template<class T>
struct ODEResult {
    std::vector<T> t;      ///< The vector of t
    std::vector<T> y;      ///< The vector of y
    int status = 0;        ///< The solver termination status
    std::string message;   ///< Termination cause description
    std::size_t nfev = 0;  ///< Number of RHS evaluations
};

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
template<class Derived, class T, std::size_t ErrOrder, std::size_t NStages,
         detail::IsFloatingPoint<T> = true>
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

        SetAtol(atol.value_or(detail::DefaultParams<T>::atol));
        SetRtol(rtol.value_or(detail::DefaultParams<T>::rtol));
        SetHmax(hmax.value_or(detail::DefaultParams<T>::hmax));
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
        Callable&& rhs, const std::array<T, 2>& interval, const T& y0
    ) {
        static_assert(
            std::is_invocable_r_v<T, Callable&&, const T&, const T&>,
            "Invalid signature or return type of the ODE right-hand-side!"
        );

        int flag = -1;
        std::size_t nfev = 0;
        std::vector<T> tvals, yvals;

        // Right-hand side wrapper with nfev calculation support
        auto rhs_wrapper = [&](const T& t, const T& y) {
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
        yvals.push_back(y_);

        // Main integration loop
        while (flag == -1) {
            bool step_state = Step(rhs_wrapper, interval[1]);

            if (step_state) {
                // current step accepted
                tvals.push_back(t_);
                yvals.push_back(y_);
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
    T CalcInitialStep(Callable&& rhs, const T& t0, const T& y0, const T& f0) {
        // calculate step for second derivative approximation
        T scale = atol_ + std::abs(y0) * rtol_;
        T d0 = std::abs(y0 / scale);
        T d1 = std::abs(f0 / scale);
        T h0;
        if (d0 < T(1e-5) || d1 < T(1e-5)) { h0 = T(1e-6); }
        else { h0 = T(0.01) * d0 / d1; }

        // second derivative approximation
        T y1 = y0 + h0 * f0;
        T f1 = rhs(t0 + h0, y1);
        T d2 = std::abs((f1 - f0) / scale) / h0;

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
            T delta = atol_ + std::max(std::abs(y_), std::abs(y_new)) * rtol_;
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
    std::pair<T, T> RKStep(Callable&& rhs) {
        K_[0] = f_;
        for (std::size_t i = 1; i < NStages; i++) {
            auto it = std::next(Accessor::A[i].begin(), i);
            T dy = h_ * std::inner_product(
                Accessor::A[i].begin(), it, K_.begin(), T(0)
            );
            K_[i] = rhs(t_ + Accessor::C[i] * h_, y_ + dy);
        }

        T y_new = y_ + h_ * std::inner_product(
            Accessor::B.begin(), Accessor::B.end(), K_.begin(), T(0)
        );
        T f_new = rhs(t_ + h_, y_new);

        // Last stage calculation for the extended error estimation submatrix E
        K_.back() = f_new;

        return {y_new, f_new};
    }

    /********************************************************************
     * @brief Estimates the error of the given step.
     * @param[in] delta Error scaling
     * @return Error estimation
     *********************************************************************/
    T ErrorEstimation(const T& delta) {
        return std::abs(
            h_ * std::inner_product(
                Accessor::E.begin(), Accessor::E.end(), K_.begin(), T(0)
            ) / delta
        );
    }

    /// Coefficient-wise summation
    template<class Container, detail::IsRealContainer<Container> = true>
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
    template<class Container, detail::IsRealContainer<Container> = true>
    Container PlusCwise(Container&& left, const Container& right) {
        auto first1 = std::begin(left);
        auto last1  = std::end(left);
        auto first2 = std::begin(right);
        for (; first1 != last1; ++first1, (void)++first2) {
            *first1 += *first2;
        }
        return left;
    }

    /// Multiplies all coefficients by the given scalar
    template<class Real, class Container, detail::IsFloatingPoint<Real> = true, detail::IsRealContainer<Container> = true>
    Container MultScalar(const Container& container, const Real& scalar) {
        Container res = container;
        for (auto& elem : res) { elem *= scalar; }
        return res;
    }
    /// Multiplies all coefficients by the given scalar
    template<class Real, class Container, detail::IsFloatingPoint<Real> = true, detail::IsRealContainer<Container> = true>
    Container MultScalar(const Real& scalar, const Container& container) {
        return MultScalar(container, scalar);
    }
    /// Multiplies all coefficients by the given scalar
    template<class Real, class Container, detail::IsFloatingPoint<Real> = true, detail::IsRealContainer<Container> = true>
    Container MultScalar(Container&& container, const Real& scalar) {
        for (auto& elem : container) { elem *= scalar; }
        return container;
    }
    /// Multiplies all coefficients by the given scalar
    template<class Real, class Container, detail::IsFloatingPoint<Real> = true, detail::IsRealContainer<Container> = true>
    Container MultScalar(const Real& scalar, Container&& container) {
        return MultScalar(container, scalar);
    }

    /// Computes 2-norm on the given container
    template<class Container, detail::IsRealContainer<Container> = true>
    T Norm(const Container& container) {
        return std::sqrt(
            std::transform_reduce(
                std::begin(container), std::end(container), std::begin(container), T(0)
            )
        );
    }

protected:
    T atol_;                   ///< absolute tolerance
    T rtol_;                   ///< relative tolerance
    T hmax_;                   ///< max step size
    std::optional<T> h_start_; ///< optional start step size

    T h_;                            ///< current step size
    T t_;                            ///< current time
    T f_;                            ///< current rhs
    T y_;                            ///< current sulution
    std::array<T, NStages + 1> K_{}; ///< current coefficients for stages

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

namespace JustODE {

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
template<class T, detail::IsFloatingPoint<T> = true>
class RK32: public RungeKuttaBase<RK32<T>, T, 2, 3> {

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
    ) : RungeKuttaBase<RK32<T>, T, 2, 3>(atol, rtol, hmax, h_start) {}

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
template<class T, detail::IsFloatingPoint<T> = true>
class RKF45: public RungeKuttaBase<RKF45<T>, T, 4, 6> {

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
    ) : RungeKuttaBase<RKF45<T>, T, 4, 6>(atol, rtol, hmax, h_start) {}

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
template<class T, detail::IsFloatingPoint<T> = true>
class DOPRI54: public RungeKuttaBase<DOPRI54<T>, T, 4, 6> {

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
    ) : RungeKuttaBase<DOPRI54<T>, T, 4, 6>(atol, rtol, hmax, h_start) {}

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
    class Callable,
    detail::IsFloatingPoint<T> = true
>
ODEResult<T> SolveIVP(
    Callable&& rhs,
    const std::array<T, 2>& interval,
    const T& y0,
    std::optional<detail::elem_type_t<decltype(interval)>> atol    = std::nullopt,
    std::optional<detail::elem_type_t<decltype(interval)>> rtol    = std::nullopt,
    std::optional<detail::elem_type_t<decltype(interval)>> hmax    = std::nullopt,
    std::optional<detail::elem_type_t<decltype(interval)>> h_start = std::nullopt
) {
    if constexpr (method == Methods::RK32) {
        auto solver = RK32<T>(atol, rtol, hmax, h_start);
        return solver.Solve(rhs, interval, y0);
    } else if constexpr (method == Methods::RKF45) {
        auto solver = RKF45<T>(atol, rtol, hmax, h_start);
        return solver.Solve(rhs, interval, y0);
    } else if constexpr (method == Methods::DOPRI54) {
        auto solver = DOPRI54<T>(atol, rtol, hmax, h_start);
        return solver.Solve(rhs, interval, y0);
    } else {
        static_assert(
            detail::always_false<T>::value,
            "Unknown method. See avaliable methods in JustODE::Methods"
        );
    }
}

}