# pragma once

#include <map>
#include <cmath>
#include <array>
#include <vector>
#include <utility>
#include <numeric>
#include <algorithm>
#include <optional>

#include "Utils.hpp"

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