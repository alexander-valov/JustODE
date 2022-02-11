#pragma once

#include "Utils.hpp"

#include <vector>
#include <tuple>
#include <cmath>
#include <algorithm>
#include <numeric>

/********************************************************************
 * @brief Adaptive explicit Runge-Kutta algorithm abstract base class.
 * 
 * This class implements an embedded Runge-Kutta method with
 * automatic step-size control and initial step-size selection.
 * 
 * - [1] Colin Barr Macdonald "THE PREDICTED SEQUENTIAL 
 * REGULARIZATION METHOD FOR DIFFERENTIAL-ALGEBRAIC EQUATIONS"
 * @see https://www.math.ubc.ca/~cbm/bscthesis/cbm-bscthesis.pdf
 * 
 * @tparam T Floating point type
 * @tparam ErrOrder Error control order
 * @tparam NStages The number of stages of the Runge-Kutta method
 *********************************************************************/
template<class T, std::size_t ErrOrder, std::size_t NStages, detail::IsFloatingPoint<T> = true>
class RungeKuttaBase {

public:

    /********************************************************************
     * @brief Default constructor.
     * hmin is equal to 10 * std::numeric_limits<T>::min()
     * hmax is numerical infinity (std::numeric_limits<T>::max())
     * atol is 1e-6
     * rtol is 1e-3
     *********************************************************************/
    RungeKuttaBase() {
        SetHmin(T(10) * std::numeric_limits<T>::min());
        SetHmax(std::numeric_limits<T>::max());
        SetAtol(T(1.0e-6));
        SetRtol(T(1.0e-3));
    }

    /********************************************************************
     * @brief Constructor.
     * @param[in] hmin Min step size
     * @param[in] hmax Max step size
     * @param[in] atol Absolute tolerance
     * @param[in] rtol Relative tolerance
     *********************************************************************/
    RungeKuttaBase(
        const T& hmin, const T& hmax,
        const T& atol = T(1.0e-6), const T& rtol = T(1.0e-3)
    ) : hmin_(hmin), hmax_(hmax), atol_(atol), rtol_(rtol) {}

    //! Set min step size
    void SetHmin(const T& hmin) { hmin_ = hmin; }
    //! Set max step size
    void SetHmax(const T& hmax) { hmax_ = hmax; }
    //! Set absolute tolerance
    void SetAtol(const T& atol) { atol_ = atol; }
    //! Set relative tolerance
    void SetRtol(const T& rtol) { rtol_ = rtol; }

    /********************************************************************
     * @brief Solves initial value problem for first-order ODE
     * y' = rhs(t, y), y(t0) = y0
     * @param[in] rhs ODE right-hand-side
     * @param[in] interval Solution interval [t0, t_final]
     * @param[in] y0 Initial data
     * @return tuple: @a flag - status: 0 - finished, 1 - too small step size,
     *                @a tvals - vector of t,
     *                @a yvals - vector of y
     *********************************************************************/
    template<class Callable, class... Args>
    std::tuple<int, std::vector<T>, std::vector<T>> Solve(
        Callable&& rhs, const std::array<T, 2>& interval, const T& y0, Args&&... args
    ) {
        static_assert(
            std::is_invocable_r_v<T, Callable&&, T, T, Args&&...>,
            "Invalid signature or return type of the ODE right-hand-side!"
        );

        int flag = -1;
        std::vector<T> tvals, yvals;

        t_ = interval[0];
        y_ = y0;
        h_ = CalcInitialStep(rhs, interval[0], y0, args...);
        tvals.push_back(t_);
        yvals.push_back(y_);

        while (flag == -1) {
            bool step_state = Step(rhs, interval[1], args...);

            if (step_state) {
                // current step accepted
                tvals.push_back(t_);
                yvals.push_back(y_);
            } else {
                // current step rejected: step size h_ less than hmin_
                flag = 1;
            }

            if (t_ >= interval[1]) {
                // calculation finished
                flag = 0;
            } else if (t_ + h_ > interval[1]) {
                // trim time step
                h_ = interval[1] - t_;
            }
        }
        return std::make_tuple(flag, tvals, yvals);
    }

protected:

    /********************************************************************
     * @brief Calculates the initial step size for Runge-Kutta method.
     * @param[in] rhs ODE right-hand-side
     * @param[in] t0 Initial time
     * @param[in] y0 Initial solution
     * @return Initial step size
     *********************************************************************/
    template<class Callable, class... Args>
    T CalcInitialStep(Callable&& rhs, const T& t0, const T& y0, Args&&... args) {
        // calculate step for second derivative approximation
        T f0 = rhs(t0, y0, args...);
        T scale = atol_ + std::abs(y0) * rtol_;
        T d0 = std::abs(y0 / scale);
        T d1 = std::abs(f0 / scale);
        T h0;
        if (d0 < T(1e-5) || d1 < T(1e-5)) { h0 = T(1e-6); }
        else { h0 = T(0.01) * d0 / d1; }

        // second derivative approximation
        T y1 = y0 + h0 * f0;
        T f1 = rhs(t0 + h0, y1, args...);
        T d2 = std::abs((f1 - f0) / scale) / h0;

        T h1;
        if (d1 <= T(1e-15) && d2 <= T(1e-15)) {
            h1 = std::max(T(1e-6), T(1e-3) * h0);
        } else {
            h1 = std::pow(T(0.01) / std::max(d1, d2), 1 / (ErrOrder + 1));
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
     * If actual step size h_ less than hmin_ then returns false
     * (too small step size).
     * 
     * @param[in] rhs ODE right-hand-side
     * @param[in] t_final Final time of the given solution interval
     * @return Step status (success or fail if step size is too small)
     *********************************************************************/
    template<class Callable, class... Args>
    bool Step(Callable&& rhs, const T& t_final, Args&&... args) {
        hmin_ = std::max(T(10) * std::abs(std::nextafter(t_, std::numeric_limits<T>::max()) - t_), hmin_);

        bool is_accepted = false;
        while (!is_accepted) {
            // if the step size is too small, then the step is rejected and abort calculations
            if (h_ < hmin_) { return false; }

            // perform solving RK-step for current step-size and estimate error
            T y_new = RKStep(rhs, t_, y_, h_, args...);
            T delta = atol_ + std::max(std::abs(y_), std::abs(y_new)) * rtol_;
            T xi = ErrorEstimation(K_, delta);

            if (xi <= 1) {
                // step is accepted
                is_accepted = true;
                t_ = t_ + h_;
                y_ = y_new;
            }
            
            // calculate new step size according to estimated error
            T scale = max_factor_;
            if (xi != 0) { scale = safety_factor_ * std::pow(T(1) / xi, error_exponent_); }
            scale = std::clamp(scale, min_factor_, max_factor_);
            h_ = std::min(scale * h_, hmax_);
        }
        return true;
    }

    /********************************************************************
     * @brief Runge-Kutta single step.
     * @param[in] rhs ODE right-hand-side
     * @param[in] t Time from the previous step
     * @param[in] y Solution from the previous step
     * @param[in] h Current step size
     * @return Solution for the current step
     *********************************************************************/
    template<class Callable, class... Args>
    T RKStep(Callable&& rhs, const T& t, const T& y, const T& h, Args&&... args) {
        for (std::size_t i = 0; i < NStages; i++) {
            T dy = std::inner_product(A()[i].begin(), A()[i].end(), K_.begin(), T(0));
            K_[i] = h * rhs(t + C()[i] * h, y + dy, args...);
        }
        return y + std::inner_product(B().begin(), B().end(), K_.begin(), T(0));
    }

    /********************************************************************
     * @brief Estimates the error of the given step.
     * @param[in] K Runge-Kutta stages
     * @param[in] delta Error scaling
     * @return Error estimation
     *********************************************************************/
    T ErrorEstimation(const std::array<T, NStages>& K, const T& delta) {
        return std::abs(std::inner_product(E().begin(), E().end(), K.begin(), T(0)) / delta);
    }

protected:

    //! C Butcher tableau submatrix, pure virtual
    virtual const std::array<T, NStages>& C() const = 0;
    //! A Butcher tableau submatrix, pure virtual
    virtual const std::array<std::array<T, NStages>, NStages>& A() const = 0;
    //! B Butcher tableau submatrix, pure virtual
    virtual const std::array<T, NStages>& B() const = 0;
    //! E matrix, provides error estimation, pure virtual
    virtual const std::array<T, NStages>& E() const = 0;

protected:
    T hmin_;    ///< min step size
    T hmax_;    ///< max step size
    T atol_;    ///< absolute tolerance
    T rtol_;    ///< relative tolerance

    T h_;                           ///< current step size
    T t_;                           ///< current time
    T y_;                           ///< current sulution
    std::array<T, NStages> K_{};    ///< current coefficients for stages

    const T safety_factor_ = T(0.9);                       ///< safety factor
    const T max_factor_ = T(10);                           ///< max step increasing factor
    const T min_factor_ = T(0.2);                          ///< max step decreasing factor
    const T error_exponent_ = T(1) / (T(1) + ErrOrder);    ///< error estimation exponent
};


/********************************************************************
 * @brief Adaptive Runge-Kutta 4(5) algorithm.
 * 
 * This class inherits from RungeKuttaBase which provides embedded
 * Runge-Kutta methods workflow with automatic step-size control and
 * initial step-size selection.
 * 
 * Scheme order: 4
 * Error order: 5
 * Stages: 6
 * 
 * [1] Colin Barr Macdonald "THE PREDICTED SEQUENTIAL 
 * REGULARIZATIONMETHOD FOR DIFFERENTIAL-ALGEBRAIC EQUATIONS"
 * @see https://www.math.ubc.ca/~cbm/bscthesis/cbm-bscthesis.pdf
 * 
 * @tparam T Floating point type
 *********************************************************************/
template<class T, detail::IsFloatingPoint<T> = true>
class RungeKutta45: public RungeKuttaBase<T, 4, 6> {

public:

    RungeKutta45() : RungeKuttaBase<T, 4, 6>() {}
    RungeKutta45(const T& hmin, const T& hmax, const T& atol, const T& rtol)
        : RungeKuttaBase<T, 4, 6>(hmin, hmax, atol, rtol) {}

protected:

    //! C Butcher tableau submatrix
    const std::array<T, 6>& C() const { return C_data; }
    //! A Butcher tableau submatrix
    const std::array<std::array<T, 6>, 6>& A() const { return A_data; }
    //! B Butcher tableau submatrix
    const std::array<T, 6>& B() const { return B_data; }
    //! E matrix, provides error estimation
    const std::array<T, 6>& E() const { return E_data; }

private:

    const std::array<T, 6> C_data{
        T(0), T(1)/T(4), T(3)/T(8), T(12)/T(13), T(1), T(1)/T(2)
    };
    const std::array<std::array<T, 6>, 6> A_data{
        std::array<T, 6>{T(0)           , T(0)            , T(0)            , T(0)           , T(0)        , T(0)},
        std::array<T, 6>{T(1)/T(4)      , T(0)            , T(0)            , T(0)           , T(0)        , T(0)},
        std::array<T, 6>{T(3)/T(32)     , T(9)/T(32)      , T(0)            , T(0)           , T(0)        , T(0)},
        std::array<T, 6>{T(1932)/T(2197), T(-7200)/T(2197), T(7296)/T(2197) , T(0)           , T(0)        , T(0)},
        std::array<T, 6>{T(439)/T(216)  , T(-8)           , T(3680)/T(513)  , T(-845)/T(4104), T(0)        , T(0)},
        std::array<T, 6>{T(-8)/T(27)    , T(2)            , T(-3544)/T(2565), T(1859)/T(4104), T(-11)/T(40), T(0)}
    };
    const std::array<T, 6> B_data{
        T(25)/T(216), T(0), T(1408)/T(2565), T(2197)/T(4104), T(-1)/T(5), T(0)
    };
    const std::array<T, 6> E_data{
        T(1)/T(360), T(0), T(-128)/T(4275), T(-2197)/T(75240), T(1)/T(50), T(2)/T(55)
    };
};