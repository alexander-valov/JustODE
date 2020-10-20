#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <limits>
#include <map>
#include <tuple>
#include <type_traits>


template <class T, std::size_t ErrOrder, std::size_t NStages>
class RungeKuttaAdaptive {

    public:

        /********************************************************************
         * @brief Default constructor
         * hmin is equal to length of [t, std::nextafter(t, infinity)] interval
         * hmax is numerical infinity (std::numeric_limits<T>::max())
         * atol is 1e-6
         * rtol is 1e-3
         *********************************************************************/
        RungeKuttaAdaptive() {
            SetHmin(T(10) * std::numeric_limits<T>::min());
            SetHmax(std::numeric_limits<T>::max());
            SetAtol(T(1.0e-6));
            SetRtol(T(1.0e-3));
        }

        /********************************************************************
         * @brief Default constructor
         * @param[in] hmin min step size
         * @param[in] hmax max step size
         * @param[in] atol absolute tolerance
         * @param[in] rtol relative tolerance
         *********************************************************************/
        RungeKuttaAdaptive(const T& hmin, const T& hmax, const T& atol, const T& rtol)
         : hmin_(hmin), hmax_(hmax), atol_(atol), rtol_(rtol) {}

        //! Set min step size
        void SetHmin(const T& hmin) { hmin_ = hmin; }
        //! Set max step size
        void SetHmax(const T& hmax) { hmax_ = hmax; }
        //! Set absolute tolerance
        void SetAtol(const T& atol) { atol_ = atol; }
        //! Set relative tolerance
        void SetRtol(const T& rtol) { rtol_ = rtol; }

        /********************************************************************
         * @brief Solve initial value problem for first-order ODE
         * y' = rhs(t, y), y(t0) = y0
         * @param[in] rhs ODE right-hand-side
         * @param[in] interval solution interval [t0, t_final]
         * @param[in] y0 initial data
         * @return tuple: flag - solution status,
         *                tvals - vector of t,
         *                yvals - vector of y
         *********************************************************************/
        template <class F>
        std::tuple<int, std::vector<T>, std::vector<T>> Solve(
            F rhs, const std::array<T, 2>& interval, const T& y0
        ) {
            int flag = -1;
            std::vector<T> tvals, yvals;

            t_ = interval[0];
            y_ = y0;
            h_ = CalcInitialStep(rhs, interval[0], y0);
            tvals.push_back(t_);
            yvals.push_back(y_);

            while (flag == -1) {
                bool step_state = Step(rhs, interval[1]);

                if (step_state) {
                    /* current step accepted */
                    tvals.push_back(t_);
                    yvals.push_back(y_);
                } else {
                    /* current step rejected: step size h_ less than hmin_ */
                    flag = 1;
                }

                if (t_ >= interval[1]) {
                    /* calculation finished */
                    flag = 0;
                } else if (t_ + h_ > interval[1]) {
                    /* trim time step */
                    h_ = interval[1] - t_;
                }
            }
            return std::make_tuple(flag, tvals, yvals);
        }

    protected:

        template <class F>
        T CalcInitialStep(F rhs, const T& t0, const T& y0) {
            T f0 = rhs(t0, y0);
            T scale = atol_ + std::abs(y0) * rtol_;
            T d0 = std::abs(y0 / scale);
            T d1 = std::abs(f0 / scale);
            T h0;
            if (d0 < T(1e-5) || d1 < T(1e-5)) {
                h0 = T(1e-6);
            }
            else {
                h0 = T(0.01) * d0 / d1;
            }

            T y1 = y0 + h0 * f0;
            T f1 = rhs(t0 + h0, y1);
            T d2 = std::abs((f1 - f0) / scale) / h0;

            T h1;
            if (d1 <= T(1e-15) && d2 <= T(1e-15)) {
                h1 = std::max(T(1e-6), T(1e-3) * h0);
            } else {
                h1 = std::pow(T(0.01) / std::max(d1, d2), 1 / (ErrOrder + 1));
            }

            return std::min(T(100) * h0, h1);
        }

        template <class F>
        bool Step(F rhs, const T& t_final) {
            hmin_ = std::max(T(10) * std::abs(std::nextafter(t_, std::numeric_limits<T>::max()) - t_), hmin_);

            bool is_accepted = false;

            while (!is_accepted) {
                if (h_ < hmin_) { return false; }

                T y_new = RKStep(rhs, t_, y_, h_);
                T delta = atol_ + std::max(std::abs(y_), std::abs(y_new)) * rtol_;
                T xi = ErrorEstimation(K_, delta);

                if (xi <= 1) {
                    is_accepted = true;
                    t_ = t_ + h_;
                    y_ = y_new;
                }
                
                T scale = max_factor_;
                if (xi != 0) { scale = safety_factor_ * std::pow(T(1) / xi, error_exponent_); }
                scale = std::clamp(scale, min_factor_, max_factor_);
                h_ = std::min(scale * h_, hmax_);
            }

            return true;
        }

        template <class F>
        T RKStep(F rhs, const T& t, const T& y, const T& h) {
            for (std::size_t i = 0; i < NStages; i++) {
                T dy = std::inner_product(A()[i].begin(), A()[i].end(), K_.begin(), T(0));
                K_[i] = h * rhs(t + C()[i] * h, y + dy);
            }
            return y + std::inner_product(B().begin(), B().end(), K_.begin(), T(0));
        }

        T ErrorEstimation(const std::array<T, NStages>& K, const T& delta) {
            return std::abs(std::inner_product(E().begin(), E().end(), K.begin(), T(0))) / delta;
        }

    protected:

        virtual const std::array<T, NStages>& C() const = 0;
        virtual const std::array<std::array<T, NStages>, NStages>& A() const = 0;
        virtual const std::array<T, NStages>& B() const = 0;
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
        const T max_factor_ = T(4);                            ///< max step increasing factor
        const T min_factor_ = T(0.25);                         ///< max step decreasing factor
        const T error_exponent_ = T(1) / (T(1) + ErrOrder);    ///< error estimation exponent
};


template <class T>
class RungeKutta45: public RungeKuttaAdaptive<T, 4, 6> {

    public:

        RungeKutta45() : RungeKuttaAdaptive<T, 4, 6>() {}
        RungeKutta45(const T& hmin, const T& hmax, const T& atol, const T& rtol)
            : RungeKuttaAdaptive<T, 4, 6>(hmin, hmax, atol, rtol) {}

    protected:

        const std::array<T, 6>& C() const { return C_data; }
        const std::array<std::array<T, 6>, 6>& A() const { return A_data; }
        const std::array<T, 6>& B() const { return B_data; }
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


template <class T, class F>
std::tuple<int, std::vector<T>, std::vector<T>> SolveRK45(F rhs, const std::array<T, 2>& interval, const T& y0) {
    int error_order = 4;
    T safety_factor = 0.8;
    T tol = 1.0e-5;
    T hmin = 0.05;
    T hmax = 0.2;

    std::array<T, 6> C{
        T(0), T(1)/T(4), T(3)/T(8), T(12)/T(13), T(1), T(1)/T(2)
    };
    std::array<std::array<T, 6>, 6> A{
        std::array<T, 6>{T(0)           , T(0)            , T(0)            , T(0)           , T(0)        , T(0)},
        std::array<T, 6>{T(1)/T(4)      , T(0)            , T(0)            , T(0)           , T(0)        , T(0)},
        std::array<T, 6>{T(3)/T(32)     , T(9)/T(32)      , T(0)            , T(0)           , T(0)        , T(0)},
        std::array<T, 6>{T(1932)/T(2197), T(-7200)/T(2197), T(7296)/T(2197) , T(0)           , T(0)        , T(0)},
        std::array<T, 6>{T(439)/T(216)  , T(-8)           , T(3680)/T(513)  , T(-845)/T(4104), T(0)        , T(0)},
        std::array<T, 6>{T(-8)/T(27)    , T(2)            , T(-3544)/T(2565), T(1859)/T(4104), T(-11)/T(40), T(0)}
    };
    std::array<T, 6> B{
        T(25)/T(216), T(0), T(1408)/T(2565), T(2197)/T(4104), T(-1)/T(5), T(0)
    };
    std::array<T, 6> E{
        T(1)/T(360), T(0), T(-128)/T(4275), T(-2197)/T(75240), T(1)/T(50), T(2)/T(55)
    };
    
    T h = hmax;
    T t = interval[0];
    T y = y0;

    std::vector<T> tvals, yvals;
    tvals.push_back(t);
    yvals.push_back(y);

    int flag = -1;
    while (flag == -1) {
        std::array<T, 6> K{};
        for (int i = 0; i < 6; i++) {
            T dy = std::inner_product(A[i].begin(), A[i].end(), K.begin(), T(0));
            K[i] = h * rhs(t + C[i] * h, y + dy);
        }

        T r = std::abs(std::inner_product(E.begin(), E.end(), K.begin(), T(0))) / h;
        if (r <= tol) {
            t = t + h;
            y = y + std::inner_product(B.begin(), B.end(), K.begin(), T(0));
            tvals.push_back(t);
            yvals.push_back(y);
        }

        T delta = safety_factor * std::pow(tol / r, T(1) / error_order);
        delta = std::clamp(delta, T(0.125), T(4));
        h = std::min(delta * h, hmax);

        if (t >= interval[1]) { flag = 0; }
        else if (t + h > interval[1]) { h = interval[1] - t; }
        else if (h < hmin) { flag = 1; }
    }
    return std::make_tuple(flag, tvals, yvals);
}

template <class T>
void PrintResults(int flag, const std::vector<T>& tvals, const std::vector<T>& yvals) {
    std::cout << "flag: " << flag << "\n";
    std::cout << "t = [";
    for (size_t i = 0; i < tvals.size(); i++) {
        std::cout << tvals[i];
        if (i < tvals.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
    std::cout << "y = [";
    for (size_t i = 0; i < yvals.size(); i++) {
        std::cout << yvals[i];
        if (i < yvals.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
}

int main() {
    // auto rhs = [](double t, double y) { return y; };
    // auto [flag, tvals, yvals] = SolveRK45(rhs, {0.0, 1.0}, 1.0);
    // PrintResults(flag, tvals, yvals);

    int nfev = 0;
    auto rhs = [&](double t, double y) {
        nfev += 1;
        return -2 * y + std::exp(-2 * (t - 6) * (t - 6));
    };
    auto [flag, tvals, yvals] = SolveRK45(rhs, {0.0, 15.0}, 1.0);
    std::cout << "nfev: " << nfev << "\n";
    std::cout << "size: " << tvals.size() << "\n";
    PrintResults(flag, tvals, yvals);

    nfev = 0;
    auto solver = RungeKutta45(0.05, 2.0, 1.0e-6, 1.0e-5);
    std::tie(flag, tvals, yvals) = solver.Solve(rhs, {0.0, 15.0}, 1.0);
    std::cout << "nfev: " << nfev << "\n";
    std::cout << "size: " << tvals.size() << "\n";
    PrintResults(flag, tvals, yvals);
}
