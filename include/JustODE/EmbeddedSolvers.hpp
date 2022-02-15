#pragma once

#include <limits>

#include "Utils.hpp"
#include "RungeKuttaBase.hpp"

namespace JustODE {

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
     * @param[in] hmin Min step size. Default is zero.
     *********************************************************************/
    RKF45(
        const T& atol = T(1.0e-6), const T& rtol = T(1.0e-3),
        const T& hmax = std::numeric_limits<T>::max(), const T& hmin = T(0)
    ) : RungeKuttaBase<RKF45<T>, T, 4, 6>(atol, rtol, hmax, hmin) {}

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
 * @brief Dormand–Prince 5(4) method.
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
     * @param[in] hmin Min step size. Default is zero.
     *********************************************************************/
    DOPRI54(
        const T& atol = T(1.0e-6), const T& rtol = T(1.0e-3),
        const T& hmax = std::numeric_limits<T>::max(), const T& hmin = T(0)
    ) : RungeKuttaBase<DOPRI54<T>, T, 4, 6>(atol, rtol, hmax, hmin) {}

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