#include <iostream>

#include "RungeKutta/RungeKutta.hpp"
#include "matplot/matplot.h"

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

    auto rhs = [&](double t, double y) {
        return -2 * y + std::exp(-2 * (t - 6) * (t - 6));
    };

    auto exact_sol = [](const double& t) {
        return 0.25 * std::exp(-2 * t) * (4 + std::sqrt(2 * M_PI) * std::exp(25.0 / 2.0) * (std::erf(13.0 / std::sqrt(2)) + std::erf((-13.0 + 2 * t) / std::sqrt(2))));
    };

    auto solver = RungeKutta45(1.0e-6, 1.0e-10, 1.0e10, 0.0);
    auto [flag, nfev, message, tvals, yvals] = solver.Solve(rhs, {0.0, 15.0}, 1.0);
    std::cout << "nfev: " << nfev << "\n";
    std::cout << "size: " << tvals.size() << "\n";
    PrintResults(flag, tvals, yvals);

    std::vector<double> exact_x = matplot::linspace(tvals.front(), tvals.back(), 300);
    std::vector<double> exact_y = matplot::transform(exact_x, exact_sol);

    auto p = matplot::plot(exact_x, exact_y, tvals, yvals);
    p[0]->line_width(5);
    p[1]->line_width(2)
         .line_style("--")
         .marker(matplot::line_spec::marker_style::circle)
         .marker_size(10);

    matplot::show();

    return 0;
}
