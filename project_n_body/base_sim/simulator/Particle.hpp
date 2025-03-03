#pragma once
#include <array>

struct Particle
{
    double m{0.0};
    std::array<double, 3> r{0.0, 0.0, 0.0};
    std::array<double, 3> v{0.0, 0.0, 0.0};
};