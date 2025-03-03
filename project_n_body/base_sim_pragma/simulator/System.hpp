#pragma once

#include "Particle.hpp"
#include "Utils.hpp"
#include <vector>
#include <string>

class System
{
public:
    System() = default;

    void setParticles(const std::vector<Particle> &particles);
    
    // Updated runSimulation: additional parameters for output filename and metadata.
    void runSimulation(double dt, int steps, double stepFreq,
                       const std::string &outputFilename,
                       const std::string &metadata);

private:
    void velocityVerlet(double dt);
    std::vector<Particle> particles;
    std::vector<std::string> filenames;
};
