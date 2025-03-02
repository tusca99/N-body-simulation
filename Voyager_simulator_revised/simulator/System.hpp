//// filepath: /home/alessio/ssd_data/Alessio/uni magistrale/Modern_Computing_Physics/project_n_body/object/System.hpp
#pragma once

#include "Particle.hpp"
#include "Utils.hpp"  // for ParticleVel definition
#include <vector>
#include <string>

class System
{
public:
    System() = default;

    // We no longer handle file I/O directly in this class
    // so we remove loadParticles, loadNASAData, etc.

    // Instead, pass in data from outside
    void setParticles(const std::vector<Particle> &particles);
    void setNASAData(const std::vector<ParticleVel> &data);
    void setFilenames(const std::vector<std::string> &files);

    void runSimulation(double dt, int steps, double stepFreq, bool correction, double sv);

private:
    // Core simulation routines
    void velocityVerlet(double dt);
    void correctVelocity(int iteration, int stepFreq, double sv);

    // Data
    std::vector<Particle> particles;
    std::vector<ParticleVel> Vpv;
    std::vector<std::string> filenames;
};