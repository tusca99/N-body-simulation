#include "System.hpp"
#include <iostream>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <sstream>
#include <fstream>
#include <vector>
#include <cstdio>
#include <omp.h>

namespace
{
    constexpr double G = 6.67430e-11;

    Particle centerOfMass(const std::vector<Particle> &s)
    {
        Particle cm;
        for (const auto &part : s) cm.m += part.m;
        for (const auto &part : s)
        {
            for (int k = 0; k < 3; ++k)
            {
                cm.r[k] += part.m * part.r[k];
                cm.v[k] += part.m * part.v[k];
            }
        }
        for (int k = 0; k < 3; ++k)
        {
            cm.r[k] /= cm.m;
            cm.v[k] /= cm.m;
        }
        return cm;
    }

    std::array<double, 3> computeForce(const std::vector<Particle> &s, const Particle &pi)
    {
        std::array<double, 3> totalForce{0.0, 0.0, 0.0};
        #pragma omp parallel for reduction(+: totalForce[0], totalForce[1], totalForce[2])
        for (int idx = 0; idx < static_cast<int>(s.size()); ++idx)
        {
            const auto &other = s[idx];
            if (&other == &pi) continue;
            std::array<double, 3> diff;
            for (int k = 0; k < 3; ++k)
                diff[k] = other.r[k] - pi.r[k];
            double dist = std::sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]);
            if (dist <= 0.0) continue;
            double forceVal = (G * other.m * pi.m) / (dist * dist * dist);
            for (int k = 0; k < 3; ++k)
            {
                totalForce[k] += forceVal * diff[k];
            }
        }
        return totalForce;
    }

    void flushUnifiedOutput(const std::vector<std::vector<std::string>> &outStates,
                            const std::string &filename,
                            const std::string &metadata)
    {
        std::ofstream unifiedFile(filename);
        if (!unifiedFile)
        {
            std::cerr << "Could not open unified output file " << filename << " for writing.\n";
            return;
        }
        // Write header and metadata.
        unifiedFile << "# Unified simulation data file\n";
        unifiedFile << "# Metadata:\n" << metadata << "\n\n";
        unifiedFile << "# Format: particle_id time r0 r1 r2 v0 v1 v2 modr modv\n\n";
        for (std::size_t id = 0; id < outStates.size(); ++id)
        {
            for (const auto &line : outStates[id])
            {
                unifiedFile << id << " " << line << "\n";
            }
        }
    }
}

void System::setParticles(const std::vector<Particle> &p)
{
    particles = p;
}

void System::runSimulation(double dt, int steps, double stepFreq,
                           const std::string &outputFilename,
                           const std::string &metadata)
{
    // Remove any old output file if exists.
    std::remove(outputFilename.c_str());
    
    std::vector<std::vector<std::string>> outputStates(particles.size());
    int approxLines = 1 + (stepFreq > 0 ? static_cast<int>(steps/stepFreq) : 0);
    for (auto &states : outputStates)
        states.reserve(approxLines);

    auto formatState = [&](double t, const Particle &p) -> std::string {
        std::ostringstream oss;
        double modr = std::sqrt(p.r[0]*p.r[0] + p.r[1]*p.r[1] + p.r[2]*p.r[2]);
        double modv = std::sqrt(p.v[0]*p.v[0] + p.v[1]*p.v[1] + p.v[2]*p.v[2]);
        oss << t << " " 
            << p.r[0] << " " << p.r[1] << " " << p.r[2] << " "
            << p.v[0] << " " << p.v[1] << " " << p.v[2] << " "
            << modr << " " << modv;
        return oss.str();
    };
    
    for (std::size_t i = 0; i < particles.size(); ++i)
        outputStates[i].push_back(formatState(0.0, particles[i]));

    int printCounter = 0;
    for (int i = 0; i < steps; ++i)
    {
        velocityVerlet(dt);
        if (printCounter == static_cast<int>(stepFreq))
        {
            double currentTime = i * dt;
            for (std::size_t j = 0; j < particles.size(); ++j)
                outputStates[j].push_back(formatState(currentTime, particles[j])); // saving only each stepFreq-th state
            printCounter = 0;
        }
        printCounter++;
    }
    
    flushUnifiedOutput(outputStates, outputFilename, metadata);
}

void System::velocityVerlet(double dt)
{
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(particles.size()); ++i)
    {
        // Update positions
        for (int k = 0; k < 3; ++k)
            particles[i].r[k] += particles[i].v[k] * dt;

        // Compute forces
        auto force = computeForce(particles, particles[i]);

        // Update velocities
        for (int k = 0; k < 3; ++k)
            particles[i].v[k] += force[k] / particles[i].m * dt;
    }
}