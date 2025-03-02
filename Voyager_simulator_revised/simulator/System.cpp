//// filepath: /home/alessio/ssd_data/Alessio/uni magistrale/Modern_Computing_Physics/project_n_body/object/System.cpp
#include "System.hpp"
#include <iostream>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace
{
    constexpr double G = 6.67430e-11;

    // Compute the center of mass for all particles in the system
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

    // Compute the gravitational force on a single particle from the rest of the system
    std::array<double, 3> computeForce(const std::vector<Particle> &s, const Particle &pi)
    {
        std::array<double, 3> totalForce{0.0, 0.0, 0.0};
        for (const auto &other : s)
        {
            // Avoid self-force
            if (&other == &pi) continue;

            std::array<double, 3> diff;
            for (int k = 0; k < 3; ++k)
                diff[k] = other.r[k] - pi.r[k];

            double dist = std::sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]);
            if (dist <= 0.0) continue; // prevent division by zero

            double forceVal = (G * other.m * pi.m) / (dist * dist * dist);
            for (int k = 0; k < 3; ++k)
            {
                totalForce[k] += forceVal * diff[k];
            }
        }
        return totalForce;
    }
}

void System::setParticles(const std::vector<Particle> &p)
{
    particles = p;
    filenames = generateFilenames(particles);
}

void System::setNASAData(const std::vector<ParticleVel> &data)
{
    Vpv = data;
}

void System::setFilenames(const std::vector<std::string> &files)
{
    filenames = files;
}

void System::runSimulation(double dt, int steps, double stepFreq, bool correction, double sv)
{
    // For example, do file cleanup + first write
    removeOldDataFiles(filenames);
    appendStateToFiles(particles, filenames, 0.0);

    auto startTime = std::chrono::high_resolution_clock::now();
    int printCounter = 0;

    for (int i = 0; i < steps; ++i)
    {
        velocityVerlet(dt);

        if (printCounter == static_cast<int>(stepFreq))
        {
            double currentTime = i * dt;
            appendStateToFiles(particles, filenames, currentTime);
            printCounter = 0;

            if (correction)
            {
                correctVelocity(i, static_cast<int>(stepFreq), sv);
            }
        }
        printCounter++;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
    std::cout << "Simulation finished in " << seconds << " seconds\n";
}

void System::velocityVerlet(double dt)
{
    // Implement the velocity Verlet integration method
    // This is a placeholder implementation
    for (auto &p : particles)
    {
        // Update positions
        for (int k = 0; k < 3; ++k)
        {
            p.r[k] += p.v[k] * dt;
        }

        // Compute forces
        auto force = computeForce(particles, p);

        // Update velocities
        for (int k = 0; k < 3; ++k)
        {
            p.v[k] += force[k] / p.m * dt;
        }
    }
}

void System::correctVelocity(int iteration, int stepFreq, double sv)
{
    // Implement the velocity correction method
    // This is a placeholder implementation
    if (iteration % stepFreq == 0)
    {
        for (auto &p : particles)
        {
            for (int k = 0; k < 3; ++k)
            {
                p.v[k] *= sv;
            }
        }
    }
}