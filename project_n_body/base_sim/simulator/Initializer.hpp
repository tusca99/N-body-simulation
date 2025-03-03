#pragma once
#include <vector>
#include <string>
#include "Particle.hpp"

// Structure to hold initialization result and metadata
struct InitResult {
    std::vector<Particle> particles;
    std::string metadata;
};

class Initializer {
public:
    // Generates random particles and returns the InitResult including metadata.
    static InitResult initRandomParticles(int n,
                                            double minMass,
                                            double maxMass,
                                            double L,
                                            double maxVelocity,
                                            double minDistance);

    // Generates a galaxy with a central black hole, stars and optional planets.
    static InitResult initGalaxy(int nStars,
                                 double blackHoleMass,
                                 double starMassMin,
                                 double starMassMax,
                                 double maxPosition,
                                 double maxVelocity,
                                 bool addPlanets,
                                 int minPlanets,
                                 int maxPlanets,
                                 double planetMassMin,
                                 double planetMassMax,
                                 double planetDistanceMax,
                                 double planetVelocityMax);
};