#pragma once
#include <vector>
#include <string>
#include "Particles.hpp"

enum class InitializerMethod {
    FROM_FILE,
    RANDOM,
    GALAXY,
    STELLAR_SYSTEM,
    SPIRAL_GALAXY
};

// Structure to hold initialization result and metadata
struct InitResult {
    Particles particles;
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

    // New method: read up to 'count' bodies from a local file (sistema.dat).
    static InitResult initFromSistemadat(const std::string &filePath, int count);

    // New method: generate a stellar system with a central star, planets and optional moons.
    static InitResult initStellarSystem(double starMass,
                                        int nPlanets,
                                        double planetMassMin,
                                        double planetMassMax,
                                        double minPlanetDistance,
                                        double maxPlanetDistance,
                                        bool addMoons,
                                        int minMoons,
                                        int maxMoons,
                                        double moonMassMin,
                                        double moonMassMax,
                                        double moonDistanceMin,
                                        double moonDistanceMax);
                    
    static InitResult initSpiralGalaxy(int nStars,
                                        double totalMass,   // in Msol
                                        int nArms,
                                        double galaxyRadius, // in AU (~15 kpc)
                                        double thickness,    // in AU (~0.3 kpc)
                                        double perturbation);        // in AU/yr);
};


