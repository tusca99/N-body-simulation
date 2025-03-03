#include "Initializer.hpp"
#include <random>
#include <sstream>
#include <cmath>
#include <stdexcept>

using namespace std;

InitResult Initializer::initRandomParticles(int n,
                                              double minMass,
                                              double maxMass,
                                              double L,
                                              double maxVelocity,
                                              double minDistance)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> massDist(minMass, maxMass);
    uniform_real_distribution<double> posDist(-L/2.0, L/2.0);
    uniform_real_distribution<double> velDist(-maxVelocity, maxVelocity);
    
    vector<Particle> particles;
    particles.reserve(n);
    const int maxRetries = 1000;
    
    for (int i = 0; i < n; ++i)
    {
        Particle p;
        p.m = massDist(gen);
        int retries = 0;
        bool valid = false;
        while (!valid)
        {
            for (int k = 0; k < 3; ++k)
            {
                p.r[k] = posDist(gen);
                p.v[k] = velDist(gen);
            }
            valid = true;
            for (const auto &other : particles)
            {
                double distSquared = 0.0;
                for (int k = 0; k < 3; ++k)
                {
                    double diff = p.r[k] - other.r[k];
                    distSquared += diff * diff;
                }
                if (sqrt(distSquared) < minDistance)
                {
                    valid = false;
                    break;
                }
            }
            if (!valid && ++retries > maxRetries)
                throw runtime_error("Failed to place particle " + to_string(i) + " with required separation.");
        }
        particles.push_back(p);
    }
    
    ostringstream metaStream;
    metaStream << "Initializer: Random\n";
    metaStream << "nParticles: " << n << "\n";
    metaStream << "minMass: "    << minMass << "\n";
    metaStream << "maxMass: "    << maxMass << "\n";
    metaStream << "Box side (L): " << L << "\n";
    metaStream << "maxVelocity: " << maxVelocity << "\n";
    metaStream << "minDistance: " << minDistance << "\n";
    
    return { particles, metaStream.str() };
}

InitResult Initializer::initGalaxy(int nStars,
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
                                   double planetVelocityMax)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> starMassDist(starMassMin, starMassMax);
    uniform_real_distribution<double> posDist(-maxPosition, maxPosition);
    uniform_real_distribution<double> velDist(-maxVelocity, maxVelocity);
    uniform_int_distribution<int> planetCountDist(minPlanets, maxPlanets);
    uniform_real_distribution<double> planetMassDist(planetMassMin, planetMassMax);
    uniform_real_distribution<double> planetPosOffset(-planetDistanceMax, planetDistanceMax);
    uniform_real_distribution<double> planetVelOffset(-planetVelocityMax, planetVelocityMax);
    
    vector<Particle> particles;
    particles.reserve(1 + nStars * (1 + (addPlanets ? maxPlanets : 0)));
    
    // Create central black hole
    {
        Particle blackHole;
        blackHole.m = blackHoleMass;
        blackHole.r = {0.0, 0.0, 0.0};
        blackHole.v = {0.0, 0.0, 0.0};
        particles.push_back(blackHole);
    }
    
    for (int i = 0; i < nStars; ++i)
    {
        Particle star;
        star.m = starMassDist(gen);
        for (int k = 0; k < 3; ++k)
        {
            star.r[k] = posDist(gen);
            star.v[k] = velDist(gen);
        }
        particles.push_back(star);
        if (addPlanets)
        {
            int numPlanets = planetCountDist(gen);
            for (int j = 0; j < numPlanets; ++j)
            {
                Particle planet;
                planet.m = planetMassDist(gen);
                for (int k = 0; k < 3; ++k)
                {
                    planet.r[k] = star.r[k] + planetPosOffset(gen);
                    planet.v[k] = star.v[k] + planetVelOffset(gen);
                }
                particles.push_back(planet);
            }
        }
    }
    
    ostringstream metaStream;
    metaStream << "Initializer: Galaxy\n";
    metaStream << "nStars: " << nStars << "\n";
    metaStream << "blackHoleMass: " << blackHoleMass << "\n";
    metaStream << "starMassMin: " << starMassMin << "\n";
    metaStream << "starMassMax: " << starMassMax << "\n";
    metaStream << "maxPosition: " << maxPosition << "\n";
    metaStream << "maxVelocity: " << maxVelocity << "\n";
    metaStream << "addPlanets: " << std::boolalpha << addPlanets << "\n";
    if(addPlanets)
        metaStream << "minPlanets: " << minPlanets << "\n"
                   << "maxPlanets: " << maxPlanets << "\n"
                   << "planetMassMin: " << planetMassMin << "\n"
                   << "planetMassMax: " << planetMassMax << "\n"
                   << "planetDistanceMax: " << planetDistanceMax << "\n"
                   << "planetVelocityMax: " << planetVelocityMax << "\n";
    
    return { particles, metaStream.str() };
}