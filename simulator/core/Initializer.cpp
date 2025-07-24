#include "Initializer.hpp"
#include <random>
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <array>

// Gravitational constant in astronomical units: AU^3/(Msol * yr^2)
namespace {
    const double G_AU = 39.47841760435743;
}

// Using std namespace consistently to improve readability
using namespace std;

InitResult Initializer::initFromSistemadat(const string &filePath, int count)
{
    ifstream file(filePath);
    if (!file)
        throw runtime_error("Cannot open file: " + filePath);

    // The first line might contain the total number of bodies:
    int total;
    file >> total; // read integer
    if (!file.good())
        throw runtime_error("Failed reading first line from sistema.dat.");

    // Now read up to 'count' bodies from the rest of the file
    Particles p(count);

    for (int i = 0; i < count; ++i)
    {
        double mass, px, py, pz, vx, vy, vz;
        file >> mass >> px >> py >> pz >> vx >> vy >> vz;
        if (!file.good()) break;

        p.setParticle(i,
            mass / 1.989e30,  // Convert mass to Msun
            px / 1.496e8, py / 1.496e8, pz / 1.496e8,  // Convert position to AU
            vx * 0.2108, vy * 0.2108, vz * 0.2108      // Convert velocity to AU/yr
        );
    }

    ostringstream metaStream;
    metaStream << "# Initializer: From sistema.dat\n"
               << "# Total in file: " << total << "\n"
               << "# Using first: " << p.n << "\n"
               << "# Total particle count: " << p.n << "\n"
               << "# filePath: " << filePath << "\n";

    return { p, metaStream.str() };
}

//------------------------------------------------------------------------------
// initGalaxy:
// Creates a central black hole and nStars randomly distributed in a cube
// with side length 2*maxPosition (in AU). Velocities are assigned randomly
// within ±maxVelocity (AU/yr).
// Optionally, if addPlanets is true, each star gets a random number of planets
// with small positional and velocity offsets.
// Units:
//   Distance: AU
//   Mass: Solar Masses (Msun)
//   Velocity: AU/yr
// Default parameters:
//   blackHoleMass = 4e6 Msun
//   starMass range = [0.1, 10] Msun
//   maxPosition   = 330000 AU (~1.6 pc)
//   maxVelocity   = 42.15 AU/yr (indicative value)
InitResult Initializer::initGalaxy(
    int nStars,
    double blackHoleMass = 4e6,   // in Msun
    double starMassMin = 0.1,     // in Msun
    double starMassMax = 10.0,    // in Msun
    double maxPosition = 330000,  // AU (~1.6 pc)
    double maxVelocity = 42.15,   // AU/yr
    bool addPlanets = false,
    int minPlanets = 0,
    int maxPlanets = 0,
    double planetMassMin = 0,     // not used if addPlanets is false
    double planetMassMax = 0,
    double planetDistanceMax = 0,
    double planetVelocityMax = 0
) {
    int capacity = 1 + nStars * (1 + (addPlanets ? maxPlanets : 0));
    Particles particles(capacity);
    size_t currentIndex = 0;

    // Create central black hole at origin with zero velocity
    particles.setParticle(currentIndex++, blackHoleMass, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    random_device rd;
    mt19937 gen(rd());
    // Distributions for stars
    uniform_real_distribution<double> starMassDist(starMassMin, starMassMax);
    uniform_real_distribution<double> posDist(-maxPosition, maxPosition);
    uniform_real_distribution<double> velDist(-maxVelocity, maxVelocity);
    // Distributions for planets (if used)
    uniform_real_distribution<double> planetMassDist(planetMassMin, planetMassMax);
    uniform_real_distribution<double> planetPosOffsetDist(-planetDistanceMax, planetDistanceMax);
    uniform_real_distribution<double> planetVelOffsetDist(-planetVelocityMax, planetVelocityMax);
    uniform_int_distribution<int> planetCountDist(minPlanets, maxPlanets);

    // Create stars
    for (int i = 0; i < nStars; ++i) {
        double mass = starMassDist(gen);
        double px = posDist(gen), py = posDist(gen), pz = posDist(gen);
        double vx = velDist(gen), vy = velDist(gen), vz = velDist(gen);

        particles.setParticle(currentIndex++, mass, px, py, pz, vx, vy, vz);

        // Optionally, add planets orbiting the star
        if (addPlanets) {
            int nPlanetsLocal = planetCountDist(gen);
            for (int j = 0; j < nPlanetsLocal; ++j) {
                double planetMass = planetMassDist(gen);
                double offsetX = planetPosOffsetDist(gen), offsetY = planetPosOffsetDist(gen), offsetZ = planetPosOffsetDist(gen);
                double offsetVx = planetVelOffsetDist(gen), offsetVy = planetVelOffsetDist(gen), offsetVz = planetVelOffsetDist(gen);

                particles.setParticle(
                    currentIndex++,
                    planetMass,
                    px + offsetX, py + offsetY, pz + offsetZ,
                    vx + offsetVx, vy + offsetVy, vz + offsetVz
                );
            }
        }
    }

    // Adjust actual particle count
    if (currentIndex < particles.n) {
        particles.n = currentIndex;
    }

    ostringstream metaStream;
    metaStream << "# Initializer: Galaxy\n";
    metaStream << "# Units: AU (distance), Solar Masses (mass), AU/yr (velocity)\n";
    metaStream << "# nStars: " << nStars << "\n";
    metaStream << "# blackHoleMass: " << blackHoleMass << " Msun\n";
    metaStream << "# starMass range: [" << starMassMin << ", " << starMassMax << "] Msun\n";
    metaStream << "# maxPosition: " << maxPosition << " AU\n";
    metaStream << "# maxVelocity: " << maxVelocity << " AU/yr\n";
    metaStream << "# addPlanets: " << boolalpha << addPlanets << "\n";
    if (addPlanets) {
        metaStream << "# Planets per star: [" << minPlanets << ", " << maxPlanets << "]\n";
        metaStream << "# planetMass range: [" << planetMassMin << ", " << planetMassMax << "] Msun\n";
        metaStream << "# planetDistanceMax: " << planetDistanceMax << " AU\n";
        metaStream << "# planetVelocityMax: " << planetVelocityMax << " AU/yr\n";
    }
    metaStream << "# Total particle count: " << particles.n << "\n";

    return { particles, metaStream.str() };
}

//------------------------------------------------------------------------------
// initRandomParticles:
// Creates n random particles within a cubic box of side L (in AU)
// with masses in the range [minMass, maxMass] (in Msun) and
// velocities in the range ±maxVelocity (AU/yr). It ensures that
// particles are at least minDistance apart (in AU).
// Default parameters (in astronomical units):
//   minMass      = 1e-10 Msun
//   maxMass      = 1e-5 Msun
//   L            = 1e3 AU
//   maxVelocity  = 1 AU/yr
//   minDistance  = 1e-3 AU
InitResult Initializer::initRandomParticles(
    int n,
    double minMass,
    double maxMass,
    double L,
    double maxVelocity,
    double minDistance
) {
    Particles p(n);
    random_device rd;
    mt19937 gen(rd());
    
    uniform_real_distribution<double> xDist(-L/2, L/2);
    uniform_real_distribution<double> yDist(-L/2, L/2);
    uniform_real_distribution<double> zDist(-L/2, L/2);
    uniform_real_distribution<double> vDist(-maxVelocity, maxVelocity);
    uniform_real_distribution<double> mDist(minMass, maxMass);
    
    const int maxRetries = 1000;
    for (int i = 0; i < n; ++i) {
        double mass, px, py, pz, vx, vy, vz;
        bool valid = false;
        int retries = 0;
        
        while (!valid) {
            mass = mDist(gen);
            px = xDist(gen);
            py = yDist(gen);
            pz = zDist(gen);
            vx = vDist(gen);
            vy = vDist(gen);
            vz = vDist(gen);
            
            valid = true;
            // Check distance from already placed particles if minDistance > 0
            if (minDistance > 0.0) {
                // Fix signedness mismatch here: use int j to match int i
                for (int j = 0; j < i; ++j) {
                    double dx = px - p.posMass[j].x;
                    double dy = py - p.posMass[j].y;
                    double dz = pz - p.posMass[j].z;
                    double dist = sqrt(dx*dx + dy*dy + dz*dz);
                    if (dist < minDistance) {
                        valid = false;
                        break;
                    }
                }
            }
            
            if (!valid && ++retries > maxRetries)
                throw runtime_error("Failed to place particle " + to_string(i));
        }
        
        // Use the setParticle method to set values in our double4 structure
        p.setParticle(i, mass, px, py, pz, vx, vy, vz);
    }
    
    ostringstream metadata;
    metadata << "# Random particles\n";
    metadata << "# Number of particles: " << n << "\n";
    metadata << "# Box size (AU): " << L << "\n";
    metadata << "# Velocity range (AU/year): [-" << maxVelocity << ", " << maxVelocity << "]\n";
    metadata << "# Mass range (solar masses): [" << minMass << ", " << maxMass << "]\n";
    if (minDistance > 0.0) {
        metadata << "# Minimum distance between particles: " << minDistance << " AU\n";
    }
    metadata << "# Total particle count: " << p.n << "\n";
    metadata << "# Format: time,x,y,z,vx,vy,vz,energy\n";
    metadata << "# Units: years, AU, AU/year, energy/(solar mass)\n";
    
    return {p, metadata.str()};
}

//----------------------------------------------------
// initStellarSystem: A central star with orbiting planets and moons.
// All orbits are calculated in the XY plane using astronomical units.
//
// Default parameters (in AU, Msol, yr):
// - starMass: 1.0 Msol
// - nPlanets: 8
// - planetMass: [3e-6, 1e-3] Msol (Earth ~3e-6, Jupiter ~1e-3)
// - planetDistance: [0.4, 30] AU
// - For each planet, orbital speed: v = sqrt(G_AU * starMass / r) [AU/yr]
//
// - Moons (optional):
//   - moons per planet: [0, 3]
//   - moonMass: [1e-8, 1e-5] Msol
//   - moonDistance: [0.001, 0.01] AU  (typical satellite orbit radius)
//   - Orbital speed around the planet: v = sqrt(G_AU * planetMass / r)
InitResult Initializer::initStellarSystem(
    double starMass = 1.0,                // in Msol
    int nPlanets = 8,
    double planetMassMin = 3e-6, double planetMassMax = 1e-3,  // in Msol
    double minPlanetDistance = 0.4, double maxPlanetDistance = 30.0, // in AU
    bool addMoons = true,
    int minMoons = 0, int maxMoons = 3,
    double moonMassMin = 1e-8, double moonMassMax = 1e-5,  // in Msol
    double moonDistanceMin = 0.001, double moonDistanceMax = 0.01 // in AU
)
{
    int capacity = 1 + nPlanets * (1 + (addMoons ? maxMoons : 0));
    Particles soA(capacity);
    size_t currentIndex = 0;

    random_device rd;
    mt19937 gen(rd());
    
    // Distributions for planet parameters
    uniform_real_distribution<double> planetMassDist(planetMassMin, planetMassMax);
    uniform_real_distribution<double> planetDistanceDist(minPlanetDistance, maxPlanetDistance);
    uniform_real_distribution<double> angleDist(0.0, 2 * M_PI);
    
    // Distributions for moon parameters
    uniform_int_distribution<int> moonCountDist(minMoons, maxMoons);
    uniform_real_distribution<double> moonMassDist(moonMassMin, moonMassMax);
    uniform_real_distribution<double> moonDistanceDist(moonDistanceMin, moonDistanceMax);
    uniform_real_distribution<double> moonAngleDist(0.0, 2 * M_PI);
    
    // Create central star at origin with zero velocity.
    soA.setParticle(currentIndex++, starMass, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    // For each planet.
    for (int i = 0; i < nPlanets; ++i)
    {
        double pmass = planetMassDist(gen);
        double distance = planetDistanceDist(gen);
        double theta = angleDist(gen);
        double px = distance * cos(theta);
        double py = distance * sin(theta);
        double pz = 0.0;
        double orbitalSpeed = sqrt(G_AU * starMass / distance);
        double vx = -orbitalSpeed * sin(theta);
        double vy =  orbitalSpeed * cos(theta);
        double vz = 0.0;

        soA.setParticle(currentIndex++, pmass, px, py, pz, vx, vy, vz);

        // Optionally add moons orbiting around the planet.
        if (addMoons)
        {
            int nMoonsHere = moonCountDist(gen);
            for (int j = 0; j < nMoonsHere; ++j)
            {
                double mmass = moonMassDist(gen);
                double mdist = moonDistanceDist(gen);
                double phi = moonAngleDist(gen);
                double mx = px + mdist * cos(phi);
                double my = py + mdist * sin(phi);
                double mz = pz; 
                double moonOrbSpeed = sqrt(G_AU * pmass / mdist);
                double mvx = vx - moonOrbSpeed * sin(phi);
                double mvy = vy + moonOrbSpeed * cos(phi);
                double mvz = vz;

                soA.setParticle(currentIndex++, mmass, mx, my, mz, mvx, mvy, mvz);
            }
        }
    }
    
    if (currentIndex < soA.n) {
        soA.n = currentIndex;
    }

    ostringstream metaStream;
    metaStream << "# Initializer: Stellar System \n";
    metaStream << "# starMass: " << starMass << " Msol\n";
    metaStream << "# nPlanets: " << nPlanets << "\n";
    metaStream << "# planetMass range: [" << planetMassMin << ", " << planetMassMax << "] Msol\n";
    metaStream << "# planetDistance range: [" << minPlanetDistance << ", " << maxPlanetDistance << "] AU\n";
    metaStream << "# addMoons: " << boolalpha << addMoons << "\n";
    if (addMoons)
    {
        metaStream << "# moons per planet: [" << minMoons << ", " << maxMoons << "]\n";
        metaStream << "# moonMass range: [" << moonMassMin << ", " << moonMassMax << "] Msol\n";
        metaStream << "# moonDistance range: [" << moonDistanceMin << ", " << moonDistanceMax << "] AU\n";
    }
    metaStream << "# Total particle count: " << soA.n << "\n";

    return { soA, metaStream.str() };
}

//----------------------------------------------------
// initSpiralGalaxy: Initializes a spiral galaxy with stars distributed along spiral arms.
// The disk orientation is random so you possono generare galassie con piani diversi (utile per collisioni).
//
// Default parameters (astronomical units):
// - nStars: 10000
// - totalMass: 1e11 Msol (galaxy mass)
// - nArms: 2
// - galaxyRadius: 15 kpc in AU (1 pc = 206265 AU; 15 kpc = 15e3*206265 ≈ 3.09398e9 AU)
// - thickness: 0.3 kpc in AU (≈ 0.3e3*206265 ≈ 6.18795e8 AU)
// - perturbation: 5 AU/yr (small random velocity perturbation)
InitResult Initializer::initSpiralGalaxy(
    int nStars = 10000,
    double totalMass = 1e11,   // in Msol
    int nArms = 2,
    double galaxyRadius = 3.09398e9, // in AU (~15 kpc)
    double thickness = 6.18795e8,    // in AU (~0.3 kpc)
    double perturbation = 5.0        // in AU/yr
)
{
    Particles soA(nStars);

    random_device rd;
    mt19937 gen(rd());
    // Distribution for radial distance: for uniform area density use r = galaxyRadius * sqrt(u)
    uniform_real_distribution<double> uDist(0.0, 1.0);
    // Base random angle.
    uniform_real_distribution<double> angleDist(0.0, 2 * M_PI);
    // Small offset for spiral arm deviation.
    uniform_real_distribution<double> armOffsetDist(-M_PI/8, M_PI/8);
    // Distribution for thickness (vertical offset).
    uniform_real_distribution<double> thicknessDist(-thickness/2, thickness/2);
    // Distribution for velocity perturbation.
    uniform_real_distribution<double> velPerturbDist(-perturbation, perturbation);

    // Compute base circular velocity at galaxyRadius: v0 = sqrt(G_AU * totalMass / galaxyRadius)
    double v0 = sqrt(G_AU * totalMass / galaxyRadius);
    
    // Choose a random disk orientation by generating a random unit normal vector.
    uniform_real_distribution<double> thetaDist(0, M_PI);
    uniform_real_distribution<double> phiDist(0, 2 * M_PI);
    double thetaN = thetaDist(gen);
    double phiN = phiDist(gen);
    array<double, 3> n = {
        sin(thetaN) * cos(phiN),
        sin(thetaN) * sin(phiN),
        cos(thetaN)
    };
    // Build an arbitrary basis (e1, e2) for the disk plane.
    array<double, 3> e1;
    if (fabs(n[0]) < 1e-6 && fabs(n[1]) < 1e-6) {
        e1 = {1, 0, 0};
    } else {
        e1 = { -n[1], n[0], 0 };
        double norm = sqrt(e1[0]*e1[0] + e1[1]*e1[1] + e1[2]*e1[2]);
        e1[0] /= norm; e1[1] /= norm; e1[2] /= norm;
    }
    // e2 = n cross e1
    array<double, 3> e2 = {
        n[1]*e1[2] - n[2]*e1[1],
        n[2]*e1[0] - n[0]*e1[2],
        n[0]*e1[1] - n[1]*e1[0]
    };

    // For each star.
    for (int i = 0; i < nStars; ++i)
    {
        double massVal = totalMass / nStars;
        // Generate radial coordinate r with uniform area density.
        double u = uDist(gen);
        double r = galaxyRadius * sqrt(u);
        // Determine base angle from spiral arm pattern.
        int armIndex = i % nArms;
        double baseArmAngle = (2 * M_PI * armIndex) / nArms;
        // Add spiral twist proportional to r.
        double spiralTwist = (r / galaxyRadius) * 2 * M_PI;
        double phi = baseArmAngle + spiralTwist + armOffsetDist(gen);

        // Position in disk coordinates.
        double xDisk = r * cos(phi);
        double yDisk = r * sin(phi);
        double zDisk = thicknessDist(gen); // vertical offset

        // Convert disk coordinates to global coordinates.
        double globalX = xDisk * e1[0] + yDisk * e2[0] + zDisk * n[0];
        double globalY = xDisk * e1[1] + yDisk * e2[1] + zDisk * n[1];
        double globalZ = xDisk * e1[2] + yDisk * e2[2] + zDisk * n[2];

        // Compute orbital velocity using a simplified rotation curve:
        // v = v0 * sqrt(r/galaxyRadius)
        double v = v0 * sqrt(r / galaxyRadius);
        // Tangential unit vector in disk coordinates: (-sin(phi), cos(phi), 0)
        double tx = -sin(phi);
        double ty =  cos(phi);
        //double tz = 0.0;
        // Convert tangential vector to global coordinates.
        double velX = tx * e1[0] + ty * e2[0];
        double velY = tx * e1[1] + ty * e2[1];
        double velZ = tx * e1[2] + ty * e2[2];
        // Add small velocity perturbation.
        double vx = v * velX + velPerturbDist(gen);
        double vy = v * velY + velPerturbDist(gen);
        double vz = v * velZ + velPerturbDist(gen);

        soA.setParticle(i, massVal, globalX, globalY, globalZ, vx, vy, vz);
    }

    ostringstream metaStream;
    metaStream << "# Initializer: Spiral Galaxy\n";
    metaStream << "# nStars: " << nStars << "\n";
    metaStream << "# totalMass: " << totalMass << " Msol\n";
    metaStream << "# nArms: " << nArms << "\n";
    metaStream << "# galaxyRadius: " << galaxyRadius << " AU\n";
    metaStream << "# thickness: " << thickness << " AU\n";
    metaStream << "# base circular velocity at galaxyRadius: " << v0 << " AU/yr\n";
    metaStream << "# random disk orientation (unit normal): ("
               << n[0] << ", " << n[1] << ", " << n[2] << ")\n";
    metaStream << "# Total particle count: " << soA.n << "\n";

    return { soA, metaStream.str() };
}
