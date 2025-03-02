//// filepath: /home/alessio/ssd_data/Alessio/uni magistrale/Modern_Computing_Physics/project_n_body/VoyagerII/ModernVoyagercorr.cpp
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdio>
#include <stdexcept>

struct Particle
{
    double m{0.0};
    std::array<double, 3> r{0.0, 0.0, 0.0};
    std::array<double, 3> v{0.0, 0.0, 0.0};
};

struct ParticleVel
{
    std::array<double, 3> r{0.0, 0.0, 0.0};
    std::array<double, 3> v{0.0, 0.0, 0.0};
};

// Utility function to safely read a Particle from file
Particle readParticle(std::ifstream &file)
{
    Particle p;
    if (!(file >> p.m)) throw std::runtime_error("Error reading particle mass");

    for (int i = 0; i < 3; ++i)
    {
        if (!(file >> p.r[i])) throw std::runtime_error("Error reading particle position");
    }
    for (int i = 0; i < 3; ++i)
    {
        if (!(file >> p.v[i])) throw std::runtime_error("Error reading particle velocity");
    }
    // Convert from km, km/s to SI units
    for (int i = 0; i < 3; ++i)
    {
        p.r[i] *= 1000.0; 
        p.v[i] *= 1000.0;
    }
    return p;
}

// Initialize system from file
std::vector<Particle> initSystem(std::ifstream &file)
{
    int n = 0;
    if (!(file >> n)) throw std::runtime_error("Error reading number of particles");

    std::vector<Particle> s;
    s.reserve(n);
    for (int i = 0; i < n; ++i)
    {
        s.push_back(readParticle(file));
    }
    return s;
}

// Sync particles: create a copy
std::vector<Particle> syncParticles(const std::vector<Particle> &s)
{
    return s; // straightforward copy
}

// Find center of mass
Particle centerOfMass(const std::vector<Particle> &s)
{
    Particle cm;
    // Sum masses
    for (const auto &part : s) cm.m += part.m;

    // Weighted sum of positions, velocities
    for (const auto &part : s)
    {
        for (int k = 0; k < 3; ++k)
        {
            cm.r[k] += part.m * part.r[k];
            cm.v[k] += part.m * part.v[k];
        }
    }
    // Divide to get average
    for (int k = 0; k < 3; ++k)
    {
        cm.r[k] /= cm.m;
        cm.v[k] /= cm.m;
    }
    return cm;
}

// Compute gravitational force on pi from the rest of the system
std::array<double, 3> computeForce(const std::vector<Particle> &s, const Particle &pi)
{
    constexpr double G = 6.67430e-11;
    std::array<double, 3> totalForce{0.0, 0.0, 0.0};

    for (const auto &other : s)
    {
        // Ensure we don't compute force from pi itself
        if (other.m == pi.m && other.r == pi.r && other.v == pi.v) continue;

        std::array<double, 3> diff;
        for (int k = 0; k < 3; ++k) diff[k] = other.r[k] - pi.r[k];
        double dist = std::sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);

        // Avoid division by zero
        if (dist <= 0.0) continue;

        double f = (G * other.m * pi.m) / (dist * dist * dist);
        for (int k = 0; k < 3; ++k)
        {
            totalForce[k] += f * diff[k];
        }
    }
    return totalForce;
}

// Velocity Verlet integrator
std::vector<Particle> velocityVerlet(std::vector<Particle> &s, double dt)
{
    Particle cm = centerOfMass(s);
    std::vector<std::array<double, 3>> forces;
    forces.reserve(s.size());

    // First half-step
    for (std::size_t i = 0; i < s.size(); ++i)
    {
        // Shift velocity by center-of-mass velocity
        for (int k = 0; k < 3; ++k)
        {
            s[i].v[k] -= cm.v[k];
        }
        auto f = computeForce(s, s[i]);
        forces.push_back(f);
        // Update position
        for (int k = 0; k < 3; ++k)
        {
            s[i].r[k] += s[i].v[k] * dt + (0.5 / s[i].m) * f[k] * dt * dt;
        }
    }
    // Second half-step
    for (std::size_t i = 0; i < s.size(); ++i)
    {
        auto fNew = computeForce(s, s[i]);
        for (int k = 0; k < 3; ++k)
        {
            s[i].v[k] += (0.5 / s[i].m) * (forces[i][k] + fNew[k]) * dt;
        }
    }
    return s;
}

// Generate filenames
std::vector<std::string> generateFilenames(const std::vector<Particle> &s)
{
    std::vector<std::string> fnames;
    fnames.reserve(s.size());
    for (std::size_t i = 0; i < s.size(); ++i)
    {
        char buffer[30];
        std::sprintf(buffer, "object%zu.dat", i);
        fnames.emplace_back(buffer);
    }
    return fnames;
}

// Append system state to files
void appendStateToFiles(const std::vector<Particle> &s, const std::vector<std::string> &fnames, double t)
{
    for (std::size_t i = 0; i < s.size(); ++i)
    {
        std::ofstream file(fnames[i], std::ios_base::app);
        double modr = std::sqrt(s[i].r[0]*s[i].r[0] + s[i].r[1]*s[i].r[1] + s[i].r[2]*s[i].r[2]);
        double modv = std::sqrt(s[i].v[0]*s[i].v[0] + s[i].v[1]*s[i].v[1] + s[i].v[2]*s[i].v[2]);
        file << t << " "
             << s[i].r[0] << " " << s[i].r[1] << " " << s[i].r[2] << " "
             << s[i].v[0] << " " << s[i].v[1] << " " << s[i].v[2] << " "
             << modr << " " << modv << "\n";
    }
}

// Remove old data files
void removeOldFiles(const std::vector<std::string> &fnames)
{
    for (auto &fname : fnames)
    {
        try
        {
            if (std::filesystem::remove(fname))
                std::cout << "file " << fname << " deleted.\n";
            else
                std::cout << "file " << fname << " not found.\n";
        }
        catch (const std::filesystem::filesystem_error &err)
        {
            std::cout << "filesystem error: " << err.what() << '\n';
        }
    }
}

// Check velocity difference vs NASA data
bool checkVelocity(const std::vector<Particle> &s, const std::vector<ParticleVel> &Vpv, int i, double step, double sv)
{
    // Compare last particle velocity with NASA data
    const auto &last = s.back(); 
    auto index = static_cast<std::size_t>(i / step);
    if (index >= Vpv.size()) return false;

    const auto &nasaV = Vpv[index].v;
    double vxRel = std::fabs(last.v[0] - nasaV[0]) / std::fabs(nasaV[0]);
    double vyRel = std::fabs(last.v[1] - nasaV[1]) / std::fabs(nasaV[1]);
    double vzRel = std::fabs(last.v[2] - nasaV[2]) / std::fabs(nasaV[2]);

    return (vxRel < sv && vyRel < sv && vzRel < sv);
}

// Potential energy of a single body
double potentialEnergy(const std::vector<Particle> &s, const Particle &pi)
{
    constexpr double G = 6.67430e-11;
    double U = 0.0;

    for (const auto &other : s)
    {
        if (other.m == pi.m && other.r == pi.r && other.v == pi.v) continue;

        std::array<double, 3> diff;
        for (int k = 0; k < 3; ++k)
        {
            diff[k] = other.r[k] - pi.r[k];
        }
        double dist = std::sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]);
        U += -(G * other.m * pi.m) / dist;
    }
    return U;
}

// Kinetic energy of a single body
double kineticEnergy(const Particle &pi)
{
    double modv2 = pi.v[0]*pi.v[0] + pi.v[1]*pi.v[1] + pi.v[2]*pi.v[2];
    return 0.5 * pi.m * modv2;
}

int main()
{
    try
    {
        // Read initial data
        std::ifstream file("sistema.dat");
        if (!file.is_open()) throw std::runtime_error("Could not open sistema.dat");
        auto systemParticles = initSystem(file);
        file.close();

        // Set parameters
        double giorno = 8.64e4;
        double step   = 240.0;
        double dt     = giorno / step;
        double anni   = 2.0;
        long double T = giorno * 365.0 * anni;
        long int n    = static_cast<long int>(T / dt);
        double sv     = 0.003;
        bool correction = true;

        // NASA data
        std::ifstream filenasa("pvinnasa.dat");
        if (!filenasa.is_open()) throw std::runtime_error("Could not open pvinnasa.dat");

        std::vector<ParticleVel> Vpv;
        Vpv.reserve(static_cast<std::size_t>(365 * anni));
        for (int i = 0; i < 365 * anni; ++i)
        {
            ParticleVel tmp;
            if (!(filenasa >> tmp.r[0] >> tmp.r[1] >> tmp.r[2]
                           >> tmp.v[0] >> tmp.v[1] >> tmp.v[2]))
            {
                break;
            }
            // Convert from km to meters, km/s to m/s
            for (int k = 0; k < 3; ++k)
            {
                tmp.r[k] *= 1000.0;
                tmp.v[k] *= 1000.0;
            }
            Vpv.push_back(tmp);
        }
        filenasa.close();

        // Prepare output files
        auto fnames = generateFilenames(systemParticles);
        removeOldFiles(fnames);

        std::cout << "numero di cicli: " << n << "\n";
        std::cout << "frequenza dei cicli: " << dt << " secondi\n";
        std::cout << "numero dati nasa: " << Vpv.size() << "\n";

        appendStateToFiles(systemParticles, fnames, 0.0);

        // Main loop: Velocity Verlet with optional velocity correction
        auto startTime = std::chrono::high_resolution_clock::now();

        int printCounter = 0;
        std::vector<Particle> s1;
        std::vector<double> pvswap; // times of velocity corrections
        std::vector<double> Uv, Kv, Kvnasa, Uvnasa;

        for (int i = 0; i < n; ++i)
        {
            s1 = velocityVerlet(systemParticles, dt);

            if (printCounter == static_cast<int>(1 * step))
            {
                double currentTime = i * dt;
                appendStateToFiles(s1, fnames, currentTime);
                printCounter = 0;

                if (correction)
                {
                    if (!checkVelocity(s1, Vpv, i, step, sv))
                    {
                        // Overwrite last particle velocity with NASA data
                        std::size_t idx = s1.size() - 1;
                        auto vIdx = static_cast<std::size_t>(i / step);
                        if (vIdx < Vpv.size())
                        {
                            for (int k = 0; k < 3; ++k)
                            {
                                s1[idx].v[k] = Vpv[vIdx].v[k];
                            }
                            pvswap.push_back(vIdx);
                        }
                    }
                }
                // Compute energies
                Uv.push_back(potentialEnergy(s1, s1.back()));
                Kv.push_back(kineticEnergy(s1.back()));

                // NASA energies
                std::size_t idx = s1.size() - 1;
                std::size_t vIdx = static_cast<std::size_t>(i / step);
                if (vIdx < Vpv.size())
                {
                    double modv2 =
                        Vpv[vIdx].v[0]*Vpv[vIdx].v[0] +
                        Vpv[vIdx].v[1]*Vpv[vIdx].v[1] +
                        Vpv[vIdx].v[2]*Vpv[vIdx].v[2];
                    Kvnasa.push_back(0.5 * s1[idx].m * modv2);

                    double Uvn = 0.0;
                    for (auto &other : s1)
                    {
                        if (other.m == s1[idx].m &&
                            other.r == s1[idx].r &&
                            other.v == s1[idx].v)
                        {
                            continue;
                        }
                        std::array<double, 3> diff;
                        for (int k = 0; k < 3; ++k)
                        {
                            diff[k] = other.r[k] - Vpv[vIdx].r[k];
                        }
                        double dist = std::sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]);
                        Uvn += -(6.67430e-11 * other.m * s1[idx].m) / dist;
                    }
                    Uvnasa.push_back(Uvn);
                }
            }
            systemParticles = syncParticles(s1);
            printCounter += 1;
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
        std::cout << "Program has been running for " << seconds << " seconds\n";

        // Optional: Write energies, corrections, etc. to files if desired

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}