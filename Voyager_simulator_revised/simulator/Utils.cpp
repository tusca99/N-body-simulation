//// filepath: /home/alessio/ssd_data/Alessio/uni magistrale/Modern_Computing_Physics/project_n_body/object/Utils.cpp
#include "Utils.hpp"
#include <iostream>
#include <filesystem>
#include <cmath>
#include <cstdio>

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

std::vector<Particle> initSystem(const std::string &filename)
{
    std::ifstream ifs(filename);
    if (!ifs.is_open()) throw std::runtime_error("Could not open " + filename);

    int n = 0;
    if (!(ifs >> n)) throw std::runtime_error("Error reading number of particles");

    std::vector<Particle> particles;
    particles.reserve(n);

    for (int i = 0; i < n; ++i)
    {
        particles.push_back(readParticle(ifs));
    }
    return particles;
}

std::vector<ParticleVel> loadNASAData(const std::string &filename, int days)
{
    std::ifstream filenasa(filename);
    if (!filenasa.is_open()) throw std::runtime_error("Could not open " + filename);

    std::vector<ParticleVel> Vpv;
    Vpv.reserve(days);
    for (int i = 0; i < days; ++i)
    {
        ParticleVel tmp;
        if (!(filenasa >> tmp.r[0] >> tmp.r[1] >> tmp.r[2]
                       >> tmp.v[0] >> tmp.v[1] >> tmp.v[2]))
        {
            break;
        }
        // Convert from km to m, km/s to m/s
        for (int k = 0; k < 3; ++k)
        {
            tmp.r[k] *= 1000.0;
            tmp.v[k] *= 1000.0;
        }
        Vpv.push_back(tmp);
    }
    return Vpv;
}

std::vector<std::string> generateFilenames(const std::vector<Particle> &particles)
{
    std::vector<std::string> filenames;
    filenames.reserve(particles.size());
    for (std::size_t i = 0; i < particles.size(); ++i)
    {
        char buffer[30];
        std::sprintf(buffer, "object%zu.dat", i);
        filenames.emplace_back(buffer);
    }
    return filenames;
}

void removeOldDataFiles(const std::vector<std::string> &filenames)
{
    for (auto &fname : filenames)
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
            std::cerr << "filesystem error: " << err.what() << '\n';
        }
    }
}

void appendStateToFiles(const std::vector<Particle> &particles,
                        const std::vector<std::string> &filenames,
                        double t)
{
    for (std::size_t i = 0; i < particles.size(); ++i)
    {
        std::ofstream file(filenames[i], std::ios_base::app);
        double modr = std::sqrt(particles[i].r[0]*particles[i].r[0] +
                                particles[i].r[1]*particles[i].r[1] +
                                particles[i].r[2]*particles[i].r[2]);
        double modv = std::sqrt(particles[i].v[0]*particles[i].v[0] +
                                particles[i].v[1]*particles[i].v[1] +
                                particles[i].v[2]*particles[i].v[2]);
        file << t << " "
             << particles[i].r[0] << " " << particles[i].r[1] << " " << particles[i].r[2] << " "
             << particles[i].v[0] << " " << particles[i].v[1] << " " << particles[i].v[2] << " "
             << modr << " " << modv << "\n";
    }
}