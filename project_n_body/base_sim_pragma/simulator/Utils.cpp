#include "Utils.hpp"
#include <iostream>
#include <filesystem>
#include <cmath>
#include <cstdio>

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
             << modr << " " << modv << std::endl;
    }
}