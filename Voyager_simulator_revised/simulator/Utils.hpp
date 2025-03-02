//// filepath: /home/alessio/ssd_data/Alessio/uni magistrale/Modern_Computing_Physics/project_n_body/object/Utils.hpp
#pragma once

#include "Particle.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>

/**
 * Read a single Particle from file.
 * Converts positions and velocities from km, km/s to SI units.
 */
Particle readParticle(std::ifstream &file);

/**
 * Initialize Particle vector from file.
 */
std::vector<Particle> initSystem(const std::string &filename);

/**
 * Load NASA data into a vector of ParticleVel.
 */
std::vector<ParticleVel> loadNASAData(const std::string &filename, int days);

/**
 * Generate a list of filenames for each Particle.
 */
std::vector<std::string> generateFilenames(const std::vector<Particle> &particles);

/**
 * Remove old data files.
 */
void removeOldDataFiles(const std::vector<std::string> &filenames);

/**
 * Append system state to output files.
 */
void appendStateToFiles(const std::vector<Particle> &particles,
                        const std::vector<std::string> &filenames,
                        double t);