#pragma once

#include "Particle.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>

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