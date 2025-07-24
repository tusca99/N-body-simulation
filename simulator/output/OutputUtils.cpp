#include "OutputUtils.hpp"
#include "OutputData.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>

// Updated CSV output function to use the optimized OutputData format
void flushCSVOutput(const OutputData &outputData,
                   const std::string &filename, 
                   const std::string &metadata)
{
    std::ofstream csvFile(filename);
    if (!csvFile) {
        std::cerr << "Could not open CSV file " << filename << " for writing.\n";
        return;
    }
    
    // Write metadata
    csvFile << metadata << "\n";
    
    // Write header - updated for new format
    csvFile << "particle_id,time,x,y,z,vx,vy,vz,mass,energy\n";
    
    // Iterate through all timesteps and particles
    for (size_t t = 0; t < outputData.numTimeSteps; ++t) {
        // Get time and energy for this timestep
        double time = outputData.h_systemData[t * OutputData::valuesPerSystem + 0];
        double energy = outputData.h_systemData[t * OutputData::valuesPerSystem + 1];
        
        // For each particle at this timestep
        for (size_t p = 0; p < outputData.numParticles; ++p) {
            // Get particle static data (ID and mass)
            double particleId = outputData.h_staticData[p * OutputData::valuesPerStatic + 0];
            double mass = outputData.h_staticData[p * OutputData::valuesPerStatic + 1];
            
            // Get particle dynamic data (position and velocity)
            size_t particleDataIdx = (t * outputData.numParticles + p) * OutputData::valuesPerParticle;
            double x = outputData.h_particleData[particleDataIdx + 0];
            double y = outputData.h_particleData[particleDataIdx + 1];
            double z = outputData.h_particleData[particleDataIdx + 2];
            double vx = outputData.h_particleData[particleDataIdx + 3];
            double vy = outputData.h_particleData[particleDataIdx + 4];
            double vz = outputData.h_particleData[particleDataIdx + 5];
            
            // Write line to CSV
            csvFile << std::fixed << std::setprecision(8)
                    << particleId << "," // particle_id
                    << time << ","       // time
                    << x << "," << y << "," << z << "," // position
                    << vx << "," << vy << "," << vz << "," // velocity
                    << mass << "," // mass
                    << std::fixed << std::setprecision(15)
                    << energy << "\n"; // energy (same for all particles at this timestep)
        }
    }
    
    csvFile.close();
    std::cout << "CSV output written to " << filename << "\n";
}

void printProgressBar(int currentStep, int totalSteps)
{
    double progress = (totalSteps == 0) ? 0.0 : static_cast<double>(currentStep) / totalSteps;
    int barWidth = 50;
    int pos = static_cast<int>(barWidth * progress);
    std::cout << "\r[";
    for (int j = 0; j < barWidth; ++j) {
        if (j < pos) std::cout << "=";
        else if (j == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %";
    std::cout.flush();
}