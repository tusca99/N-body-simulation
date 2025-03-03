#include "System.hpp"
#include "Utils.hpp"
#include <iostream>
#include <filesystem>
#include "Initializer.hpp"
#include <sstream>

int main()
{
    try
    {
        //std::filesystem::current_path("/home/alessio/ssd_data/Alessio/uni magistrale/Modern_Computing_Physics/project_n_body/object");
        std::string outputDir = "data_out/";
        std::string outputFile = "simulation_data_box.txt";
        
        double years = 1.0;
        double stepsPerDay = 360.0;

        // Choose one initializer by calling the corresponding method:
        
        // ----- Random Initializer Block ----- //
        
        int nParticles = 25;
        double minMass = 1.0;
        double maxMass = 10.0;
        double L = 100;             // Box side length for random initializer
        double maxVelocity = 2.0;
        double minDistance = 3.0;

        auto init_start = std::chrono::high_resolution_clock::now();

        auto result = Initializer::initRandomParticles(nParticles, minMass, maxMass, L, maxVelocity, minDistance);
        
        /*
        // ----- Galaxy Initializer Block ----- //
        int nStars = 10;
        double blackHoleMass = 1.0e6;
        double starMassMin = 1.0;
        double starMassMax = 10.0;
        double maxPosition = 100.0;
        double maxVelocity = 1000.0;
        bool addPlanets = true;
        int minPlanets = 1;
        int maxPlanets = 4;
        double planetMassMin = 0.5;
        double planetMassMax = 5.0;
        double planetDistanceMax = 20.0;
        double planetVelocityMax = 50.0;
        
        auto init_start = std::chrono::high_resolution_clock::now();

        auto result = Initializer::initGalaxy(nStars, blackHoleMass, starMassMin, starMassMax,
            maxPosition, maxVelocity,
            addPlanets, minPlanets, maxPlanets,
            planetMassMin, planetMassMax, planetDistanceMax, planetVelocityMax);
            
        */

        auto init_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> init_elapsed = init_end - init_start;
        std::cout << "Initialization time: " << init_elapsed.count() << " seconds\n";


        // The metadata string is now available as result.metadata
        std::string metadata = std::string("years: ") + std::to_string(years) + "\n" + result.metadata;
        
        std::string unifiedOutputFile = outputDir + outputFile;

        System sys;
        sys.setParticles(result.particles);
        
        double secondsInDay = 8.64e4;
        double dt = secondsInDay / stepsPerDay;
        long double T = secondsInDay * 365.0 * years;
        long int n = static_cast<long int>(T / dt);
        
        std::cout << "numero di cicli: " << n << std::endl;
        std::cout << "frequenza dei cicli: 1 ogni " << dt << " secondi" << std::endl;

        auto start = std::chrono::high_resolution_clock::now();

        sys.runSimulation(dt, n, stepsPerDay, unifiedOutputFile, metadata);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Simulation time: " << elapsed.count() << " seconds\n";
        
        return 0;
    }
    catch(const std::exception &ex)
    {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}

