//// filepath: /home/alessio/ssd_data/Alessio/uni magistrale/Modern_Computing_Physics/project_n_body/object/main.cpp
#include "System.hpp"
#include "Utils.hpp"   // for I/O
#include <iostream>
#include <filesystem>

int main()
{
    try
    {
        // Set working directory to 'object'
        std::filesystem::current_path("/home/alessio/ssd_data/Alessio/uni magistrale/Modern_Computing_Physics/project_n_body/object");

        // Define output directory
        std::string outputDir = "data_out/";

        // Step 1: read system data
        auto systemParticles = initSystem("data_in/sistema.dat");

        // Step 2: read NASA data
        double anni  = 1.0;
        int daysNasa = static_cast<int>(365 * anni);
        auto Vpv     = loadNASAData("data_in/pvinnasa.dat", daysNasa);

        // Generate filenames with prefix
        auto filenames = generateFilenames(systemParticles);
        for (auto &fname : filenames)
        {
            fname = outputDir + fname;
        }

        // Create the system
        System sys;
        sys.setParticles(systemParticles);
        sys.setNASAData(Vpv);

        // Override output filenames
        sys.setFilenames(filenames);

        // Prepare the simulation
        double giorno= 8.64e4;        // seconds in a day
        double step  = 240.0;         // steps per day
        double dt    = giorno / step; // integration time step
        long double T= giorno * 365.0 * anni;
        long int n   = static_cast<long int>(T / dt);
        double sv    = 0.003;
        bool corr    = true;

        std::cout << "numero di cicli: " << n << "\n";
        std::cout << "frequenza dei cicli: " << dt << " secondi\n";
        std::cout << "numero dati nasa: " << daysNasa << "\n";

        // Run
        sys.runSimulation(dt, n, step, corr, sv);

        return 0;
    }
    catch(const std::exception &ex)
    {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}