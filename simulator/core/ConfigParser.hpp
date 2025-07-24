#pragma once

#include <string>
#include <fstream>
#include <nlohmann/json.hpp>
#include "Initializer.hpp"
#include "ExecutionModes.hpp"
#include "System.hpp"
#include "OutputModes.hpp"

using json = nlohmann::json;

// Structure to hold all configuration parameters
struct Config {
    std::string initSelected;
    json initParams;
    int threads;
    double years;
    double dtYears;
    std::string outputDir;
    std::string outputFile;
    IntegrationMethod integrationMethod;
    ForceMethod forceMethod;
    ExecutionMode executionMode;
    OutputMode outputMode;
};

// Function to parse the JSON configuration file
Config parseConfig(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open config file: " + path);
    }
    
    json j;
    in >> j;
    Config cfg;
    
    // Basic parameters
    cfg.initSelected = j["init"]["selected"].get<std::string>();
    cfg.initParams = j["init"][cfg.initSelected];
    cfg.threads = j["threads"].get<int>();
    cfg.years = j["years"].get<double>();
    cfg.dtYears = j["dtYears"].get<double>();
    cfg.outputDir = j["output"]["dir"].get<std::string>();
    cfg.outputFile = j["output"]["file"].get<std::string>();
    
    // Parse integration method
    std::string method = j["integrationMethod"].get<std::string>();
    if (method == "EULER") {
        cfg.integrationMethod = IntegrationMethod::EULER;
    } else if (method == "VELOCITY_VERLET") {
        cfg.integrationMethod = IntegrationMethod::VELOCITY_VERLET;
    } else {
        throw std::runtime_error("Unknown integration method: " + method);
    }
    
    // Parse force method
    std::string force = j["forceMethod"].get<std::string>();
    if (force == "PAIRWISE") {
        cfg.forceMethod = ForceMethod::PAIRWISE;
    } else if (force == "PAIRWISE_AVX2_FP32") {
        cfg.forceMethod = ForceMethod::PAIRWISE_AVX2_FP32;
    } else if (force == "ADAPTIVE_MUTUAL") {
        cfg.forceMethod = ForceMethod::ADAPTIVE_MUTUAL;
    } else if (force == "BARNES_HUT") {
        cfg.forceMethod = ForceMethod::BARNES_HUT;
    } else {
        throw std::runtime_error("Unknown force method: " + force);
    }
    
    // Parse execution mode
    std::string exec = j["executionMode"].get<std::string>();
    if (exec == "CPU") {
        cfg.executionMode = ExecutionMode::CPU;
    } else if (exec == "GPU") {
        cfg.executionMode = ExecutionMode::GPU;
    } else {
        throw std::runtime_error("Unknown execution mode: " + exec);
    }
    
    // Parse output mode
    std::string output = j["outputMode"].get<std::string>();
    if (output == "BENCHMARK") {
        cfg.outputMode = OutputMode::BENCHMARK;
    } else if (output == "FILE_CSV") {
        cfg.outputMode = OutputMode::FILE_CSV;
    } else if (output == "VISUALIZATION") {
        cfg.outputMode = OutputMode::VISUALIZATION;
    } else {
        throw std::runtime_error("Unknown output mode: " + output);
    }
    
    return cfg;
}

// Function to create an initializer based on the config
InitResult createInitializerFromConfig(const Config& cfg) {
    InitResult result;
    
    if (cfg.initSelected == "FROM_FILE") {
        std::string filePath = cfg.initParams["filePath"].get<std::string>();
        int partialCount = cfg.initParams["partialCount"].get<int>();
        result = Initializer::initFromSistemadat(filePath, partialCount);
    } 
    else if (cfg.initSelected == "RANDOM") {
        int nParticles = cfg.initParams["nParticles"].get<int>();
        double minMass = cfg.initParams["minMass"].get<double>();
        double maxMass = cfg.initParams["maxMass"].get<double>();
        double L = cfg.initParams["L"].get<double>();
        double maxVelocity = cfg.initParams["maxVelocity"].get<double>();
        double minDistance = cfg.initParams["minDistance"].get<double>();
        result = Initializer::initRandomParticles(nParticles, minMass, maxMass, L, maxVelocity, minDistance);
    } 
    else if (cfg.initSelected == "GALAXY") {
        int nStars = cfg.initParams["nStars"].get<int>();
        double blackHoleMass = cfg.initParams["blackHoleMass"].get<double>();
        double starMassMin = cfg.initParams["starMassMin"].get<double>();
        double starMassMax = cfg.initParams["starMassMax"].get<double>();
        double maxPosition = cfg.initParams["maxPosition"].get<double>();
        double maxVelocity = cfg.initParams["maxVelocity"].get<double>();
        bool addPlanets = cfg.initParams["addPlanets"].get<bool>();
        int minPlanets = cfg.initParams["minPlanets"].get<int>();
        int maxPlanets = cfg.initParams["maxPlanets"].get<int>();
        double planetMassMin = cfg.initParams["planetMassMin"].get<double>();
        double planetMassMax = cfg.initParams["planetMassMax"].get<double>();
        double planetDistanceMax = cfg.initParams["planetDistanceMax"].get<double>();
        double planetVelocityMax = cfg.initParams["planetVelocityMax"].get<double>();
        result = Initializer::initGalaxy(nStars, blackHoleMass, starMassMin, starMassMax,
                                         maxPosition, maxVelocity, addPlanets, minPlanets,
                                         maxPlanets, planetMassMin, planetMassMax,
                                         planetDistanceMax, planetVelocityMax);
    } 
    else if (cfg.initSelected == "STELLAR_SYSTEM") {
        int nPlanets = cfg.initParams["nPlanets"].get<int>();
        double starMass = cfg.initParams["starMass"].get<double>();
        double planetMassMin = cfg.initParams["planetMassMin"].get<double>();
        double planetMassMax = cfg.initParams["planetMassMax"].get<double>();
        double minPlanetDistance = cfg.initParams["minPlanetDistance"].get<double>();
        double maxPlanetDistance = cfg.initParams["maxPlanetDistance"].get<double>();
        bool addMoons = cfg.initParams["addMoons"].get<bool>();
        int minMoons = cfg.initParams["minMoons"].get<int>();
        int maxMoons = cfg.initParams["maxMoons"].get<int>();
        double moonMassMin = cfg.initParams["moonMassMin"].get<double>();
        double moonMassMax = cfg.initParams["moonMassMax"].get<double>();
        double moonDistanceMax = cfg.initParams["moonDistanceMax"].get<double>();
        double moonVelocityMax = cfg.initParams["moonVelocityMax"].get<double>();
        result = Initializer::initStellarSystem(starMass, nPlanets, planetMassMin, planetMassMax,
                                                minPlanetDistance, maxPlanetDistance,
                                                addMoons, minMoons, maxMoons,
                                                moonMassMin, moonMassMax,
                                                moonDistanceMax, moonVelocityMax);
    } 
    else if (cfg.initSelected == "SPIRAL_GALAXY") {
        int nStars = cfg.initParams["nStars"].get<int>();
        double totalMass = cfg.initParams["totalMass"].get<double>();
        int nArms = cfg.initParams["nArms"].get<int>();
        double galaxyRadius = cfg.initParams["galaxyRadius"].get<double>();
        double thickness = cfg.initParams["thickness"].get<double>();
        double perturbation = cfg.initParams["perturbation"].get<double>();
        result = Initializer::initSpiralGalaxy(nStars, totalMass, nArms, galaxyRadius, thickness, perturbation);
    }
    else {
        throw std::runtime_error("Unknown initialization method: " + cfg.initSelected);
    }
    
    return result;
}
