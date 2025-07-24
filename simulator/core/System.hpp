#pragma once

#include "OutputUtils.hpp"
#include "ExecutionModes.hpp" // Include this first
#include "OutputData.hpp"
#include "Particles.hpp"
#include "Forces.hpp" // Now includes the ForceMethod enum
#include "OutputModes.hpp" // Include the output modes
#include <vector>
#include <string>

// Forward declarations for CUDA functions (to avoid circular includes)
class Particles;

enum class IntegrationMethod {
    EULER,
    VELOCITY_VERLET
};

class System
{
public:
    System(Particles&& p, IntegrationMethod m, ForceMethod f, ExecutionMode execMode = ExecutionMode::CPU,
           OutputMode outMode = OutputMode::FILE_CSV)
        : particles(std::move(p)), method(m), forceMethod(f), executionMode(execMode), outputMode(outMode)
    {
        // Update particles execution mode to match system mode
        particles.setExecutionMode(executionMode);
    }

    // Make copy constructor and assignment operator delete to prevent accidental copies
    System(const System&) = delete;
    System& operator=(const System&) = delete;

    // Proper move operations
    System(System&& other) noexcept
        : particles(std::move(other.particles)), 
          method(other.method), 
          forceMethod(other.forceMethod),
          executionMode(other.executionMode),
          outputMode(other.outputMode)
    {}

    System& operator=(System&& other) noexcept {
        if (this != &other) {
            particles = std::move(other.particles);
            method = other.method;
            forceMethod = other.forceMethod;
            executionMode = other.executionMode;
            outputMode = other.outputMode;
        }
        return *this;
    }

    void setParticles(const Particles &p) {
        // Move from p to this->particles
        this->particles = std::move(p);
        // Ensure particles have the correct execution mode
        particles.setExecutionMode(executionMode);
    }
    
    // Let the user select the integrator:
    void setIntegrationMethod(IntegrationMethod newMethod) {
        this->method = newMethod;
    }

    void setForceMethod(ForceMethod newforceMethod) {
        this->forceMethod = newforceMethod;
    }
    
    // Updated method to set execution mode
    void setExecutionMode(ExecutionMode mode) {
        this->executionMode = mode;
        // Also update particles to match system mode
        particles.setExecutionMode(mode);
    }
    
    // New method to set output mode
    void setOutputMode(OutputMode mode) {
        this->outputMode = mode;
    }

    // Simplified central method to calculate accelerations using the dispatcher
    Accelerations calculateAccelerations() const {
        // Using theta=0.5 as default for Barnes-Hut methods
        return calculateForces(particles, forceMethod, 0.5);
    }

    // Compute the total energy of the system 
    double computeTotalEnergy() const;

    // Unified runSimulation that works with different output modes
    void runSimulation(double dt, int steps, double stepFreq,
                       const std::string &outputFilename,
                       const std::string &metadata);
    
    // Benchmark method that runs simulation without any output (no energy calculation)
    void runBenchmark(double dt, int steps);
    
    // Visualization method that runs simulation with real-time rendering
    void runVisualization(double dt, int steps, double stepFreq);
    
    // Methods to support CUDA implementation
    const Particles& getParticles() const { return particles; }
    Particles& getMutableParticles() { return particles; }
    
    void performIntegrationStep(double dt);

    // Add this if it's missing
    double getSimulationTime() const { return m_simulationTime; }

private:
    void velocityVerlet(double dt);
    void eulerIntegration(double dt);

    // For GPU execution - pass OutputData instead of returning it
    void runGPUSimulation(double dt, int steps, double stepFreq, OutputData& outputData);
    
    // No output GPU simulation for benchmarking
    void runGPUBenchmark(double dt, int steps);
    
    // GPU visualization implementation
    void runGPUVisualization(double dt, int steps, double stepFreq);

    // Fix initialization order to match constructor initialization
    Particles particles;
    IntegrationMethod method{IntegrationMethod::EULER};
    ForceMethod forceMethod{ForceMethod::PAIRWISE};
    ExecutionMode executionMode{ExecutionMode::CPU};
    OutputMode outputMode{OutputMode::FILE_CSV};

    // Add this member variable if it doesn't exist
    double m_simulationTime = 0.0;
};
