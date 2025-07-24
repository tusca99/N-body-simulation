#pragma once

#include "Particles.hpp"
#include "System.hpp"
#include <string>


// Main entry point for visualization
extern "C" void runVisualizationOnGPU(Particles& particles, 
                                      IntegrationMethod method,
                                      ForceMethod forceMethod,
                                      double dt, int steps, double stepFreq,
                                      int physicsStepsPerFrame = 0);  // 0 means auto-calculate
