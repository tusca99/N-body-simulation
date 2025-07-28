#pragma once

#include "Particles.hpp"
#include "System.hpp"
#include "BenchmarkKernels.h"

#ifdef __cplusplus
extern "C" {
#endif

// Run benchmark simulation with minimal overhead (no energy calculation, no data transfers)
void runBenchmarkOnGPU(Particles& particles, 
                      IntegrationMethod method,
                      ForceMethod forceMethod,
                      double dt, int steps, int BLOCK_SIZE);

#ifdef __cplusplus
}
#endif
