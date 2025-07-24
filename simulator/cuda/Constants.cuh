#pragma once

#include <cuda_runtime.h>

// Macros for device code
#define G_AU (39.47841760435743)
#define ETA (0.01)
#define EPSILON_MIN (1e-4)

#ifdef __cplusplus
extern "C" {
#endif

// Function declaration (without extern)
void initializeConstants();

#ifdef __cplusplus
}
#endif