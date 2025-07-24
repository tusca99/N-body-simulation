#include "Constants.cuh"

// Define constants here - without extern keyword
__constant__ double d_G_AU;
__constant__ double d_eta;
__constant__ double d_epsilon_min;

// Initialize device constant memory with host values
void initializeConstants() {
    const double h_G_AU = G_AU;  // Use the macro value
    const double h_eta = ETA;
    const double h_epsilon_min = EPSILON_MIN;
    
    cudaMemcpyToSymbol(d_G_AU, &h_G_AU, sizeof(double));
    cudaMemcpyToSymbol(d_eta, &h_eta, sizeof(double));
    cudaMemcpyToSymbol(d_epsilon_min, &h_epsilon_min, sizeof(double));
}