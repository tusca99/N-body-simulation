#pragma once
#include "Particles.hpp"
#include <vector>

struct Accelerations {
    size_t n;
    double3* accel;  // Using double3 instead of forces (no mass component needed)
    
    Accelerations(size_t n_particles) : n(n_particles) {
        accel = new double3[n]();  // Initialize to zero
    }
    
    ~Accelerations() {
        delete[] accel;
    }
    
    Accelerations(const Accelerations&) = delete;
    Accelerations& operator=(const Accelerations&) = delete;
    
    // Move constructor
    Accelerations(Accelerations&& other) noexcept : n(other.n), accel(other.accel) {
        other.n = 0;
        other.accel = nullptr;
    }
    
    // Move assignment
    Accelerations& operator=(Accelerations&& other) noexcept {
        if (this != &other) {
            delete[] accel;
            n = other.n;
            accel = other.accel;
            other.n = 0;
            other.accel = nullptr;
        }
        return *this;
    }
};

// Force calculation method enum - moved from System.hpp
enum class ForceMethod {
    PAIRWISE,                   // standard pairwise force without softening
    PAIRWISE_AVX2_FP32,              // pairwise force with AVX2 optimization and softening
    ADAPTIVE_MUTUAL,            // adaptive force with mutual softening
    BARNES_HUT                // Barnes-Hut with adaptive softening and quadrupole moments

};

// Main force selector function that dispatches to appropriate implementation
Accelerations calculateForces(const Particles &p, ForceMethod method, double theta = 0.5);

// Direct N-body methods
Accelerations pairwiseAcceleration(const Particles &p);
Accelerations pairwiseAcceleration_AVX2(const Particles& p);
Accelerations pairwiseAccelerationAdaptiveMutual(const Particles &p);

// Function to compute the average mass of particles
double computeAverageMass(const Particles& p);

// Flattened tree structure definition - included here instead of a separate file
struct FlatOctreeNode {
    double minBound[3];  // Bounds of the node
    double maxBound[3];
    double cm[3];        // Center of mass 
    double totalMass;    // Total mass of particles in this node
    int firstChild;      // Index of first child (-1 if leaf)
    int parent;          // Index of parent (-1 if root)
    int next;            // Index of next sibling (-1 if last)
    std::vector<size_t> indices;  // Particle indices in this node
    
    FlatOctreeNode() : totalMass(0.0), firstChild(-1), parent(-1), next(-1) {
        for (int i = 0; i < 3; i++) {
            cm[i] = 0.0;
            minBound[i] = 0.0;
            maxBound[i] = 0.0;
        }
    }
};

// Barnes-Hut methods - keep default arguments only in header
Accelerations barnesHutAcceleration(const Particles &p, double theta = 0.5, 
                                  double eta = 0.05, double epsilon_min = 1e-4);

// Helper functions for tree construction
std::vector<FlatOctreeNode> buildFlatOctree(const Particles &p, double xmin, double xmax,
                                           double ymin, double ymax, double zmin, double zmax,
                                           size_t maxParticlesPerLeaf);

// Auto-tuning functions - keep default arguments only in header
double determineOptimalTheta(size_t particleCount, double desiredAccuracy = 1e-5);

// Keep only one version of the function with a default parameter
size_t determineOptimalMaxParticlesPerLeaf(size_t particleCount, double desiredAccuracy = 1e-5);