#include "Forces.hpp"
#include "Particles.hpp" 
#include <cmath>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <cstdint>
#include <iostream>
#include <chrono>  // For profiling
#include <iomanip> // For formatting output
#include <functional> // Added for std::function

namespace {
    constexpr double G_AU = 39.47841760435743;
    
    // Simple profiling utility
    class ScopedTimer {
    private:
        std::chrono::high_resolution_clock::time_point start;
        std::string name;
        bool active;
        
    public:
        ScopedTimer(const std::string& name, bool active = true) 
            : start(std::chrono::high_resolution_clock::now()), name(name), active(active) {}
        
        ~ScopedTimer() {
            if (active) {
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end - start;
                std::cout << std::fixed << std::setprecision(5)
                          << "[Profile] " << name << ": " 
                          << elapsed.count() << " seconds" << std::endl;
            }
        }
    };
    
    // Global profiling stats - reset on each call
    struct BHStats {
        int maxTreeDepth = 0;
        size_t totalNodes = 0;
        size_t leafNodes = 0;
        size_t emptyNodes = 0;
        size_t maxParticlesInLeaf = 0;
        double buildTime = 0.0;
        double traversalTime = 0.0;
        double totalTime = 0.0;
        
        void print() {
            std::cout << std::fixed << std::setprecision(3)
                      << "Barnes-Hut stats: "
                      << "Nodes=" << totalNodes << " (" << leafNodes << " leaves, " << emptyNodes << " empty), "
                      << "Max depth=" << maxTreeDepth << ", "
                      << "Max particles in leaf=" << maxParticlesInLeaf << ", "
                      << "Build=" << buildTime << "s, Traverse=" << traversalTime << "s, Total=" << totalTime << "s" 
                      << std::endl;
        }
    };
    
    // Need to be enabled/disabled for production use
    const bool ENABLE_PROFILING = true;
    const bool VERBOSE_PROFILING = false; // Even more detailed output

}

// Main force selection function - similar to your CUDA implementation's pattern
Accelerations calculateForces(const Particles &p, ForceMethod method, double theta) {
    switch (method) {
        case ForceMethod::PAIRWISE:
            return pairwiseAcceleration(p);

        case ForceMethod::PAIRWISE_AVX2_FP32:
            return pairwiseAcceleration_AVX2(p);
        
        case ForceMethod::ADAPTIVE_MUTUAL:
            return pairwiseAccelerationAdaptiveMutual(p);
            
        case ForceMethod::BARNES_HUT:
            return barnesHutAcceleration(p, theta); // Quadrupole implementation
        default:
            throw std::runtime_error("Unknown force calculation method");
    }
}

double computeAverageMass(const Particles& p)
{
    if (p.n == 0) {
        return 1.0;
    }
    double totalMass = 0.0;
    for (size_t i = 0; i < p.n; ++i) {
        totalMass += p.posMass[i].w;
    }
    return totalMass / p.n;
}

// Now calculates acceleration directly instead of force with tiling and cache optimization
Accelerations pairwiseAcceleration(const Particles &p)
{
    const size_t n = p.n;
    Accelerations accel(n);
    
    // Optimal tile size based on cache size
    const size_t TILE_SIZE = 64; 
    
    #pragma omp parallel
    {
        // Create thread-local acceleration array
        double3* localAccel = new double3[n](); // Initialize to zero
        
        // Tile-based computation for better cache efficiency
        #pragma omp for nowait
        for (size_t i_tile = 0; i_tile < n; i_tile += TILE_SIZE) {
            const size_t i_end = std::min(i_tile + TILE_SIZE, n);
            
            for (size_t j_tile = 0; j_tile < n; j_tile += TILE_SIZE) {
                const size_t j_end = std::min(j_tile + TILE_SIZE, n);
                
                // Load j-tile positions into contiguous local array for better access pattern
                double4 tile_pos[TILE_SIZE];
                for (size_t j = j_tile, local_idx = 0; j < j_end; ++j, ++local_idx) {
                    tile_pos[local_idx] = p.posMass[j];
                }
                
                // Process all particles in i-tile against all particles in j-tile
                for (size_t i = i_tile; i < i_end; ++i) {
                    double ix = p.posMass[i].x;
                    double iy = p.posMass[i].y;
                    double iz = p.posMass[i].z;
                    
                    double ax = 0.0, ay = 0.0, az = 0.0;
                    
                    // Vectorizable inner loop
                    for (size_t j = j_tile, local_idx = 0; j < j_end; ++j, ++local_idx) {
                        if (i == j) continue;
                        
                        const double4& jpos = tile_pos[local_idx];
                        
                        double dx = jpos.x - ix;
                        double dy = jpos.y - iy;
                        double dz = jpos.z - iz;
                        double distSqr = dx*dx + dy*dy + dz*dz;
                        
                        if (distSqr < 1e-10) continue; // Avoid division by zero
                        
                        double dist = std::sqrt(distSqr);
                        // Calculate acceleration directly: a = G * m_j * r_ij / |r_ij|^3
                        double factor = (G_AU * jpos.w) / (dist * distSqr);
                        
                        // Accumulate acceleration
                        ax += factor * dx;
                        ay += factor * dy;
                        az += factor * dz;
                    }
                    
                    // Update thread-local acceleration
                    localAccel[i].x += ax;
                    localAccel[i].y += ay;
                    localAccel[i].z += az;
                }
            }
        }
        
        // Critical section to combine accelerations from all threads
        #pragma omp critical
        {
            for (size_t i = 0; i < n; ++i) {
                accel.accel[i].x += localAccel[i].x;
                accel.accel[i].y += localAccel[i].y;
                accel.accel[i].z += localAccel[i].z;
            }
        }
        
        // Free thread-local memory
        delete[] localAccel;
    }
    
    return accel;
}

// Pairwise acceleration with AVX and tiling for better cache efficiency
#include <immintrin.h> // For AVX intrinsics

// Approximate double rsqrt using float rsqrt (AVX2 has no native double rsqrt)
inline __m256d rsqrt_avx2_pd(__m256d x) {
    // step 1: split x in due __m128 di double e convertirli in float
    __m128 lo_f = _mm_cvtpd_ps(_mm256_castpd256_pd128(x));           // low 2 double → 2 float in lanes [0,1]
    __m128 hi_f = _mm_cvtpd_ps(_mm256_extractf128_pd(x, 1));       // high 2 double → 2 float in lanes [0,1]

    // step 2: rsqrt in float su 4 lanes (solo le prime 2 sono significative)
    __m128 rs_lo = _mm_rsqrt_ps(lo_f);
    __m128 rs_hi = _mm_rsqrt_ps(hi_f);

    // step 3: converti ciascun __m128 (4 float) in __m256d (4 double)
    __m256d out_lo = _mm256_cvtps_pd(rs_lo);      // prende le 4 float di rs_lo → 4 double
    __m256d out_hi = _mm256_cvtps_pd(rs_hi);      // prende le 4 float di rs_hi → 4 double

    // step 4: ricostruisci il risultato __m256d con low=out_lo.low128, high=out_hi.low128
    return _mm256_insertf128_pd(out_lo, _mm256_castpd256_pd128(out_hi), 1);
}

Accelerations pairwiseAcceleration_AVX2(const Particles& p) {
    const size_t n = p.n;
    Accelerations accel(n);

    constexpr size_t TILE_SIZE = 64;
    constexpr double EPS2 = 1e-10;
    const __m256d G = _mm256_set1_pd(G_AU);
    const __m256d eps2 = _mm256_set1_pd(EPS2);

    const int nthreads = omp_get_max_threads();
    std::vector<std::vector<double3>> threadAccel(nthreads, std::vector<double3>(n));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& localAccel = threadAccel[tid];

        #pragma omp for schedule(static)
        for (size_t i_tile = 0; i_tile < n; i_tile += TILE_SIZE) {
            const size_t i_end = std::min(i_tile + TILE_SIZE, n);

            for (size_t j_tile = 0; j_tile < n; j_tile += TILE_SIZE) {
                const size_t j_end = std::min(j_tile + TILE_SIZE, n);
                const size_t j_tile_size = j_end - j_tile;

                double4 tile_j[TILE_SIZE];
                for (size_t j = j_tile, local_idx = 0; j < j_end; ++j, ++local_idx) {
                    tile_j[local_idx] = p.posMass[j];
                }

                for (size_t i = i_tile; i < i_end; ++i) {
                    double ix = p.posMass[i].x;
                    double iy = p.posMass[i].y;
                    double iz = p.posMass[i].z;

                    __m256d acc_x = _mm256_setzero_pd();
                    __m256d acc_y = _mm256_setzero_pd();
                    __m256d acc_z = _mm256_setzero_pd();

                    for (size_t j = 0; j + 3 < j_tile_size; j += 4) {
                        __m256d jx = _mm256_set_pd(tile_j[j+3].x, tile_j[j+2].x, tile_j[j+1].x, tile_j[j+0].x);
                        __m256d jy = _mm256_set_pd(tile_j[j+3].y, tile_j[j+2].y, tile_j[j+1].y, tile_j[j+0].y);
                        __m256d jz = _mm256_set_pd(tile_j[j+3].z, tile_j[j+2].z, tile_j[j+1].z, tile_j[j+0].z);
                        __m256d mw = _mm256_set_pd(tile_j[j+3].w, tile_j[j+2].w, tile_j[j+1].w, tile_j[j+0].w);

                        // Skip self-interaction
                        // Mask mass to 0 if j == i (in tile)
                        if (j_tile <= i && i < j_tile + j_tile_size) {
                            for (int k = 0; k < 4; ++k) {
                                if ((j + k + j_tile) == i)
                                    ((double*)&mw)[3 - k] = 0.0; // AVX set_pd loads in reverse order
                            }
                        }

                        __m256d dx = _mm256_sub_pd(jx, _mm256_set1_pd(ix));
                        __m256d dy = _mm256_sub_pd(jy, _mm256_set1_pd(iy));
                        __m256d dz = _mm256_sub_pd(jz, _mm256_set1_pd(iz));

                        __m256d r2 = _mm256_fmadd_pd(dx, dx, _mm256_fmadd_pd(dy, dy, _mm256_mul_pd(dz, dz)));
                        __m256d mask = _mm256_cmp_pd(r2, eps2, _CMP_GT_OQ);

                        __m256d inv_r = rsqrt_avx2_pd(r2);
                        __m256d inv_r3 = _mm256_mul_pd(inv_r, _mm256_mul_pd(inv_r, inv_r));

                        __m256d scale = _mm256_mul_pd(G, _mm256_mul_pd(mw, inv_r3));
                        scale = _mm256_and_pd(scale, mask); // zero out invalid

                        acc_x = _mm256_fmadd_pd(scale, dx, acc_x);
                        acc_y = _mm256_fmadd_pd(scale, dy, acc_y);
                        acc_z = _mm256_fmadd_pd(scale, dz, acc_z);
                    }

                    double ax = ((double*)&acc_x)[0] + ((double*)&acc_x)[1] + ((double*)&acc_x)[2] + ((double*)&acc_x)[3];
                    double ay = ((double*)&acc_y)[0] + ((double*)&acc_y)[1] + ((double*)&acc_y)[2] + ((double*)&acc_y)[3];
                    double az = ((double*)&acc_z)[0] + ((double*)&acc_z)[1] + ((double*)&acc_z)[2] + ((double*)&acc_z)[3];

                    localAccel[i].x += ax;
                    localAccel[i].y += ay;
                    localAccel[i].z += az;
                }
            }
        }
    }

    // Final reduction
    for (int t = 0; t < nthreads; ++t) {
        for (size_t i = 0; i < n; ++i) {
            accel.accel[i].x += threadAccel[t][i].x;
            accel.accel[i].y += threadAccel[t][i].y;
            accel.accel[i].z += threadAccel[t][i].z;
        }
    }

    return accel;
}


// Pairwise acceleration with adaptive mutual method

Accelerations pairwiseAccelerationAdaptiveMutual(const Particles &p)
{
    const size_t n = p.n;
    Accelerations accel(n);
    const double eta = 0.01;
    const double epsilon_min = 1e-4;
    
    // Compute average mass - do this outside the parallel region
    double avgMass = computeAverageMass(p);
    
    const size_t TILE_SIZE = 64;

    #pragma omp parallel
    {
        // Create thread-local acceleration array
        double3* localAccel = new double3[n](); // Initialize to zero
        
        // Tile-based computation for better cache efficiency
        #pragma omp for nowait
        for (size_t i_tile = 0; i_tile < n; i_tile += TILE_SIZE) {  // Fixed: i_tile < n
            const size_t i_end = std::min(i_tile + TILE_SIZE, n);
            
            for (size_t j_tile = 0; j_tile < n; j_tile += TILE_SIZE) {  // Fixed: j_tile < n
                const size_t j_end = std::min(j_tile + TILE_SIZE, n);
                
                // Pre-cache j-tile data
                double4 j_posMass[TILE_SIZE];
                for (size_t j = j_tile, local_j = 0; j < j_end; ++j, ++local_j) {
                    j_posMass[local_j] = p.posMass[j];
                }
                
                for (size_t i = i_tile; i < i_end; ++i) {
                    const double ix = p.posMass[i].x;
                    const double iy = p.posMass[i].y;
                    const double iz = p.posMass[i].z;
                    const double imass = p.posMass[i].w;
                    
                    double ax = 0.0, ay = 0.0, az = 0.0;
                    
                    for (size_t j = j_tile, local_j = 0; j < j_end; ++j, ++local_j) {
                        if (i == j) continue;
                        
                        const double4& jpos = j_posMass[local_j];
                        
                        double dx = jpos.x - ix;
                        double dy = jpos.y - iy;
                        double dz = jpos.z - iz;
                        double distSqr = dx*dx + dy*dy + dz*dz;
                        
                        if (distSqr < 1e-10) continue;
                        
                        double dist = std::sqrt(distSqr);
                        double combinedMass = imass + jpos.w;
                        double epsilon = eta * dist * std::cbrt(combinedMass / (3.0 * avgMass));
                        epsilon = std::max(epsilon, epsilon_min);
                        double softDistSqr = distSqr + epsilon*epsilon;
                        double softDist = std::sqrt(softDistSqr);
                        
                        // Calculate acceleration directly
                        double factor = (G_AU * jpos.w) / (softDist * softDistSqr);
                        
                        ax += factor * dx;
                        ay += factor * dy;
                        az += factor * dz;
                    }
                    
                    localAccel[i].x += ax;
                    localAccel[i].y += ay;
                    localAccel[i].z += az;
                }
            }
        }
        
        #pragma omp critical
        {
            for (size_t i = 0; i < n; ++i) {
                accel.accel[i].x += localAccel[i].x;
                accel.accel[i].y += localAccel[i].y;
                accel.accel[i].z += localAccel[i].z;
            }
        }
        
        delete[] localAccel;
    }
    
    return accel;
}



// Auto-tuning function for theta parameter (remove default argument)
double determineOptimalTheta(size_t particleCount, double desiredAccuracy) {
    // Base theta value provides moderate accuracy
    double baseTheta = 0.5;
    
    // Adjust based on particle count (more particles -> smaller theta for better accuracy)
    if (particleCount < 1000) {
        // For small systems, we can use a larger theta (less accurate, but faster)
        return std::min(1.0, baseTheta + 0.3);
    } else if (particleCount > 100000) {
        // For very large systems, use a smaller theta for better accuracy
        return std::max(0.3, baseTheta - 0.2);
    }
    
    // Adjust based on desired accuracy
    // Lower values mean higher accuracy but slower computation
    double accuracyFactor = -0.2 * std::log10(desiredAccuracy);  // Maps 1e-3 -> 0.6, 1e-5 -> 1.0
    
    // Calculate final theta, constrained between reasonable bounds
    double theta = baseTheta + accuracyFactor;
    return std::min(std::max(theta, 0.2), 0.8);
}

// Auto-tuning function for maxParticlesPerLeaf parameter
size_t determineOptimalMaxParticlesPerLeaf(size_t particleCount, double desiredAccuracy) {
    // Start with base value that works well for most situations
    size_t baseValue = 32;
    
    // Adjust for particle count
    if (particleCount < 1000) {
        // Small systems work well with smaller nodes
        baseValue = 16;
    } else if (particleCount > 100000) {
        // Large systems benefit from larger nodes to reduce tree depth
        baseValue = 64;
    }
    
    // Adjust for desired accuracy
    double accuracyFactor = -0.1 * std::log10(desiredAccuracy);  // Maps 1e-3 -> 0.3, 1e-5 -> 0.5
    baseValue = static_cast<size_t>(baseValue * (1.0 + accuracyFactor));
    
    // Adjust for hardware
    int numThreads = omp_get_max_threads();
    float threadFactor = 32.0f / std::max(1, std::min(80, numThreads));
    baseValue = static_cast<size_t>(baseValue * threadFactor);
    
    // Clamp to reasonable range - increase upper limit to reduce tree depth
    return std::max(size_t(16), std::min(size_t(256), baseValue));
}

// Helper function to calculate Morton code for better spatial locality
std::uint64_t computeMortonCode(const double3& pos, const double3& min, const double3& size) {
    // Scale position to [0,1] range
    double x = std::max(0.0, std::min(1.0, (pos.x - min.x) / size.x));
    double y = std::max(0.0, std::min(1.0, (pos.y - min.y) / size.y));
    double z = std::max(0.0, std::min(1.0, (pos.z - min.z) / size.z));
    
    // Convert to integer range [0, 1023]
    std::uint32_t xx = static_cast<std::uint32_t>(x * 1023);
    std::uint32_t yy = static_cast<std::uint32_t>(y * 1023);
    std::uint32_t zz = static_cast<std::uint32_t>(z * 1023);
    
    // Spread bits using Morton code (Z-order curve)
    auto spreadBits = [](std::uint32_t v) -> std::uint64_t {
        std::uint64_t x = v;
        x = (x | (x << 16)) & 0x0000FFFF0000FFFF;
        x = (x | (x << 8)) & 0x00FF00FF00FF00FF;
        x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F;
        x = (x | (x << 2)) & 0x3333333333333333;
        x = (x | (x << 1)) & 0x5555555555555555;
        return x;
    };
    
    return spreadBits(xx) | (spreadBits(yy) << 1) | (spreadBits(zz) << 2);
}



void calculateNodeMassProperties(const Particles &p, FlatOctreeNode& node) {
    // Calculate center of mass and total mass
    double totalMass = 0.0;
    double cmx = 0.0, cmy = 0.0, cmz = 0.0;
    
    for (size_t idx : node.indices) {
        double m = p.posMass[idx].w;
        totalMass += m;
        cmx += p.posMass[idx].x * m;
        cmy += p.posMass[idx].y * m;
        cmz += p.posMass[idx].z * m;
    }
    
    if (totalMass > 0.0) {
        cmx /= totalMass; cmy /= totalMass; cmz /= totalMass;
    }
    
    node.totalMass = totalMass;
    node.cm[0] = cmx; node.cm[1] = cmy; node.cm[2] = cmz;
}


// Calculate tree depth and statistics recursively
void calculateTreeStats(const std::vector<FlatOctreeNode>& nodes, size_t nodeIdx, int depth, BHStats& stats) {
    if (nodeIdx >= nodes.size()) return;
    
    const FlatOctreeNode& node = nodes[nodeIdx];
    
    // Update maximum depth
    stats.maxTreeDepth = std::max(stats.maxTreeDepth, depth);
    
    // Count node types
    if (node.firstChild == -1) {
        stats.leafNodes++;
        stats.maxParticlesInLeaf = std::max(stats.maxParticlesInLeaf, node.indices.size());
    }
    
    if (node.totalMass == 0.0) {
        stats.emptyNodes++;
    }
    
    // Recursively process children
    if (node.firstChild != -1) {
        int childIdx = node.firstChild;
        while (childIdx != -1 && childIdx < nodes.size()) {
            calculateTreeStats(nodes, childIdx, depth + 1, stats);
            childIdx = nodes[childIdx].next;
        }
    }
}

// Simplified tree construction with fewer options
std::vector<FlatOctreeNode> buildFlatOctree(const Particles &p, 
                                          double xmin, double xmax,
                                          double ymin, double ymax,
                                          double zmin, double zmax,
                                          size_t maxParticlesPerLeaf = 32) {
    // Initialize root node
    std::vector<FlatOctreeNode> nodes;
    nodes.reserve(p.n * 2); // Reasonable estimate
    
    FlatOctreeNode root;
    root.minBound[0] = xmin; root.maxBound[0] = xmax;
    root.minBound[1] = ymin; root.maxBound[1] = ymax;
    root.minBound[2] = zmin; root.maxBound[2] = zmax;
    
    // Add all particles to root
    root.indices.resize(p.n);
    for (size_t i = 0; i < p.n; ++i) {
        root.indices[i] = i;
    }
    
    nodes.push_back(root);
    
    // Process nodes (breadth-first)
    for (size_t currentIdx = 0; currentIdx < nodes.size(); ++currentIdx) {
        FlatOctreeNode& node = nodes[currentIdx];
        
        // Skip empty nodes or nodes with few enough particles
        if (node.indices.empty() || node.indices.size() <= maxParticlesPerLeaf) {
            calculateNodeMassProperties(p, node);
            continue;
        }
        
        // Subdivide the node
        const double xmid = (node.minBound[0] + node.maxBound[0]) * 0.5;
        const double ymid = (node.minBound[1] + node.maxBound[1]) * 0.5;
        const double zmid = (node.minBound[2] + node.maxBound[2]) * 0.5;
        
        // Create children
        size_t childStartIdx = nodes.size();
        node.firstChild = childStartIdx;
        
        // Add all 8 children at once
        nodes.resize(childStartIdx + 8);
        
        // Initialize each child
        for (int octant = 0; octant < 8; ++octant) {
            FlatOctreeNode& child = nodes[childStartIdx + octant];
            
            child.parent = currentIdx;
            child.firstChild = -1;
            child.next = (octant < 7) ? childStartIdx + octant + 1 : -1;
            
            // Set bounding box
            child.minBound[0] = (octant & 1) ? xmid : node.minBound[0];
            child.maxBound[0] = (octant & 1) ? node.maxBound[0] : xmid;
            child.minBound[1] = (octant & 2) ? ymid : node.minBound[1];
            child.maxBound[1] = (octant & 2) ? node.maxBound[1] : ymid;
            child.minBound[2] = (octant & 4) ? zmid : node.minBound[2];
            child.maxBound[2] = (octant & 4) ? node.maxBound[2] : zmid;
        }
        
        // Distribute particles to children
        for (size_t idx : node.indices) {
            const double4& pos = p.posMass[idx];
            
            // Determine octant
            int octant = 0;
            if (pos.x >= xmid) octant |= 1;
            if (pos.y >= ymid) octant |= 2;
            if (pos.z >= zmid) octant |= 4;
            
            // Add to appropriate child
            nodes[childStartIdx + octant].indices.push_back(idx);
        }
        
        // Clear parent indices to free memory
        node.indices.clear();
        node.indices.shrink_to_fit();
    }
    
    // Bottom-up pass to compute mass properties
    for (int i = nodes.size() - 1; i >= 0; --i) {
        FlatOctreeNode& node = nodes[i];
        
        // Skip leaf nodes - they already have mass properties
        if (node.firstChild == -1) continue;
        
        // Reset properties
        node.totalMass = 0.0;
        for (int j = 0; j < 3; j++) node.cm[j] = 0.0;
        
        // Sum up from children
        int childIdx = node.firstChild;
        while (childIdx != -1 && childIdx < nodes.size()) {
            FlatOctreeNode& child = nodes[childIdx];
            
            // Skip empty children
            if (child.totalMass <= 0.0) {
                childIdx = child.next;
                continue;
            }
            
            // Accumulate mass and center of mass
            node.totalMass += child.totalMass;
            for (int j = 0; j < 3; j++) {
                node.cm[j] += child.cm[j] * child.totalMass;
            }
            
            childIdx = child.next;
        }
        
        // Normalize center of mass
        if (node.totalMass > 0.0) {
            for (int j = 0; j < 3; j++) {
                node.cm[j] /= node.totalMass;
            }
        }
    }
    
    return nodes;
}

void symmetrizeForces(Accelerations& accel, const Particles& p) {
    // Calculate total momentum
    double3 totalMomentum = {0, 0, 0};
    double totalMass = 0;
    
    for (size_t i = 0; i < p.n; i++) {
        totalMomentum.x += p.posMass[i].w * accel.accel[i].x;
        totalMomentum.y += p.posMass[i].w * accel.accel[i].y;
        totalMomentum.z += p.posMass[i].w * accel.accel[i].z;
        totalMass += p.posMass[i].w;
    }
    
    // Distribute correction to maintain zero momentum
    if (totalMass > 0) {
        double3 correction = {
            totalMomentum.x / totalMass,
            totalMomentum.y / totalMass,
            totalMomentum.z / totalMass
        };
        
        for (size_t i = 0; i < p.n; i++) {
            accel.accel[i].x -= correction.x;
            accel.accel[i].y -= correction.y;
            accel.accel[i].z -= correction.z;
        }
    }
}

// Main Barnes-Hut acceleration calculation
Accelerations barnesHutAcceleration(const Particles &p, double theta,
                                  double eta, double epsilon_min) {
    Accelerations accel(p.n);
    
    // Find bounding box in a single pass
    double xmin = p.posMass[0].x, xmax = p.posMass[0].x;
    double ymin = p.posMass[0].y, ymax = p.posMass[0].y;
    double zmin = p.posMass[0].z, zmax = p.posMass[0].z;
    
    for (size_t i = 1; i < p.n; ++i) {
        xmin = std::min(xmin, p.posMass[i].x);
        xmax = std::max(xmax, p.posMass[i].x);
        ymin = std::min(ymin, p.posMass[i].y);
        ymax = std::max(ymax, p.posMass[i].y);
        zmin = std::min(zmin, p.posMass[i].z);
        zmax = std::max(zmax, p.posMass[i].z);
    }
    
    // Add margin and make cubic
    double maxSize = std::max({xmax-xmin, ymax-ymin, zmax-zmin});
    double margin = 0.05 * maxSize;
    double halfSize = maxSize * 0.5 + margin;
    
    double xcenter = (xmax + xmin) * 0.5;
    double ycenter = (ymax + ymin) * 0.5;
    double zcenter = (zmax + zmin) * 0.5;
    
    xmin = xcenter - halfSize;
    xmax = xcenter + halfSize;
    ymin = ycenter - halfSize;
    ymax = ycenter + halfSize;
    zmin = zcenter - halfSize;
    zmax = zcenter + halfSize;
    
    // Use default if theta is invalid
    if (theta <= 0.0) {
        theta = determineOptimalTheta(p.n);
    }
    
    size_t maxParticlesPerLeaf = determineOptimalMaxParticlesPerLeaf(p.n);
    
    // Build the tree
    std::vector<FlatOctreeNode> nodes = buildFlatOctree(p, xmin, xmax, ymin, ymax, zmin, zmax, maxParticlesPerLeaf);
    
    // Calculate acceleration in parallel
    #pragma omp parallel
    {
        const double thetaSq = theta * theta;
        
        #pragma omp for schedule(dynamic, 64)
        for (size_t i = 0; i < p.n; ++i) {
            const double ix = p.posMass[i].x;
            const double iy = p.posMass[i].y;
            const double iz = p.posMass[i].z;
            
            double ax = 0.0, ay = 0.0, az = 0.0;
            
            // Stack for tree traversal
            int nodeStack[128];
            int stackSize = 0;
            nodeStack[stackSize++] = 0; // Start with root
            
            while (stackSize > 0) {
                int nodeIdx = nodeStack[--stackSize];
                const FlatOctreeNode& node = nodes[nodeIdx];
                
                // Skip empty nodes
                if (node.totalMass <= 0.0) continue;
                
                // Vector FROM particle TO node center of mass (for attraction)
                double dx = node.cm[0] - ix;
                double dy = node.cm[1] - iy;
                double dz = node.cm[2] - iz;
                double distSqr = dx*dx + dy*dy + dz*dz;
                
                // Skip self-interaction
                if (distSqr < 1e-10) continue;
                
                // Node size calculation
                double nodeSize = std::max({
                    node.maxBound[0] - node.minBound[0],
                    node.maxBound[1] - node.minBound[1],
                    node.maxBound[2] - node.minBound[2]
                });
                
                // Apply Barnes-Hut criterion: use approximation if s/d < theta
                if (node.firstChild == -1 || (nodeSize*nodeSize < distSqr * thetaSq)) {
                    // Softened distance
                    double softDistSqr = distSqr + epsilon_min*epsilon_min;
                    double softDist = std::sqrt(softDistSqr);
                    
                    // Gravitational acceleration
                    double invDist3 = 1.0 / (softDist * softDistSqr);
                    double factor = G_AU * node.totalMass * invDist3;
                    
                    ax += factor * dx;
                    ay += factor * dy;
                    az += factor * dz;
                }
                else if (node.firstChild != -1) {
                    // Add children to stack
                    int childIdx = node.firstChild;
                    while (childIdx != -1 && childIdx < nodes.size() && stackSize < 128) {
                        nodeStack[stackSize++] = childIdx;
                        childIdx = nodes[childIdx].next;
                    }
                }
            }
            
            // Store acceleration
            accel.accel[i].x = ax;
            accel.accel[i].y = ay;
            accel.accel[i].z = az;
        }
    }
    
    
    // Minimal momentum conservation check - just for verification
    double totalMomentumX = 0.0, totalMomentumY = 0.0, totalMomentumZ = 0.0;
    for (size_t i = 0; i < p.n; i++) {
        totalMomentumX += p.posMass[i].w * accel.accel[i].x;
        totalMomentumY += p.posMass[i].w * accel.accel[i].y;
        totalMomentumZ += p.posMass[i].w * accel.accel[i].z;
    }
    double momentumMagnitude = sqrt(totalMomentumX*totalMomentumX + 
                                    totalMomentumY*totalMomentumY + 
                                    totalMomentumZ*totalMomentumZ);
    
    
    //if (momentumMagnitude > 1e-10) {
    //    std::cout << "Warning: Net momentum: " << momentumMagnitude << std::endl;
    //}
    
    
    // Symmetrize forces to ensure momentum conservation
    symmetrizeForces(accel, p);

    return accel;
}

