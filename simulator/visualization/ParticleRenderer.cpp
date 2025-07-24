#include "ParticleRenderer.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <limits> // Add this for std::numeric_limits
#include <fstream>
#include <sstream>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>


ParticleRenderer::ParticleRenderer() 
    : m_vao(0), 
      m_vbo(0),
      m_velocityVBO(0),
      m_texture(0),
      m_pointSize(30.0f),  // Default point size
      m_numParticles(0),
      m_zoom(1.0f),        // Legacy zoom factor
      m_minMass(1.0f),
      m_maxMass(1.0f),
      m_maxVelocity(0.1f),
      m_colorMode(ColorMode::VELOCITY_MAGNITUDE),
      m_velocityCalculated(false),
      m_showAxes(true),
      m_positionResource(nullptr),
      m_velocityResource(nullptr),
      m_devicePositions(nullptr),
      m_deviceVelocities(nullptr),
      m_resourcesRegistered(false),
      m_resourcesMapped(false)
{
    // Brighter star-like color
    m_color[0] = 1.0f;
    m_color[1] = 1.0f;
    m_color[2] = 1.0f;
    m_color[3] = 0.9f;  // Higher alpha for more visibility
}

ParticleRenderer::~ParticleRenderer() {
    cleanup();
}


bool checkCudaGLInteropCapability() {
    // Get number of CUDA devices
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "Error getting CUDA device count: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    std::cout << "Found " << deviceCount << " CUDA devices" << std::endl;
    
    // Get current CUDA device
    int currentDevice = 0;
    err = cudaGetDevice(&currentDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error getting current CUDA device: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Check each device for GL interop compatibility
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp props;
        err = cudaGetDeviceProperties(&props, i);
        if (err != cudaSuccess) {
            std::cerr << "Error getting device properties for device " << i << ": " 
                    << cudaGetErrorString(err) << std::endl;
            continue;
        }
        
        std::cout << "Device " << i << ": " << props.name << std::endl;
        std::cout << "  Compute capability: " << props.major << "." << props.minor << std::endl;
        std::cout << "  Can map host memory: " << (props.canMapHostMemory ? "Yes" : "No") << std::endl;
        
        // Check if this device supports CUDA-GL interop
        unsigned int cudaGLDeviceCount = 0;
        int cudaGLDeviceID = -1;
        
        err = cudaGLGetDevices(&cudaGLDeviceCount, &cudaGLDeviceID, 1, cudaGLDeviceListAll);
        
        std::cout << "  GL-Compatible CUDA device ID: " << 
            (err == cudaSuccess && cudaGLDeviceCount > 0 ? std::to_string(cudaGLDeviceID) : "Not available") << std::endl;
        std::cout << "  GL Interop Support: " << (err == cudaSuccess && cudaGLDeviceCount > 0 ? "Yes" : "No") << std::endl;
        std::cout << "  GL Interop Status: " << (err == cudaSuccess ? "Success" : cudaGetErrorString(err)) << std::endl;
        
        if (err == cudaSuccess && cudaGLDeviceCount > 0) {
            // Use this device for CUDA-GL interop
            if (i != currentDevice) {
                std::cout << "Switching to CUDA device " << i << " for OpenGL interop" << std::endl;
                err = cudaSetDevice(i);
                if (err != cudaSuccess) {
                    std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << std::endl;
                    return false;
                }
            }
            return true;
        }
    }
    
    std::cout << "Warning: Could not get GL-compatible CUDA device: " << cudaGetErrorString(err) << std::endl;
    std::cout << "Using default device instead." << std::endl;
    
    // Reset error state before continuing
    cudaGetLastError(); // Clear any error
    
    // No CUDA device found that supports GL interop, stick with CPU transfer mode
    return false;
}

void ParticleRenderer::cleanup() {
    // Unregister CUDA resources first
    unregisterCudaResources();

    if (m_texture) {
        glDeleteTextures(1, &m_texture);
        m_texture = 0;
    }
    
    if (m_vbo) {
        glDeleteBuffers(1, &m_vbo);
        m_vbo = 0;
    }
    
    if (m_velocityVBO) {
        glDeleteBuffers(1, &m_velocityVBO);
        m_velocityVBO = 0;
    }
    
    if (m_vao) {
        glDeleteVertexArrays(1, &m_vao);
        m_vao = 0;
    }
}

bool ParticleRenderer::init(int particleCount, bool useCuda) {
    m_numParticles = particleCount;

    // Only perform CUDA operations if we're using CUDA
    if (useCuda) {
#ifdef CUDA_ENABLED
        // we can comment this out if we don't want to check for CUDA-OpenGL interop
        /*
        std::cout << "Checking CUDA-OpenGL interoperability capability..." << std::endl;
        checkCudaGLInteropCapability();
        
        // Explicitly set CUDA device to match OpenGL
        int device_count = 0;
        int gl_device_id = 0;
        cudaGetDeviceCount(&device_count);
        
        unsigned int num_gl_devices = 0;
        cudaError_t err = cudaGLGetDevices(&num_gl_devices, &gl_device_id, 1, cudaGLDeviceListAll);
        if (err != cudaSuccess) {
            std::cout << "Warning: Could not get GL-compatible CUDA device: " << cudaGetErrorString(err) << std::endl;
            std::cout << "Using default device instead." << std::endl;
            gl_device_id = 0;
        } else if (num_gl_devices == 0) {
            std::cout << "Warning: No GL-compatible CUDA devices found. Using default device." << std::endl;
            gl_device_id = 0;
        } else {
            std::cout << "Setting CUDA device to match OpenGL device: " << gl_device_id << std::endl;
            cudaSetDevice(gl_device_id);
        }
        */
#endif
    } else {
        //std::cout << "CPU mode: Skipping CUDA-OpenGL interoperability setup" << std::endl;
    }
    
    // Initialize shaders
    if (!m_shaderManager.init()) {
        std::cerr << "Failed to initialize shader manager" << std::endl;
        return false;
    }
    
    // Create VAO and VBOs
    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);
    glGenBuffers(1, &m_velocityVBO);
    
    glBindVertexArray(m_vao);
    
    // Position VBO
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, particleCount * sizeof(float) * 4, NULL, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(float) * 4, (void*)0);
    glEnableVertexAttribArray(0);
    
    // Velocity VBO
    glBindBuffer(GL_ARRAY_BUFFER, m_velocityVBO);
    glBufferData(GL_ARRAY_BUFFER, particleCount * sizeof(float) * 4, NULL, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(float) * 4, (void*)0);
    glEnableVertexAttribArray(1);
    
    // Create particle texture
    m_texture = m_shaderManager.createParticleTexture();
    
    // Set up OpenGL state for point sprites
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);
    glEnable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // Initialize camera with window size
    int width, height;
    GLFWwindow* window = glfwGetCurrentContext();
    glfwGetFramebufferSize(window, &width, &height);
    m_camera.init(width, height);
    
    // Create axes buffers
    setupAxesBuffers();
    
    return true;
}

void ParticleRenderer::setupAxesBuffers() {
    // This is now handled by CameraController
}

void ParticleRenderer::updatePositions(const double4* positions, const double4* velocities, int numParticles, 
                                     float scale, float translateX, float translateY, float zoom) {
    if (!m_vbo || !m_velocityVBO) return;
    
    // Store camera parameters
    setTranslate(translateX, translateY);
    m_zoom = zoom;  // Legacy
    
    // Find min/max mass for sizing and max velocity for coloring
    // Only calculate mass range once, since masses don't change
    static bool massRangeCalculated = false;
    
    if (numParticles > 0) {
        // For mass range calculation, we need to re-scan particles
        if (!massRangeCalculated) {
            // Initialize to extreme values to make sure we find true min/max
            m_minMass = std::numeric_limits<float>::max();
            m_maxMass = std::numeric_limits<float>::lowest(); // Use lowest() instead of min() for floating point
            
            // First pass: find the true min/max masses
            for (int i = 0; i < numParticles; i++) {
                float mass = positions[i].w;
                if (mass > 0.0f) { // Only consider positive masses
                    m_minMass = std::min(m_minMass, (float)mass);
                    m_maxMass = std::max(m_maxMass, (float)mass);
                }
            }
            
            // Only mark as calculated if we found a valid range
            if (m_minMass < m_maxMass) {
                massRangeCalculated = true;
            } else {
                // Apply fallback values if all masses are the same
                m_minMass = positions[0].w;
                m_maxMass = m_minMass * 10.0f;  // Create an artificial range for sizing
                massRangeCalculated = true;
            }
        }
        
        // Track position extremes for auto-adjusting view
        float minX = positions[0].x, maxX = positions[0].x;
        float minY = positions[0].y, maxY = positions[0].y;
        float minZ = positions[0].z, maxZ = positions[0].z;
        
        // Calculate velocity information even if not in velocity color mode
        // This ensures we have valid velocity data when switching back to velocity mode
        float currentMaxVelocity = 0.0f;
        
        // Always update velocity information since that can change
        for (int i = 0; i < numParticles; i++) {
            // Track position extremes
            minX = std::min(minX, (float)positions[i].x);
            maxX = std::max(maxX, (float)positions[i].x);
            minY = std::min(minY, (float)positions[i].y);
            maxY = std::max(maxY, (float)positions[i].y);
            minZ = std::min(minZ, (float)positions[i].z);
            maxZ = std::max(maxZ, (float)positions[i].z);
            
            // Calculate full 3D velocity magnitude
            float vx = velocities[i].x;
            float vy = velocities[i].y;
            float vz = velocities[i].z;
            float velMag = std::sqrt(vx*vx + vy*vy + vz*vz);
            currentMaxVelocity = std::max(currentMaxVelocity, velMag);
        }
        
        // Apply temporal smoothing with 90% previous value, 10% new value
        // Always update this regardless of color mode to preserve the value
        static float smoothedMaxVelocity = 0.0f;
        
        if (smoothedMaxVelocity == 0.0f || !m_velocityCalculated) {
            // First frame or reset, initialize directly
            smoothedMaxVelocity = currentMaxVelocity;
            m_velocityCalculated = true;
        } else {
            // Exponential moving average with high weight on previous value
            smoothedMaxVelocity = smoothedMaxVelocity * 0.9f + currentMaxVelocity * 0.1f;
        }
        
        // Ensure we have a reasonable minimum value
        m_maxVelocity = std::max(smoothedMaxVelocity, 0.001f);
        
        // No need to perform first-time auto-camera adjustment now
        if (!massRangeCalculated) {
            // Update the camera distance based on the particle distribution
            float maxSpan = std::max({maxX - minX, maxY - minY, maxZ - minZ});
            if (maxSpan > 0) {
                // Set camera distance to be able to see the whole system
                m_camera.setCameraDistance(maxSpan * 2.0f);
            }
        }
    }
    
    // Transform particle positions for rendering
    std::vector<float4> transformedPositions(numParticles);
    std::vector<float4> transformedVelocities(numParticles);
    
    for (int i = 0; i < numParticles; i++) {
        // Scale positions for visibility but don't apply camera transformation yet
        transformedPositions[i].x = positions[i].x * scale;
        transformedPositions[i].y = positions[i].y * scale;
        transformedPositions[i].z = positions[i].z * scale;
        transformedPositions[i].w = positions[i].w;  // Mass stays the same
        
        // Store velocities for color mapping
        // Amplify velocity to enhance color differences if they're too subtle
        float velocityFactor = 1.0f;
        transformedVelocities[i].x = velocities[i].x * velocityFactor;
        transformedVelocities[i].y = velocities[i].y * velocityFactor;
        transformedVelocities[i].z = velocities[i].z * velocityFactor;
        transformedVelocities[i].w = 0.0f;  // Unused
    }
    
    // Update position VBO
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, numParticles * sizeof(float4), transformedPositions.data());
    
    // Update velocity VBO
    glBindBuffer(GL_ARRAY_BUFFER, m_velocityVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, numParticles * sizeof(float4), transformedVelocities.data());
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// Debug method to log max velocity
void ParticleRenderer::logMaxVelocity() const {
    std::cout << "Max velocity used for coloring: " << m_maxVelocity << std::endl;
}

// Add a getter to expose the color mapping to other components
float ParticleRenderer::getMaxVelocity() const {
    return m_maxVelocity;
}

// Add coordinate axes rendering for better spatial reference
void ParticleRenderer::renderCoordinateAxes() {
    if (!m_showAxes) return;
    
    // Axes vertices data
    float axisLength = 2.0f; // Extends in both positive and negative directions
    
    // Define vertices: position and color for each axis
    float vertices[] = {
        // X axis (red)
        -axisLength, 0.0f, 0.0f, 1.0f, 0.2f, 0.2f,
        axisLength, 0.0f, 0.0f, 1.0f, 0.2f, 0.2f,
        
        // Y axis (green)
        0.0f, -axisLength, 0.0f, 0.2f, 1.0f, 0.2f,
        0.0f, axisLength, 0.0f, 0.2f, 1.0f, 0.2f,
        
        // Z axis (blue)
        0.0f, 0.0f, -axisLength, 0.2f, 0.2f, 1.0f,
        0.0f, 0.0f, axisLength, 0.2f, 0.2f, 1.0f
    };
    
    // Create VBO and VAO for axes
    static GLuint axesVAO = 0, axesVBO = 0;
    
    if (axesVAO == 0) {
        glGenVertexArrays(1, &axesVAO);
        glGenBuffers(1, &axesVBO);
        
        glBindVertexArray(axesVAO);
        glBindBuffer(GL_ARRAY_BUFFER, axesVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        
        // Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        
        // Color attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
    }
    
    // Set uniforms and draw axes
    m_shaderManager.setAxesUniforms(
        const_cast<float*>(m_camera.getViewMatrix()),
        const_cast<float*>(m_camera.getProjectionMatrix())
    );
    
    glBindVertexArray(axesVAO);
    glDrawArrays(GL_LINES, 0, 6);
    glBindVertexArray(0);
}

void ParticleRenderer::render() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // First render coordinate axes if enabled
    renderCoordinateAxes();
    
    // Prepare to render particles
    GLuint particleShader = m_shaderManager.getParticleShaderProgram();
    glUseProgram(particleShader);
    
    // Set uniform parameters
    m_shaderManager.setParticleUniforms(
        m_pointSize,
        m_color,
        const_cast<float*>(m_camera.getViewMatrix()),
        const_cast<float*>(m_camera.getProjectionMatrix()),
        m_minMass,
        m_maxMass,
        static_cast<int>(m_colorMode),
        m_maxVelocity
    );
    
    // Bind texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_texture);
    
    // Set up improved point sprite and blending settings
    glEnable(GL_POINT_SPRITE);
    glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
    
    // Set up proper blending for overlapping particles
    glEnable(GL_BLEND);
    
    // Use depth-aware blending for better overlapping particles
    // This helps with the flickering when bodies overlap
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // Enable depth testing but allow equal depths to be rendered based on order
    // This helps with the flickering when particles are at similar distances
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL); // Less than or equal (default is GL_LESS)
    
    // Sort particles by distance for correct transparency
    // This is done internally by drawing back-to-front when using GL_DEPTH_TEST
    
    // Draw points
    glBindVertexArray(m_vao);
    glDrawArrays(GL_POINTS, 0, m_numParticles);
    glBindVertexArray(0);
    
    // Restore state
    glDepthFunc(GL_LESS); // Restore default depth function
    glDisable(GL_POINT_SPRITE);
    glUseProgram(0);

}


bool ParticleRenderer::registerCudaResources() {
#ifdef CUDA_ENABLED
    if (m_resourcesRegistered) return true;
    
    // Only try to register if we determined GL interop is available
    bool interopAvailable = checkCudaGLInteropCapability();
    if (!interopAvailable) {
        //std::cout << "✓ CUDA-OpenGL interop not available: Using CPU transfer mode" << std::endl;
        // Clear any CUDA errors before continuing
        cudaGetLastError();
        return false;
    }
    
    // Register the buffer objects with CUDA
    cudaError_t err = cudaGraphicsGLRegisterBuffer(&m_positionResource, m_vbo, cudaGraphicsRegisterFlagsWriteDiscard);
    if (err != cudaSuccess) {
        std::cerr << "Failed to register position VBO: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    err = cudaGraphicsGLRegisterBuffer(&m_velocityResource, m_velocityVBO, cudaGraphicsRegisterFlagsWriteDiscard);
    if (err != cudaSuccess) {
        std::cerr << "Failed to register velocity VBO: " << cudaGetErrorString(err) << std::endl;
        cudaGraphicsUnregisterResource(m_positionResource);
        m_positionResource = nullptr;
        return false;
    }
    
    m_resourcesRegistered = true;
    return true;
#else
    return false;
#endif
}

void ParticleRenderer::unregisterCudaResources() {
#ifdef CUDA_ENABLED
    if (!m_resourcesRegistered) {
        return;
    }
    
    if (m_positionResource) {
        cudaGraphicsUnregisterResource(m_positionResource);
        m_positionResource = nullptr;
    }
    
    if (m_velocityResource) {
        cudaGraphicsUnregisterResource(m_velocityResource);
        m_velocityResource = nullptr;
    }
    
    m_resourcesRegistered = false;
#endif
    return;
}

bool ParticleRenderer::mapCudaResources() {
#ifdef CUDA_ENABLED
    //std::cout << "Starting resource mapping..." << std::endl;
    if (!m_resourcesRegistered) {
        std::cout << "Resources not registered, can't map" << std::endl;
        return false;
    }
    
    if (m_resourcesMapped) {
        std::cout << "Resources already mapped" << std::endl;
        return true;
    }
    
    // Map graphics resources
    //std::cout << "Mapping position resource" << std::endl;
    cudaError_t err = cudaGraphicsMapResources(1, &m_positionResource, 0);
    if (err != cudaSuccess) {
        std::cerr << "Failed to map position resource: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Get mapped pointer to VBO data
    //std::cout << "Getting position device pointer" << std::endl;
    size_t size;
    err = cudaGraphicsResourceGetMappedPointer((void**)&m_devicePositions, &size, m_positionResource);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get mapped position pointer: " << cudaGetErrorString(err) << std::endl;
        cudaGraphicsUnmapResources(1, &m_positionResource, 0);
        return false;
    }
    
    //std::cout << "Position buffer mapped at " << m_devicePositions << " with size " << size << std::endl;
    
    // Map velocity resource
    //std::cout << "Mapping velocity resource" << std::endl;
    err = cudaGraphicsMapResources(1, &m_velocityResource, 0);
    if (err != cudaSuccess) {
        std::cerr << "Failed to map velocity resource: " << cudaGetErrorString(err) << std::endl;
        cudaGraphicsUnmapResources(1, &m_positionResource, 0);
        return false;
    }
    
    // Get mapped pointer to velocity VBO data
    //std::cout << "Getting velocity device pointer" << std::endl;
    err = cudaGraphicsResourceGetMappedPointer((void**)&m_deviceVelocities, &size, m_velocityResource);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get mapped velocity pointer: " << cudaGetErrorString(err) << std::endl;
        cudaGraphicsUnmapResources(1, &m_velocityResource, 0);
        cudaGraphicsUnmapResources(1, &m_positionResource, 0);
        return false;
    }
    
    //std::cout << "Velocity buffer mapped at " << m_deviceVelocities << " with size " << size << std::endl;
    
    m_resourcesMapped = true;

    return true;
#else
    return false;
#endif
}


void ParticleRenderer::unmapCudaResources() {
#ifdef CUDA_ENABLED
    if (!m_resourcesRegistered || !m_resourcesMapped) {
        return;
    }
    
    cudaError_t err = cudaGraphicsUnmapResources(1, &m_positionResource, 0);
    if (err != cudaSuccess) {
        std::cerr << "Failed to unmap position resource: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    err = cudaGraphicsUnmapResources(1, &m_velocityResource, 0);
    if (err != cudaSuccess) {
        std::cerr << "Failed to unmap velocity resource: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    m_devicePositions = nullptr;
    m_deviceVelocities = nullptr;
    m_resourcesMapped = false;  // Update the mapped flag
#endif
    return;
}




