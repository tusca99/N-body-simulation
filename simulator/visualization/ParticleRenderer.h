#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <vector>
#include "Particles.hpp"
#include "ShaderManager.h"
#include "CameraController.h"

class ParticleRenderer {
public:
    ParticleRenderer();
    ~ParticleRenderer();

    bool init(int particleCount, bool useCuda = false);
    void cleanup();
    
    // Set and get rendering parameters
    void setPointSize(float size) { m_pointSize = size; }
    float getPointSize() const { return m_pointSize; }
    
    // Get OpenGL buffer handles for CUDA interop
    GLuint getPositionVBO() const { return m_vbo; }
    GLuint getVelocityVBO() const { return m_velocityVBO; }
    
    // Update and render particles
    void updatePositions(const double4* positions, const double4* velocities, int numParticles, 
                       float scale, float translateX, float translateY, float zoom);
    void render();
    
    // Auto-adjust view
    void toggleAutoAdjust() { m_camera.resetView(); }
    bool getAutoAdjustEnabled() const { return true; } // Always use camera's state
    
    // Camera delegate methods
    float getZoom() const { return m_camera.getZoom(); }
    float getTranslateX() const { return m_camera.getTranslateX(); }
    float getTranslateY() const { return m_camera.getTranslateY(); }
    float getCameraDistance() const { return m_camera.getCameraDistance(); }
    
    void setZoom(float zoom) { /* deprecated - use camera distance */ }
    void setTranslate(float x, float y) { m_camera.setTranslate(x, y); }
    void setCameraDistance(float distance) { m_camera.setCameraDistance(distance); }
    void setCameraRotation(float rotX, float rotY) { m_camera.setCameraRotation(rotX, rotY); }
    
    // Reset view
    void resetView() { m_camera.resetView(); }
    
    // Mouse input handlers (forwarded to camera)
    void handleMouseMove(double xpos, double ypos) { m_camera.handleMouseMove(xpos, ypos); }
    void handleMouseButton(int button, int action) { m_camera.handleMouseButton(button, action); }
    void handleMouseScroll(double yoffset) { m_camera.handleScroll(yoffset); }
    
    // Coordinate system rendering
    void toggleCoordinateAxes() { m_showAxes = !m_showAxes; }
    bool getShowAxes() const { return m_showAxes; }
    
    // Color mode
    enum class ColorMode {
        UNIFORM,
        VELOCITY_MAGNITUDE
    };
    
    void setColorMode(ColorMode mode) { 
        m_colorMode = mode; 
    }
    
    ColorMode getColorMode() const { return m_colorMode; }
    
    // Debug/diagnostic methods
    void logMaxVelocity() const;
    float getMaxVelocity() const;

    // Add these new methods for CUDA interop
    bool registerCudaResources();
    void unregisterCudaResources();
    bool mapCudaResources();
    void unmapCudaResources();
    
    // Accessor methods for device pointers
    float4* getDevicePositionPtr() { return m_devicePositions; }
    float4* getDeviceVelocityPtr() { return m_deviceVelocities; }
    
private:
    // OpenGL resources
    GLuint m_vao;
    GLuint m_vbo;
    GLuint m_velocityVBO;
    GLuint m_texture;
    int m_numParticles;
    
    // Component managers
    ShaderManager m_shaderManager;
    CameraController m_camera;
    
    // Rendering parameters
    float m_pointSize;
    float m_color[4];
    float m_zoom; // Legacy - now using camera distance
    
    // Coordinate system rendering
    bool m_showAxes;
    void renderCoordinateAxes();
    
    // Particle properties
    float m_minMass, m_maxMass;
    float m_maxVelocity;
    ColorMode m_colorMode;
    bool m_velocityCalculated; // Track if velocity has been processed at least once
    
    // Helper methods for axes rendering
    void setupAxesBuffers();

    // Add these new members for CUDA-OpenGL interop
    cudaGraphicsResource_t m_positionResource;
    cudaGraphicsResource_t m_velocityResource;
    float4* m_devicePositions;
    float4* m_deviceVelocities;
    bool m_resourcesRegistered;
    bool m_resourcesMapped;
    
    
};
