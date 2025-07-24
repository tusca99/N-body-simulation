#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

// Class to handle camera control and matrix generation
class CameraController {
public:
    CameraController();
    ~CameraController();

    // Initialize matrices
    void init(int windowWidth, int windowHeight);
    
    // Camera transformation methods
    void updateMatrices();
    void resetView();
    
    // Mouse input handlers
    void handleMouseMove(double xpos, double ypos);
    void handleMouseButton(int button, int action);
    void handleScroll(double yoffset);
    
    // Camera parameters
    float getTranslateX() const { return m_translateX; }
    float getTranslateY() const { return m_translateY; }
    float getZoom() const { return m_zoom; }
    float getCameraDistance() const { return m_distance; }
    
    void setTranslate(float x, float y);
    void setCameraDistance(float distance);
    void setCameraRotation(float rotX, float rotY);
    
    // Get matrices for shaders
    const float* getViewMatrix() const { return m_viewMatrix; }
    const float* getProjectionMatrix() const { return m_projMatrix; }

private:
    // Camera state
    float m_rotX;           // X-axis rotation angle (pitch)
    float m_rotY;           // Y-axis rotation angle (yaw)
    float m_distance;       // Distance from origin
    float m_translateX;     // X translation
    float m_translateY;     // Y translation
    float m_zoom;           // Zoom factor (legacy)
    bool m_autoAdjust;      // Auto-adjust view flag
    
    // Mouse interaction state
    bool m_rotatePressed;   // Right button for rotation
    bool m_panPressed;      // Left button for panning
    double m_lastMouseX;    // Last mouse X position
    double m_lastMouseY;    // Last mouse Y position
    
    // Camera matrices
    float m_viewMatrix[16];      // View matrix
    float m_projMatrix[16];      // Projection matrix
    
    // Helper methods
    void setupProjectionMatrix(float aspectRatio);
};
