#include "CameraController.h"
#include <cmath>
#include <algorithm>

CameraController::CameraController()
    : m_rotX(0.2f),  // Less extreme vertical angle
      m_rotY(0.9f),  // Default horizontal angle
      m_distance(5.0f),
      m_translateX(0.0f),
      m_translateY(0.0f),
      m_zoom(1.0f),
      m_autoAdjust(true),
      m_rotatePressed(false),
      m_panPressed(false),
      m_lastMouseX(0.0),
      m_lastMouseY(0.0)
{
    // Initialize matrices to identity
    for (int i = 0; i < 16; i++) {
        m_viewMatrix[i] = (i % 5 == 0) ? 1.0f : 0.0f;
        m_projMatrix[i] = (i % 5 == 0) ? 1.0f : 0.0f;
    }
}

CameraController::~CameraController() {
    // Nothing to clean up
}

void CameraController::init(int windowWidth, int windowHeight) {
    float aspectRatio = (float)windowWidth / (float)windowHeight;
    setupProjectionMatrix(aspectRatio);
    updateMatrices(); // Initialize the view matrix
}

void CameraController::setupProjectionMatrix(float aspectRatio) {
    float fovy = 45.0f;
    float nearZ = 0.1f;
    float farZ = 1000.0f;
    float f = 1.0f / tan(fovy * 0.5f * 3.14159f / 180.0f);
    
    // Reset to identity first
    for (int i = 0; i < 16; i++) {
        m_projMatrix[i] = 0.0f;
    }
    
    m_projMatrix[0] = f / aspectRatio;
    m_projMatrix[5] = f;
    m_projMatrix[10] = (farZ + nearZ) / (nearZ - farZ);
    m_projMatrix[11] = -1.0f;
    m_projMatrix[14] = (2.0f * farZ * nearZ) / (nearZ - farZ);
    m_projMatrix[15] = 0.0f;
}

void CameraController::updateMatrices() {
    // Start with identity matrix for view
    for (int i = 0; i < 16; i++) {
        m_viewMatrix[i] = (i % 5 == 0) ? 1.0f : 0.0f;
    }
    
    // Get trigonometric values
    float cosX = cos(m_rotX);
    float sinX = sin(m_rotX);
    float cosY = cos(m_rotY);
    float sinY = sin(m_rotY);
    
    // Rotation around X axis (pitch - looking up/down)
    float rotX[16] = {
        1,     0,      0, 0,
        0, cosX,  -sinX, 0,  // Inverting Y rotation for more natural control
        0, sinX,   cosX, 0,
        0,    0,      0, 1
    };
    
    // Rotation around Y axis (yaw - looking left/right)
    float rotY[16] = {
        cosY, 0, sinY, 0,
        0,    1,    0, 0,
        -sinY, 0, cosY, 0,  // Adjusting to make objects orbit around Y axis
        0,    0,    0, 1
    };
    
    // Multiply rotY * rotX
    float temp[16];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            temp[i*4+j] = 0;
            for (int k = 0; k < 4; k++) {
                temp[i*4+j] += rotY[i*4+k] * rotX[k*4+j];
            }
        }
    }
    
    // Copy to view matrix
    for (int i = 0; i < 16; i++) {
        m_viewMatrix[i] = temp[i];
    }
    
    // Apply translation
    m_viewMatrix[12] = m_translateX;             // X translation
    m_viewMatrix[13] = m_translateY;             // Y translation
    m_viewMatrix[14] = -m_distance;              // Z translation (negative for moving away)
}

void CameraController::resetView() {
    m_rotX = 0.2f;      // Less extreme vertical angle
    m_rotY = 0.9f;      // Default horizontal rotation
    m_distance = 5.0f;
    m_translateX = 0.0f;
    m_translateY = 0.0f;
    m_zoom = 1.0f;
    m_autoAdjust = true;
    updateMatrices();
}

void CameraController::handleMouseMove(double xpos, double ypos) {
    float deltaX = static_cast<float>(xpos - m_lastMouseX);
    float deltaY = static_cast<float>(ypos - m_lastMouseY);
    
    // Handle rotation (right button)
    if (m_rotatePressed) {
        float sensitivity = 0.01f;
        
        m_rotY += deltaX * sensitivity;
        m_rotX += deltaY * sensitivity;
        
        // Limit vertical rotation to prevent flipping
        m_rotX = std::max(std::min(m_rotX, 1.5f), -1.5f);
        
        updateMatrices();
    }
    // Handle panning (left button)
    else if (m_panPressed) {
        // Reduce pan sensitivity for better control
        float panSpeed = 0.002f * m_distance; // Scale with zoom level
        
        m_translateX += deltaX * panSpeed;
        m_translateY -= deltaY * panSpeed;
        
        m_autoAdjust = false;
        updateMatrices();
    }
    
    m_lastMouseX = xpos;
    m_lastMouseY = ypos;
}

void CameraController::handleMouseButton(int button, int action) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        m_panPressed = (action == GLFW_PRESS);
    }
    else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        m_rotatePressed = (action == GLFW_PRESS);
    }
}

void CameraController::handleScroll(double yoffset) {
    float zoomFactor = 1.1f;
    
    // Update camera distance
    m_distance *= (yoffset > 0) ? 1.0f/zoomFactor : zoomFactor;
    
    // Limit how close/far we can get
    m_distance = std::max(0.1f, std::min(m_distance, 100.0f));
    
    updateMatrices();
}

void CameraController::setTranslate(float x, float y) {
    m_translateX = x;
    m_translateY = y;
    m_autoAdjust = false;
    updateMatrices();
}

void CameraController::setCameraDistance(float distance) {
    m_distance = std::max(0.1f, std::min(distance, 100.0f));
    updateMatrices();
}

void CameraController::setCameraRotation(float rotX, float rotY) {
    m_rotX = rotX;
    m_rotY = rotY;
    updateMatrices();
}
