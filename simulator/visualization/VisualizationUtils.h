#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <string>
#include "ParticleRenderer.h"

// Common visualization utilities for both CPU and GPU implementations
namespace VisualizationUtils 
{
    // Window and renderer state
    extern GLFWwindow* window;
    extern int windowWidth;
    extern int windowHeight;
    extern bool visualizationInitialized;
    extern ParticleRenderer* renderer;
    extern float scale;

    // Initialization and cleanup
    bool initVisualization(int particleCount, const char* windowTitle = "N-body Simulation", bool useCuda = false);
    void cleanupVisualization();
    
    // Mouse callbacks - must be set up by the caller
    void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
    void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
    
    // Keyboard callback - must be set by the caller
    void setupKeyboardCallback();
    
    // Help display
    void printKeyboardControls();
    
    // Window title updates
    void updateWindowTitle(int currentStep, int totalSteps, double fps, double stepsPerSecond);
}
