#include "VisualizationUtils.h"
#include <iostream>
#include <string>
#include <cmath>
#include <algorithm>

// Add these includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

namespace VisualizationUtils 
{
    // Global state
    GLFWwindow* window = nullptr;
    int windowWidth = 1280;
    int windowHeight = 720;
    bool visualizationInitialized = false;
    ParticleRenderer* renderer = nullptr;
    float scale = 1.0e-3f;  // Default scale

    // Mouse callback implementations
    void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
        if (renderer) {
            renderer->handleMouseButton(button, action);
        }
    }
    
    void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
        if (renderer) {
            renderer->handleMouseMove(xpos, ypos);
        }
    }
    
    void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
        if (renderer) {
            renderer->handleMouseScroll(yoffset);
        }
    }

    bool initVisualization(int particleCount, const char* windowTitle, bool useCuda) {
    if (visualizationInitialized) return true;
        
        if (!glfwInit()) {
            std::cerr << "Failed to initialize GLFW" << std::endl;
            return false;
        }
        
        // Configure GLFW for compatibility
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
        glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);
        glfwWindowHint(GLFW_DEPTH_BITS, 24);
        
        window = glfwCreateWindow(windowWidth, windowHeight, windowTitle, NULL, NULL);
        if (!window) {
            std::cerr << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            return false;
        }
        
        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);
        
        std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
        std::cout << "OpenGL Vendor: " << glGetString(GL_VENDOR) << std::endl;
        std::cout << "OpenGL Renderer: " << glGetString(GL_RENDERER) << std::endl;
        
        // Try GLEW initialization with more error handling
        glewExperimental = GL_TRUE;
        GLenum glewError = glewInit();
        
        if (glewError != GLEW_OK) {
            std::cerr << "GLEW initialization failed: " << glewGetErrorString(glewError) << std::endl;
            std::cerr << "Error code: " << glewError << std::endl;
            
            // Try to continue without GLEW for basic OpenGL functions
            std::cout << "Attempting to continue without GLEW..." << std::endl;
            
            // Check if basic OpenGL functions are available
            if (!glGenBuffers) {
                std::cerr << "Critical OpenGL functions not available. Cannot continue." << std::endl;
                glfwDestroyWindow(window);
                glfwTerminate();
                return false;
            }
            
            std::cout << "Basic OpenGL functions available. Continuing..." << std::endl;
        } else {
            std::cout << "GLEW initialized successfully" << std::endl;
        }
        
        // Clear any OpenGL errors
        while (glGetError() != GL_NO_ERROR);
        
        // Set up callbacks
        glfwSetMouseButtonCallback(window, mouse_button_callback);
        glfwSetCursorPosCallback(window, cursor_position_callback);
        glfwSetScrollCallback(window, scroll_callback);
        
        glClearColor(0.02f, 0.02f, 0.05f, 1.0f);

        // Only perform CUDA device setup if using CUDA
        if (useCuda) {
    #ifdef CUDA_ENABLED
            unsigned int cudaGLDeviceCount = 0;
            int cudaGLDeviceId = 0;
            cudaError_t err = cudaGLGetDevices(&cudaGLDeviceCount, &cudaGLDeviceId, 1, cudaGLDeviceListAll);
            
            if (err != cudaSuccess || cudaGLDeviceCount == 0) {
                //std::cout << "Warning: OpenGL-compatible CUDA device not found, using default device\n";
                // Set a flag indicating interop may not work
            } else {
                // Set CUDA device to match OpenGL
                cudaSetDevice(cudaGLDeviceId);
                //std::cout << "Using CUDA device " << cudaGLDeviceId << " for OpenGL\n";
            }
    #endif
        } else {
            //std::cout << "CPU mode: CUDA initialization skipped\n";
        }
        
        // Create renderer
        renderer = new ParticleRenderer();
        if (!renderer->init(particleCount, useCuda)) {
            std::cerr << "Failed to initialize particle renderer" << std::endl;
            delete renderer;
            renderer = nullptr;
            glfwDestroyWindow(window);
            glfwTerminate();
            return false;
        }
        
        visualizationInitialized = true;
        return true;
    }
    
    void setupKeyboardCallback() {
        glfwSetKeyCallback(window, [](GLFWwindow* window, int key, int scancode, int action, int mods) {
            if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
            
            const float zoomFactor = 1.1f;
            const float panStep = 0.1f;
            
            switch (key) {
                case GLFW_KEY_EQUAL:
                case GLFW_KEY_KP_ADD:
                    // Zoom in by decreasing camera distance
                    if (renderer) {
                        float currentDist = renderer->getCameraDistance();
                        renderer->setCameraDistance(currentDist / zoomFactor);
                    }
                    break;
                case GLFW_KEY_MINUS:
                case GLFW_KEY_KP_SUBTRACT:
                    // Zoom out by increasing camera distance
                    if (renderer) {
                        float currentDist = renderer->getCameraDistance();
                        renderer->setCameraDistance(currentDist * zoomFactor);
                    }
                    break;
                case GLFW_KEY_LEFT:
                    if (renderer) {
                        float x = renderer->getTranslateX();
                        float y = renderer->getTranslateY();
                        renderer->setTranslate(x + panStep, y);
                    }
                    break;
                case GLFW_KEY_RIGHT:
                    if (renderer) {
                        float x = renderer->getTranslateX();
                        float y = renderer->getTranslateY();
                        renderer->setTranslate(x - panStep, y);
                    }
                    break;
                case GLFW_KEY_UP:
                    if (renderer) {
                        float x = renderer->getTranslateX();
                        float y = renderer->getTranslateY();
                        renderer->setTranslate(x, y - panStep);
                    }
                    break;
                case GLFW_KEY_DOWN:
                    if (renderer) {
                        float x = renderer->getTranslateX();
                        float y = renderer->getTranslateY();
                        renderer->setTranslate(x, y + panStep);
                    }
                    break;
                case GLFW_KEY_R:
                    // Reset view with the new reset function
                    if (renderer) {
                        renderer->resetView();
                    }
                    break;
                case GLFW_KEY_P:
                    // Increase point size
                    if (renderer) {
                        float currentSize = renderer->getPointSize();
                        renderer->setPointSize(currentSize * 1.5f);
                    }
                    break;
                case GLFW_KEY_O:
                    // Decrease point size
                    if (renderer) {
                        float currentSize = renderer->getPointSize();
                        renderer->setPointSize(std::max(1.0f, currentSize / 1.5f));
                    }
                    break;
                case GLFW_KEY_C:
                    // Toggle color mode
                    if (renderer) {
                        ParticleRenderer::ColorMode currentMode = renderer->getColorMode();
                        ParticleRenderer::ColorMode newMode = 
                            (currentMode == ParticleRenderer::ColorMode::UNIFORM) ?
                            ParticleRenderer::ColorMode::VELOCITY_MAGNITUDE :
                            ParticleRenderer::ColorMode::UNIFORM;
                        renderer->setColorMode(newMode);
                    }
                    break;
                case GLFW_KEY_X:
                    // Toggle coordinate axes display
                    if (renderer) {
                        renderer->toggleCoordinateAxes();
                    }
                    break;
                case GLFW_KEY_Q:
                    // Toggle between different scale factors
                    {
                        static int scaleMode = 0;
                        scaleMode = (scaleMode + 1) % 3;
                        
                        switch (scaleMode) {
                            case 0: scale = 1.0e-3f; break;  // Normal
                            case 1: scale = 1.0e-2f; break;  // 10x larger
                            case 2: scale = 1.0e-4f; break;  // 10x smaller
                            default: scale = 1.0e-3f;
                        }
                    }
                    break;
                case GLFW_KEY_B:
                    // Cycle through background colors (dark blue, black, light yellowish)
                    {
                        static int backgroundMode = 0;
                        backgroundMode = (backgroundMode + 1) % 3;  // Cycle through 3 options
                        
                        switch (backgroundMode) {
                            case 0:
                                glClearColor(0.07f, 0.07f, 0.13f, 1.0f);  // Very dark blue
                                break;
                            case 1:
                                glClearColor(0.0f, 0.0f, 0.0f, 1.0f);     // Pure black
                                break;
                            case 2:
                                glClearColor(0.992f, 0.965f, 0.89f, 1.0f); // Light yellowish (Solarized Light)
                                break;
                        }
                    }
                    break;
                // WASD rotation controls
                case GLFW_KEY_W:
                    // Rotate camera up
                    if (renderer) {
                        float rotX = -10.0f; // Rotate up (negative X rotation)
                        float rotY = 0.0f;
                        renderer->setCameraRotation(rotX, rotY);
                    }
                    break;
                case GLFW_KEY_S:
                    // Rotate camera down
                    if (renderer) {
                        float rotX = 10.0f; // Rotate down (positive X rotation)
                        float rotY = 0.0f;
                        renderer->setCameraRotation(rotX, rotY);
                    }
                    break;
                case GLFW_KEY_A:
                    // Rotate camera left
                    if (renderer) {
                        float rotX = 0.0f;
                        float rotY = -10.0f; // Rotate left (negative Y rotation)
                        renderer->setCameraRotation(rotX, rotY);
                    }
                    break;
                case GLFW_KEY_D:
                    // Rotate camera right
                    if (renderer) {
                        float rotX = 0.0f;
                        float rotY = 10.0f; // Rotate right (positive Y rotation)
                        renderer->setCameraRotation(rotX, rotY);
                    }
                    break;
                case GLFW_KEY_ESCAPE:
                    glfwSetWindowShouldClose(window, GLFW_TRUE);
                    break;
            }
        });
    }
    
    void cleanupVisualization() {
        if (!visualizationInitialized) return;
        
        if (renderer) {
            renderer->cleanup();
            delete renderer;
            renderer = nullptr;
        }
        
        if (window) {
            glfwDestroyWindow(window);
            window = nullptr;
        }
        
        glfwTerminate();
        visualizationInitialized = false;
    }
    
    void printKeyboardControls() {
        std::cout << "\n╔═════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║               KEYBOARD CONTROLS                     ║" << std::endl;
        std::cout << "╠═════════════════════════════════════════════════════╣" << std::endl;
        std::cout << "║  W/A/S/D   - Rotate camera                          ║" << std::endl;
        std::cout << "║  Left Click - Pan camera                            ║" << std::endl;
        std::cout << "║  Right Click - Rotate camera                        ║" << std::endl;
        std::cout << "║  Scroll     - Zoom in/out                           ║" << std::endl;
        std::cout << "║  R         - Reset camera view                      ║" << std::endl;
        std::cout << "║  Arrow Keys - Pan camera                            ║" << std::endl;
        std::cout << "║  +/-       - Zoom in/out                            ║" << std::endl;
        std::cout << "║  O/P       - Decrease/Increase point size           ║" << std::endl;
        std::cout << "║  C         - Toggle color mode                      ║" << std::endl;
        std::cout << "║  X         - Toggle coordinate axes                 ║" << std::endl;
        std::cout << "║  Q         - Cycle scale factors                    ║" << std::endl;
        std::cout << "║  B         - Toggle background color                ║" << std::endl;
        std::cout << "║  ESC       - Exit simulation                        ║" << std::endl;
        std::cout << "╚═════════════════════════════════════════════════════╝" << std::endl;
        std::cout << std::endl;
    }
    
    void updateWindowTitle(int currentStep, int totalSteps, double fps, double stepsPerSecond) {
        int progressPercent = (currentStep * 100) / totalSteps;
        std::string title = "N-body Simulation - " + 
                          std::to_string(progressPercent) + "% - " +
                          std::to_string(int(stepsPerSecond)) + " steps/sec - " +
                          std::to_string(int(fps)) + " FPS";
        glfwSetWindowTitle(window, title.c_str());
    }
}