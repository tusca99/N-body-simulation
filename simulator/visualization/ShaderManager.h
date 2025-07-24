#pragma once

#include <GL/glew.h>
#include <string>
#include <vector>

// Class to handle shader compilation, linking and uniform setting
class ShaderManager {
public:
    ShaderManager();
    ~ShaderManager();

    // Initialize and compile shaders
    bool init();

    // Get the shader program ID
    GLuint getParticleShaderProgram() const { return m_particleShaderProgram; }
    GLuint getAxesShaderProgram() const { return m_axesShaderProgram; }

    // Set uniforms for the particle shader
    void setParticleUniforms(float pointSize, float* color, 
                           float* viewMatrix, float* projMatrix,
                           float minMass, float maxMass, 
                           int colorMode, float maxVelocity);

    // Set uniforms for the axes shader
    void setAxesUniforms(float* viewMatrix, float* projMatrix);

    // Create particle texture
    GLuint createParticleTexture() const;

private:
    // Shader program IDs
    GLuint m_particleShaderProgram;
    GLuint m_axesShaderProgram;

    // Helper methods
    GLuint compileShader(GLenum type, const char* source);
    GLuint linkProgram(GLuint vertexShader, GLuint fragmentShader);

    // Shader source code
    const char* getParticleVertexShader() const;
    const char* getParticleFragmentShader() const;
    const char* getAxesVertexShader() const;
    const char* getAxesFragmentShader() const;
};
