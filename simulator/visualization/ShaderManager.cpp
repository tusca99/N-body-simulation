#include "ShaderManager.h"
#include <iostream>
#include <cmath>
#include <algorithm>

// Smoothstep function for texture generation (same as GLSL smoothstep)
float smoothstep(float edge0, float edge1, float x) {
    // Scale and clamp x to 0..1 range
    x = std::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    // Evaluate polynomial
    return x * x * (3 - 2 * x);
}

ShaderManager::ShaderManager()
    : m_particleShaderProgram(0),
      m_axesShaderProgram(0)
{
}

ShaderManager::~ShaderManager() {
    if (m_particleShaderProgram) {
        glDeleteProgram(m_particleShaderProgram);
        m_particleShaderProgram = 0;
    }
    
    if (m_axesShaderProgram) {
        glDeleteProgram(m_axesShaderProgram);
        m_axesShaderProgram = 0;
    }
}

bool ShaderManager::init() {
    // Compile particle shaders
    GLuint particleVertex = compileShader(GL_VERTEX_SHADER, getParticleVertexShader());
    if (!particleVertex) return false;
    
    GLuint particleFragment = compileShader(GL_FRAGMENT_SHADER, getParticleFragmentShader());
    if (!particleFragment) {
        glDeleteShader(particleVertex);
        return false;
    }
    
    m_particleShaderProgram = linkProgram(particleVertex, particleFragment);
    glDeleteShader(particleVertex);
    glDeleteShader(particleFragment);
    
    if (!m_particleShaderProgram) return false;
    
    // Compile axes shaders
    GLuint axesVertex = compileShader(GL_VERTEX_SHADER, getAxesVertexShader());
    if (!axesVertex) return false;
    
    GLuint axesFragment = compileShader(GL_FRAGMENT_SHADER, getAxesFragmentShader());
    if (!axesFragment) {
        glDeleteShader(axesVertex);
        return false;
    }
    
    m_axesShaderProgram = linkProgram(axesVertex, axesFragment);
    glDeleteShader(axesVertex);
    glDeleteShader(axesFragment);
    
    return m_axesShaderProgram != 0;
}

GLuint ShaderManager::compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cerr << "Shader compilation failed: " << infoLog << std::endl;
        glDeleteShader(shader);
        return 0;
    }
    
    return shader;
}

GLuint ShaderManager::linkProgram(GLuint vertexShader, GLuint fragmentShader) {
    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, NULL, infoLog);
        std::cerr << "Shader program linking failed: " << infoLog << std::endl;
        glDeleteProgram(program);
        return 0;
    }
    
    return program;
}

void ShaderManager::setParticleUniforms(float pointSize, float* color, 
                                float* viewMatrix, float* projMatrix,
                                float minMass, float maxMass, 
                                int colorMode, float maxVelocity) {
    glUseProgram(m_particleShaderProgram);
    
    GLint pointSizeLoc = glGetUniformLocation(m_particleShaderProgram, "pointSize");
    glUniform1f(pointSizeLoc, pointSize);
    
    GLint texLoc = glGetUniformLocation(m_particleShaderProgram, "particleTexture");
    glUniform1i(texLoc, 0);
    
    GLint colorLoc = glGetUniformLocation(m_particleShaderProgram, "particleColor");
    glUniform4fv(colorLoc, 1, color);
    
    GLint viewMatrixLoc = glGetUniformLocation(m_particleShaderProgram, "viewMatrix");
    glUniformMatrix4fv(viewMatrixLoc, 1, GL_FALSE, viewMatrix);
    
    GLint projMatrixLoc = glGetUniformLocation(m_particleShaderProgram, "projMatrix");
    glUniformMatrix4fv(projMatrixLoc, 1, GL_FALSE, projMatrix);
    
    GLint minMassLoc = glGetUniformLocation(m_particleShaderProgram, "minMass");
    glUniform1f(minMassLoc, minMass);
    
    GLint maxMassLoc = glGetUniformLocation(m_particleShaderProgram, "maxMass");
    glUniform1f(maxMassLoc, maxMass);
    
    GLint colorModeLoc = glGetUniformLocation(m_particleShaderProgram, "colorMode");
    glUniform1i(colorModeLoc, colorMode);
    
    GLint maxVelocityLoc = glGetUniformLocation(m_particleShaderProgram, "maxVelocity");
    glUniform1f(maxVelocityLoc, maxVelocity);
}

void ShaderManager::setAxesUniforms(float* viewMatrix, float* projMatrix) {
    glUseProgram(m_axesShaderProgram);
    
    GLint viewMatrixLoc = glGetUniformLocation(m_axesShaderProgram, "viewMatrix");
    glUniformMatrix4fv(viewMatrixLoc, 1, GL_FALSE, viewMatrix);
    
    GLint projMatrixLoc = glGetUniformLocation(m_axesShaderProgram, "projMatrix");
    glUniformMatrix4fv(projMatrixLoc, 1, GL_FALSE, projMatrix);
}

GLuint ShaderManager::createParticleTexture() const {
    const int texSize = 128;
    std::vector<unsigned char> texData(texSize * texSize * 4);
    
    // Create a circle with clean anti-aliased edges
    for (int y = 0; y < texSize; y++) {
        for (int x = 0; x < texSize; x++) {
            // Calculate normalized coordinates (-1 to 1)
            float nx = (x / (float)(texSize - 1)) * 2.0f - 1.0f;
            float ny = (y / (float)(texSize - 1)) * 2.0f - 1.0f;
            
            // Calculate radial distance from center
            float dist = std::sqrt(nx * nx + ny * ny);
            
            // Create smooth edge with better anti-aliasing
            float alpha;
            float innerRadius = 0.6f;
            float outerRadius = 1.0f;
            
            if (dist <= innerRadius) {
                alpha = 1.0f;  // Interior fully opaque
            } 
            else if (dist < outerRadius) {
                // Smooth transition from inner to outer using smoothstep
                alpha = smoothstep(outerRadius, innerRadius, dist);
            } 
            else {
                alpha = 0.0f;  // Outside fully transparent
            }
            
            // Apply slight brightness variation for more natural look
            unsigned char brightness = 255;
            if (dist < innerRadius) {
                // Add slight radial gradient to interior
                float center_factor = 1.0f - (dist / innerRadius) * 0.2f;  // 20% brighter in center
                brightness = static_cast<unsigned char>(255 * center_factor);
            }
            
            // Store RGBA values
            int idx = 4 * (y * texSize + x);
            texData[idx + 0] = brightness;  // R
            texData[idx + 1] = brightness;  // G
            texData[idx + 2] = brightness;  // B
            texData[idx + 3] = static_cast<unsigned char>(alpha * 255.0f);  // A
        }
    }
    
    // Create OpenGL texture
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texSize, texSize, 0, GL_RGBA, GL_UNSIGNED_BYTE, texData.data());
    
    // Set texture parameters for optimal quality
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    // Generate mipmaps for better quality at different sizes
    glGenerateMipmap(GL_TEXTURE_2D);
    
    glBindTexture(GL_TEXTURE_2D, 0);
    return texture;
}

const char* ShaderManager::getParticleVertexShader() const {
    // Simplified vertex shader with sizing algorithm similar to Python implementation
    return
        "#version 330 core\n"
        "layout(location = 0) in vec4 position;\n"  // xyz = position, w = mass
        "layout(location = 1) in vec4 velocity;\n"  // xyz = velocity, w = unused
        "uniform float pointSize;\n"
        "uniform mat4 viewMatrix;\n"
        "uniform mat4 projMatrix;\n"
        "uniform float minMass;\n"
        "uniform float maxMass;\n"
        "uniform int colorMode;\n"
        "uniform float maxVelocity;\n"
        "out vec4 particleVelocity;\n"
        "out float particleMass;\n"
        "void main() {\n"
        "    // Transform position by view and projection matrices\n"
        "    vec4 viewPos = viewMatrix * vec4(position.xyz, 1.0);\n"
        "    gl_Position = projMatrix * viewPos;\n"
        "    \n"
        "    // Store mass for fragment shader\n"
        "    float mass = position.w;\n"
        "    particleMass = mass;\n"
        "    \n"
        "    // SIMPLIFIED SIZING ALGORITHM (similar to Python plot function)\n"
        "    // Size bounds - adjust these to control overall size range\n"
        "    float minSize = 2.0;\n"
        "    float maxSize = 15.0;\n"
        "    \n"
        "    // Handle zero or negative masses\n"
        "    if (mass <= 0.0) {\n"
        "        gl_PointSize = minSize * pointSize / 1.0;\n"
        "        particleVelocity = velocity;\n"
        "        return;\n"
        "    }\n"
        "    \n"
        "    // Calculate log10 values safely (log10(x) = log(x)/log(10))\n"
        "    float log10 = log(10.0);\n"
        "    float logMass = log(max(mass, 1e-20)) / log10;\n"
        "    float logMassMin = log(max(minMass, 1e-20)) / log10;\n"
        "    float logMassMax = log(max(maxMass, minMass*1.01)) / log10;\n"
        "    \n"
        "    // Check for valid range\n"
        "    float logRange = logMassMax - logMassMin;\n"
        "    if (logRange < 1e-10) {\n"
        "        // Almost no range, use middle size\n"
        "        gl_PointSize = (minSize + maxSize) / 2.0 * pointSize / 30.0;\n"
        "        particleVelocity = velocity;\n"
        "        return;\n"
        "    }\n"
        "    \n"
        "    // Calculate normalized position in the log scale\n"
        "    float normalizedPosition = (logMass - logMassMin) / logRange;\n"
        "    \n"
        "    // Clamp between 0 and 1 to handle outliers\n"
        "    normalizedPosition = clamp(normalizedPosition, 0.0, 1.0);\n"
        "    \n"
        "    // Linear interpolation between minSize and maxSize\n"
        "    float baseSize = minSize + normalizedPosition * (maxSize - minSize);\n"
        "    \n"
        "    // Apply distance-based perspective scaling - important for 3D visualization\n"
        "    float distanceToCamera = max(length(viewPos.xyz), 0.1);\n"
        "    gl_PointSize = baseSize * 8.0 / distanceToCamera;\n"
        "    \n"
        "    // Apply user point size scaling (default is 30.0)\n"
        "    gl_PointSize = gl_PointSize * (pointSize / 30.0);\n"
        "    \n"
        "    // Cap point size to avoid excessive large points\n"
        "    gl_PointSize = min(gl_PointSize, pointSize * 0.8);\n"
        "    \n"
        "    // Pass velocity to fragment shader\n"
        "    particleVelocity = velocity;\n"
        "}\n";
}

const char* ShaderManager::getParticleFragmentShader() const {
    return
        "#version 330 core\n"
        "in vec4 particleVelocity;\n"
        "in float particleMass;\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D particleTexture;\n"
        "uniform vec4 particleColor;\n"
        "uniform int colorMode;\n"
        "uniform float maxVelocity;\n"
        "uniform float minMass;\n"
        "uniform float maxMass;\n"
        "vec3 velocityToColor(vec3 vel) {\n"
        "    // Compute the full 3D velocity magnitude\n"
        "    float speed = length(vel);  // This is sqrt(vx^2 + vy^2 + vz^2)\n"
        "    \n"
        "    // Ensure we have a valid normalization range\n"
        "    float actualMaxVelocity = max(maxVelocity, 0.00001);\n"
        "    \n"
        "    // Normalize speed to [0,1] range with better stability\n"
        "    float normalizedSpeed = clamp(speed / actualMaxVelocity, 0.0, 1.0);\n"
        "    \n"
        "    // Apply smoothing to reduce flicker\n"
        "    normalizedSpeed = smoothstep(0.0, 1.0, normalizedSpeed);\n"
        "    \n"
        "    // Smoother color gradient for less abrupt transitions\n"
        "    vec3 color;\n"
        "    \n"
        "    if (normalizedSpeed < 0.2) {\n"
        "        // Blue to Cyan (0.0 - 0.2)\n"
        "        float t = normalizedSpeed / 0.2;\n"
        "        color = mix(vec3(0.0, 0.2, 0.8), vec3(0.0, 0.7, 0.9), t);\n"
        "    } else if (normalizedSpeed < 0.5) {\n"
        "        // Cyan to Green to Yellow (0.2 - 0.5)\n"
        "        float t = (normalizedSpeed - 0.2) / 0.3;\n"
        "        color = mix(vec3(0.0, 0.7, 0.9), vec3(0.8, 0.8, 0.0), t);\n"
        "    } else if (normalizedSpeed < 0.8) {\n"
        "        // Yellow to Orange (0.5 - 0.8)\n"
        "        float t = (normalizedSpeed - 0.5) / 0.3;\n"
        "        color = mix(vec3(0.8, 0.8, 0.0), vec3(1.0, 0.5, 0.0), t);\n"
        "    } else {\n"
        "        // Orange to Red (0.8 - 1.0)\n"
        "        float t = (normalizedSpeed - 0.8) / 0.2;\n"
        "        color = mix(vec3(1.0, 0.5, 0.0), vec3(1.0, 0.0, 0.0), t);\n"
        "    }\n"
        "    \n"
        "    // Apply brightness boost based on mass to enhance visibility\n"
        "    float normalizedMass = 0.0;\n"
        "    if (particleMass > 0.0) {\n"
        "        normalizedMass = clamp((log(particleMass) - log(minMass)) / (log(maxMass) - log(minMass)), 0.0, 1.0);\n"
        "    }\n"
        "    \n"
        "    // Make larger bodies appear slightly brighter\n"
        "    float brightnessBoost = 1.0 + 0.8 * normalizedMass;\n"
        "    return color * brightnessBoost;\n"
        "}\n"
        "void main() {\n"
        "    // Calculate distance from center of point\n"
        "    vec2 center = vec2(0.5, 0.5);\n"
        "    float dist = distance(gl_PointCoord, center) * 2.0;\n"
        "    \n"
        "    // Discard fragments outside the circle\n"
        "    if (dist > 1.0) {\n"
        "        discard;\n"
        "    }\n"
        "    \n"
        "    // Get base color from texture\n"
        "    vec4 texColor = texture(particleTexture, gl_PointCoord);\n"
        "    \n"
        "    // Calculate final color based on mode\n"
        "    vec4 finalColor;\n"
        "    if (colorMode == 1) {\n"
        "        vec3 velColor = velocityToColor(particleVelocity.xyz);\n"
        "        finalColor = vec4(velColor, particleColor.a);\n"
        "    } else {\n"
        "        finalColor = particleColor;\n"
        "    }\n"
        "    \n"
        "    // Use mass to influence opacity - make larger bodies more opaque\n"
        "    float normalizedMass = 0.0;\n"
        "    if (particleMass > 0.0 && maxMass > minMass) {\n"
        "        normalizedMass = clamp((log(particleMass) - log(minMass)) / (log(maxMass) - log(minMass)), 0.0, 1.0);\n"
        "    }\n"
        "    \n"
        "    // Create a smooth edge falloff, more pronounced for smaller bodies\n"
        "    float edgeWidth = mix(0.3, 0.1, normalizedMass);  // Smaller bodies have softer edges\n"
        "    float edgeStart = 1.0 - edgeWidth;\n"
        "    float alpha = smoothstep(1.0, edgeStart, dist) * texColor.a;\n"
        "    \n"
        "    // Larger bodies are more opaque even at the edges\n"
        "    alpha = mix(alpha, alpha * 1.3, normalizedMass);\n"
        "    alpha = clamp(alpha, 0.0, 1.0);\n"
        "    \n"
        "    // Output final color with adjusted alpha\n"
        "    FragColor = vec4(finalColor.rgb, alpha);\n"
        "}\n";
}

const char* ShaderManager::getAxesVertexShader() const {
    return
        "#version 330 core\n"
        "layout(location = 0) in vec3 position;\n"
        "layout(location = 1) in vec3 color;\n"
        "uniform mat4 viewMatrix;\n"
        "uniform mat4 projMatrix;\n"
        "out vec3 vertexColor;\n"
        "void main() {\n"
        "    gl_Position = projMatrix * viewMatrix * vec4(position, 1.0);\n"
        "    vertexColor = color;\n"
        "}\n";
}

const char* ShaderManager::getAxesFragmentShader() const {
    return
        "#version 330 core\n"
        "in vec3 vertexColor;\n"
        "out vec4 FragColor;\n"
        "void main() {\n"
        "    FragColor = vec4(vertexColor, 1.0);\n"
        "}\n";
}