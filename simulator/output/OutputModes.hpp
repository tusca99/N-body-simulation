#pragma once

// Define different output modes for simulation results
enum class OutputMode {
    BENCHMARK,             // No output (for benchmarking)
    FILE_CSV,         // CSV file output (current implementation)
    VISUALIZATION     // Real-time visualization with OpenGL
};
