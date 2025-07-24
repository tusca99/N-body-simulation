#pragma once
#include <vector>
#include <string>

// Forward declarations
struct OutputData;

// Write OutputData directly to CSV file
void flushCSVOutput(const OutputData &outputData,
                    const std::string &filename,
                    const std::string &metadata);

// Progress bar display
void printProgressBar(int currentStep, int totalSteps);