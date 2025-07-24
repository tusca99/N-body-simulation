#!/bin/bash

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Function to compile with specific options
compile_target() {
    local target="n-body-simulation"
    
    echo "Compiling $target..."
    cmake .. && make -j$(nproc) $target
    
    if [ $? -eq 0 ]; then
        echo "Build successful! Executable: $target"
        cp $target ..
    else
        echo "Build failed!"
        exit 1
    fi
}

# Parse command line arguments
if [ "$1" == "clean" ]; then
    echo "Cleaning build directory..."
    rm -rf *
    cd ..
    rm -f n-body-simulation*
    exit 0
fi

# Default: standard build
compile_target


echo "Build process completed."
