cmake_minimum_required(VERSION 3.10)
project(FinalProject)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include directories from openCV
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable
add_executable(FinalProject main.cpp)

# Link OpenCV libraries
target_link_libraries(FinalProject ${OpenCV_LIBS})
