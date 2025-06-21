cmake_minimum_required(VERSION 3.16)

project(componentAugmentation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find Eigen3
find_package(Eigen3 CONFIG REQUIRED)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Source and header files
file(GLOB SOURCES
    src/*.cpp
    *.cpp
)

add_executable(componentAugmentation ${SOURCES})

# Link libraries
target_link_libraries(componentAugmentation
    ${OpenCV_LIBS}
    Eigen3::Eigen
)
