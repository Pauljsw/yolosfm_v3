cmake_minimum_required(VERSION 3.10)
project(OrbbecCalibExtractor)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OrbbecSDK
find_path(ORBBEC_INCLUDE_DIR libobsensor/ObSensor.hpp
    PATHS /usr/local/include /usr/include
)

find_library(ORBBEC_LIBRARY
    NAMES OrbbecSDK
    PATHS /usr/local/lib /usr/lib
)

if(NOT ORBBEC_INCLUDE_DIR OR NOT ORBBEC_LIBRARY)
    message(FATAL_ERROR "OrbbecSDK not found! Please install the SDK first.")
endif()

message(STATUS "Found OrbbecSDK include: ${ORBBEC_INCLUDE_DIR}")
message(STATUS "Found OrbbecSDK library: ${ORBBEC_LIBRARY}")

include_directories(${ORBBEC_INCLUDE_DIR})

add_executable(extract_calib extract_calib.cpp)
target_link_libraries(extract_calib ${ORBBEC_LIBRARY})
