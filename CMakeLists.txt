# ----------------------------------------------------------------------------
#  Root CMake file for DfD Graph Cuts
# ----------------------------------------------------------------------------
#
# To perform an out of source build do the following on the command line:
# 1. navigate to the root project directory (in this case DfD2)
# 2. mkdir build
# 3. cd build
# 4. depending on your system (windows/linux) type the following
# Example CMake build line for windows & VS2015
# windows: cmake -G "Visual Studio 14 2015 Win64" -DUSE_AVX_INSTRUCTIONS=ON ..
# linux: cmake -DUSE_AVX_INSTRUCTIONS=ON ..
#
# 5. To compile:
# windows: cmake --build . --config Release
# linux: cmake --build . --config Release -- -j10
#

cmake_minimum_required(VERSION 3.9)

#PROJECT(mnist_lib)
project(mnist_lib VERSION 1.0 DESCRIPTION "Shared library for an MNIST network with internal taps to view network during operation")
set(CMAKE_CXX_STANDARD 11)

set(LIB_NAME "mnist_lib")

message("Building DLL ...")

if(WIN32)
    # Change these to point to the right location for a particular machine
    set(DLIB_ROOT "D:/dlib-19.17")
    set(COMMON_ROOT "D:/Common")
    add_definitions(-DMNIST_DLL_EXPORTS)
    #include(${DLIB_ROOT}/dlib/cmake)	
	add_subdirectory(${DLIB_ROOT}/dlib dlib_build)
    
else()
    # Change these to point to the right location for a particular machine
    set(DLIB_ROOT "~/dlib-19.18")
    set(COMMON_ROOT "~/Projects/Common")

endif()    

#include (GenerateExportHeader)

#message("dlib root: " ${DLIB_ROOT})
#message("${DLIB_ROOT}/dlib/all/*.cpp")

# get the source files for the Library
file(GLOB SOURCES "src/*.cpp")
set(SOURCES ${SOURCES} "${DLIB_ROOT}/dlib/all/source.cpp")

set(project_includes "include" 
                    "${COMMON_ROOT}/include" 
                    ${DLIB_ROOT}
                    )

message(" ")
message("--------------------------------------------------------")
message("Source Files: ")
foreach(S ${SOURCES})
    message("  " ${S})
endforeach()
message("--------------------------------------------------------")
message(" ")

message(" ")
message("--------------------------------------------------------")
message("Project Includes: ")
foreach(S ${project_includes})
    message("  " ${S})
endforeach()
message("--------------------------------------------------------")
message(" ")


if(WIN32)
    include_directories(${project_includes})
    add_library(${LIB_NAME} SHARED ${SOURCES})
    set_target_properties(${LIB_NAME} PROPERTIES VERSION ${PROJECT_VERSION})
else()
    find_package(X11 REQUIRED)
    include_directories(${project_includes})
    add_library(${LIB_NAME} SHARED ${SOURCES})
    set_target_properties(${LIB_NAME} PROPERTIES VERSION ${PROJECT_VERSION})
    target_link_libraries(${LIB_NAME} "${DLIB_ROOT}/dlib/build/libdlib.so" ${X11_LIBRARIES})
    #target_link_libraries(${LIB_NAME} dlib::dlib)
    #add_compile_options(-fPIC)
    add_definitions("-fPIC")
endif()


#GENERATE_EXPORT_HEADER(${PROJECT_NAME}
#    BASE_NAME ${PROJECT_NAME}
#    EXPORT_MACRO_NAME ${PROJECT_NAME_EXPORT}
#    EXPORT_FILE_NAME ${LIB_NAME}_Export.h
#    STATIC_DEFINE PROJECT_NAME_BUILT_AS_STATIC
#    )

set(default_build_type "Release")
if(MSVC AND (${CMAKE_VERSION} VERSION_GREATER "3.6.0"))
    message("MSVC Version: " ${MSVC_VERSION})
    target_compile_options(${PROJECT_NAME} PUBLIC "/MP" "/bigobj")
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
endif()
    

message(" ")
message(STATUS "Build Complete!")
message(" ")



