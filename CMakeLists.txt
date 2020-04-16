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

project(mnist_lib VERSION 1.0 DESCRIPTION "Shared library for an MNIST network with internal taps to view network during operation")
set(CMAKE_CXX_STANDARD 11)

set(LIB_NAME "mnist_lib")
set(default_build_type "Release")

set(PLATFORM "$ENV{PLATFORM}" CACHE INTERNAL "Copied from environment variable")
if("${PLATFORM}" STREQUAL "")
    set(PLATFORM "NOT_SET")
    message("The PLATFORM Environment variable is not set on this system.  Setting it to:")
endif()
message("Platform: " ${PLATFORM})

option(USE_AVX_INSTRUCTIONS  "Compile your program with AVX instructions"  ON)
option(BUILD_LIB  "Build the project into a library (.dll/.so) vs a standalone executable"  ON)


if(WIN32)
    if(${PLATFORM} STREQUAL "Laptop")
        set(COMMON_ROOT "D:/Common")
        set(DLIB_ROOT "D:/dlib-19.19")
        set(DLIB_CONTRIB "D:/Projects/dlib-contrib")
    elseif(${PLATFORM} STREQUAL "LaptopN")
        set(COMMON_ROOT "C:/Projects/Common")
        set(DLIB_ROOT "C:/dlib-19.19")
        set(DLIB_CONTRIB "C:/Projects/dlib-contrib")
    elseif(${PLATFORM} STREQUAL "BEAST")
        set(COMMON_ROOT "D:/Common")
        set(DLIB_ROOT "D:/dlib-19.19")
        set(DLIB_CONTRIB "D:/Projects/dlib-contrib")
    else()
        set(COMMON_ROOT "D:/Common")
        set(DLIB_ROOT "D:/dlib-19.19")
        set(DLIB_CONTRIB "D:/Projects/dlib-contrib")
    endif() 
    
else()
    if(${PLATFORM} STREQUAL "HPC")
        set(COMMON_ROOT "~/Projects/Common")
        set(DLIB_ROOT "~/dlib-19.19")
        set(DLIB_CONTRIB "~/Projects/dlib-contrib")
    elseif(${PLATFORM} STREQUAL "MainGear")
        set(COMMON_ROOT "~/Projects/Common")
        set(DLIB_ROOT "~/dlib-19.18")
        set(DLIB_CONTRIB "~/Projects/dlib-contrib")
    else()
        set(COMMON_ROOT "~/Projects/Common")
        set(DLIB_ROOT "~/dlib-19.19")
        set(DLIB_CONTRIB "~/Projects/dlib-contrib")
    endif()
    
endif()


include(${DLIB_ROOT}/dlib/cmake)

# get the source files for the Library
file(GLOB SOURCES "src/*.cpp")
#set(SOURCES ${SOURCES} "${DLIB_ROOT}/dlib/all/source.cpp")

set(project_includes "include" 
    "${COMMON_ROOT}/include" 
    "${DLIB_CONTRIB}/include" 
    #${DLIB_ROOT}
    )
                    

message(" ")
message("--------------------------------------------------------")
message("Source Files: ")
foreach(S ${SOURCES})
    message("  " ${S})
endforeach()
message("--------------------------------------------------------")

message(" ")

message("--------------------------------------------------------")
message("Project Includes: ")
foreach(S ${project_includes})
    message("  " ${S})
endforeach()
message("--------------------------------------------------------")
message(" ")

include_directories(${project_includes})



if(BUILD_LIB)
    message("Building MNIST Library...")
    
    add_definitions(-DBUILD_LIB)

    if(WIN32)
        add_definitions(-DMNIST_DLL_EXPORTS)
        #add_subdirectory(${DLIB_ROOT}/dlib dlib_build)
        
        add_library(${LIB_NAME} SHARED ${SOURCES})
        set_target_properties(${LIB_NAME} PROPERTIES VERSION ${PROJECT_VERSION})
        target_link_libraries(${LIB_NAME} dlib::dlib)
        
        if(MSVC AND (${CMAKE_VERSION} VERSION_GREATER "3.6.0"))
            message("MSVC Version: " ${MSVC_VERSION})
            target_compile_options(${PROJECT_NAME} PUBLIC "/MP" "/bigobj")
            set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
        endif() 
    
    elseif(UNIX)
        # Change these to point to the right location for a particular machine
        #set(HOME "$ENV{HOME}" CACHE INTERNAL "Copied from environment variable")
        #add_subdirectory(${DLIB_ROOT}/dlib dlib_build)
        
        #find_package(X11 REQUIRED)
        add_library(${LIB_NAME} SHARED ${SOURCES})
        set_target_properties(${LIB_NAME} PROPERTIES VERSION ${PROJECT_VERSION})
        #target_link_libraries(${LIB_NAME} "${DLIB_ROOT}/dlib/build/libdlib.so" ${X11_LIBRARIES})
        #target_link_libraries(${LIB_NAME} ${X11_LIBRARIES})
        #target_link_libraries(${LIB_NAME} dlib::dlib ${X11_LIBRARIES})
        target_link_libraries(${LIB_NAME} dlib::dlib)
        #add_definitions("-fPIC")        
        
    elseif(APPLE)
        message(FATAL_ERROR "Apple -> Not Even!")
        
    endif()

else()
    message("Building MNIST executable...")

    add_executable(${LIB_NAME} ${SOURCES})
    target_link_libraries(${LIB_NAME} dlib::dlib)
    
    if(MSVC AND (${CMAKE_VERSION} VERSION_GREATER "3.6.0"))
        message("MSVC Version: " ${MSVC_VERSION})
        target_compile_options(${PROJECT_NAME} PUBLIC "/MP" "/bigobj")
        set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
    endif() 
    
endif()


message(" ")
message(STATUS "Build Complete!")
message(" ")



