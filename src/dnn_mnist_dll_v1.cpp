#define _CRT_SECURE_NO_WARNINGS
// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example illustrating the use of the deep learning tools from the
    dlib C++ Library.  In it, we will train the venerable LeNet convolutional
    neural network to recognize hand written digits.  The network will take as
    input a small image and classify it as one of the 10 numeric digits between
    0 and 9.

    The specific network we will run is from the paper
        LeCun, Yann, et al. "Gradient-based learning applied to document recognition."
        Proceedings of the IEEE 86.11 (1998): 2278-2324.
    except that we replace the sigmoid non-linearities with rectified linear units. 

    These tools will use CUDA and cuDNN to drastically accelerate network
    training and testing.  CMake should automatically find them if they are
    installed and configure things appropriately.  If not, the program will
    still run but will be much slower to execute.
*/

#include <cstdint>
#include <iostream>
#include <string>

// Custom includes
#include "file_parser.h"

#include "mnist_net_v0.h"

// dlib includes
#include <dlib/dnn.h>
#include <dlib/data_io.h>

using namespace std;

//----------------------------------------------------------------------------------

//----------------------------------------------------------------------------------

int main(int argc, char** argv)
{

    const std::string os_file_sep = "/";
    std::string program_root;
    std::string net_directory; 

    // typedef std::chrono::duration<double> d_sec;
    // auto start_time = chrono::system_clock::now();
    // auto stop_time = chrono::system_clock::now();
    // auto elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

    // setup save variable locations
#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
    program_root = get_path(get_path(get_path(std::string(argv[0]), "\\"), "\\"), "\\") + os_file_sep;
    net_directory = program_root + "nets/";
#else    
    net_directory = program_root + "nets/";
#endif


    try
    {

        net_type test_net;
        //config_net(test_net, filter_num);
        std::string test_net_name = (net_directory + "nmnist_net_04_13_76_56.dat"); 
        dlib::deserialize(test_net_name) >> test_net;




        
    }
    catch(std::exception& e)
    {
        std::cout << std::endl << e.what() << std::endl;
    }

    std::cout << std::endl << "Program complete.  Press Enter to close." << std::endl;
    std::cin.ignore();
    return 0;

}   // end of main

