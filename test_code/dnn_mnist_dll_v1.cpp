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

#include "mnist_net_pso_14.h"

// dlib includes
#include <dlib/dnn.h>
#include <dlib/data_io.h>

using namespace std;

//----------------------------------------------------------------------------------

// DLL internal state variables:
net_type net;


//----------------------------------------------------------------------------------

//template <typename net_type>
void init_net(std::string net_name)
{
    //net_type net;
    /*std::string test_net_name = (net_directory + "mnist_net_pso_14_97.dat");*/
    dlib::deserialize(net_name) >> net;

    //return net;
}

//template <typename net_type>
//uint64_t run_net(net_type &net, std::vector<uint8_t> &input)
uint64_t run_net(std::vector<uint8_t> &input)
{
    dlib::matrix<uint8_t> ti = dlib::reshape(dlib::mat(input), 28, 28);
    return net(ti);
}


//template <typename net_type>
//std::vector<float> get_layer1(net_type &net)
std::vector<float> get_layer1()
{
    auto& lo5 = dlib::layer<1>(net).get_output();
    const float* fc10 = lo5.host();
    std::vector<float> pd(fc10, fc10 + lo5.k());
    return pd;
}


//----------------------------------------------------------------------------------

int main(int argc, char** argv)
{


    std::vector<uint8_t> two = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 106, 119, 195, 243, 255, 255, 255, 231, 178, 1, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 47, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 241, 255, 255, 255, 255, 255, 160, 184, 148, 194, 254, 255, 176, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 235, 255, 255, 179, 48, 0, 0, 0, 0, 0, 0, 151, 255, 17, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 40, 74, 19, 0, 0, 0, 0, 0, 0, 0, 173, 255, 255, 5, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 154, 255, 255, 255, 148, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 172, 255, 255, 255, 255, 202, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 195, 255, 255, 255, 255, 113, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 199, 255, 255, 255, 255, 201, 8, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 132, 255, 255, 255, 255, 251, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 214, 255, 255, 255, 255, 46, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 47, 255, 255, 255, 255, 179, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 147, 255, 255, 255, 255, 39, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 180, 255, 255, 255, 210, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 211, 255, 255, 255, 182, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 187, 255, 255, 255, 86, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 130, 255, 255, 255, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 255, 255, 255, 145, 0, 0, 0, 17, 78, 142, 219, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 255, 255, 255, 199, 159, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 134, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 229, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 116, 255, 255, 255, 255, 255, 251, 255, 147, 127, 106, 67, 0, 0, 0, 15, 230, 144, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 38, 138, 59, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };



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

    std::vector<std::string> test_images = { "0_28.png", "1_28.png", "2_28.png", "3_28.png", "4_28.png", "5_28.png", "6_28.png", "7_28.png", "8_28.png", "9_28.png"};

    try
    {

        //net_type test_net;
        std::string test_net_name = (net_directory + "mnist_net_pso_14_97.dat"); 
        //dlib::deserialize(test_net_name) >> test_net;

        //auto test_net = init_net(test_net_name);
        init_net(test_net_name);

        //dlib::matrix<uint8_t> ti;
        //dlib::load_image(ti, "D:/Projects/mnist/data/test/" + test_images[2]);
        //auto res = test_net(ti);
        auto ti = two.data();

        //uint64_t res = run_net(test_net, two);
        uint64_t res = run_net(two);

        std::cout << "Image: " << test_images[2] << ", Result: " << res << std::endl;
            
        //std::vector<float> fc3 = get_layer1(test_net);
        std::vector<float> fc3 = get_layer1();

        dlib::matrix<float> fc3_mat = dlib::reshape(dlib::mat(fc3), 2,5);

   
    }
    catch(std::exception& e)
    {
        std::cout << std::endl << e.what() << std::endl;
    }

    std::cout << std::endl << "Program complete.  Press Enter to close." << std::endl;
    std::cin.ignore();
    return 0;

}   // end of main

