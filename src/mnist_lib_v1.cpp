#define _CRT_SECURE_NO_WARNINGS

#include <cstdint>
#include <string>
#include <vector>

// Custom includes
#include "file_ops.h"
#include "mnist_lib.h"
#include "mnist_net_pso_14.h"

// dlib includes
#include <dlib/dnn.h>
#include <dlib/data_io.h>

//----------------------------------------------------------------------------------
// DLL internal state variables:
net_type net;

//----------------------------------------------------------------------------------
void init_net(const char *net_name)
{   
    dlib::deserialize(net_name) >> net;
}

//----------------------------------------------------------------------------------
void run_net(unsigned char* input, unsigned int nr, unsigned int nc, unsigned int *res)
{
    dlib::matrix<uint8_t> ti = dlib::pointer_to_matrix(input, nr, nc); 
    *res = net(ti);
}

//----------------------------------------------------------------------------------
void get_layer_01(struct layer_struct *data, const float* &data_params)
{
    auto& lo = dlib::layer<1>(net).get_output();
    data->k = lo.k();
    data->n = lo.num_samples();
    data->nr = lo.nr();
    data->nc = lo.nc();
    data->size = lo.size();
    data_params = lo.host();
}

//----------------------------------------------------------------------------------
void get_layer_02(struct layer_struct *data, const float* &data_params)
{
    auto& lo = dlib::layer<2>(net).get_output();
    data->k = lo.k();
    data->n = lo.num_samples();
    data->nr = lo.nr();
    data->nc = lo.nc();
    data->size = lo.size();
    data_params = lo.host();
}

//----------------------------------------------------------------------------------
void get_layer_05(struct layer_struct *data, const float* &data_params)
{
    auto& lo = dlib::layer<5>(net).get_output();
    data->k = lo.k();
    data->n = lo.num_samples();
    data->nr = lo.nr();
    data->nc = lo.nc();
    data->size = lo.size();
    data_params = lo.host();
}

//----------------------------------------------------------------------------------
void get_layer_08(struct layer_struct *data, const float* &data_params)
{
    auto& lo = dlib::layer<8>(net).get_output();
    data->k = lo.k();
    data->n = lo.num_samples();
    data->nr = lo.nr();
    data->nc = lo.nc();
    data->size = lo.size();
    data_params = lo.host();
}

//----------------------------------------------------------------------------------
void get_layer_09(struct layer_struct *data, const float* &data_params)
{
    auto& lo = dlib::layer<9>(net).get_output();
    data->k = lo.k();
    data->n = lo.num_samples();
    data->nr = lo.nr();
    data->nc = lo.nc();
    data->size = lo.size();
    data_params = lo.host();
}

//----------------------------------------------------------------------------------
void get_layer_12(struct layer_struct* data, const float* &data_params)
{
    auto& lo = dlib::layer<12>(net).get_output();
    data->k = lo.k();
    data->n = lo.num_samples();
    data->nr = lo.nr();
    data->nc = lo.nc();
    data->size = lo.size();
    data_params = lo.host();
}

//----------------------------------------------------------------------------------
// check to see if we are building the library or a standalone executable
#if !defined(BUILD_LIB)

int main(int argc, char** argv)
{
    uint32_t idx;
    std::string program_root;
    std::string net_directory;
    std::string image_directory;
    std::string test_net_name;
    unsigned int result;

    std::vector<std::string> test_images = { "0_28.png", "1_28.png", "2_28.png", "3_28.png", "4_28.png", "5_28.png", "6_28.png", "7_28.png", "8_28.png", "9_28.png" };

    // setup save variable locations
#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
    program_root = path_check(get_path(get_path(get_path(std::string(argv[0]), "\\"), "\\"), "\\"));
#else 
    program_root = get_ubuntu_path();
#endif

    net_directory = program_root + "nets/";
    image_directory = program_root + "images/";

    try
    {
        test_net_name = (net_directory + "mnist_net_pso_14_97.dat");

        // initialize the network
        init_net(test_net_name.c_str());

        dlib::matrix<uint8_t> ti;
        
        // run through some images to test the code
        for (idx = 0; idx < test_images.size(); ++idx)
        {
            dlib::load_image(ti, image_directory + test_images[idx]);
            unsigned char* input = &(ti)(0, 0);
            run_net(&(ti)(0, 0), ti.nr(), ti.nc(), &result);

            std::cout << test_images[idx] << ": " << idx << " - " << result << std::endl;
        }

    }
    catch (std::exception & e)
    {
        std::cout << std::endl;
        std::cout << e.what() << std::endl;
    }

    std::cout << std::endl << "Program complete.  Press Enter to close." << std::endl;
    std::cin.ignore();
    return 0;
}

#endif  // BUILD_LIB
