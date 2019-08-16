#define _CRT_SECURE_NO_WARNINGS

#include <cstdint>
#include <string>
#include <vector>

// Custom includes
#include "mnist_dll.h"
#include "mnist_net_pso_14.h"

// dlib includes
#include <dlib/dnn.h>
#include <dlib/data_io.h>

using namespace std;

//----------------------------------------------------------------------------------
// DLL internal state variables:
net_type net;

//----------------------------------------------------------------------------------
void init_net(const char *net_name)
{
    //net_type net;
    
    /*std::string test_net_name = (net_directory + "mnist_net_pso_14_97.dat");*/
    dlib::deserialize(net_name) >> net;

    //return net;
}

//----------------------------------------------------------------------------------
//uint64_t run_net(std::vector<uint8_t> &input)
uint64_t run_net(uint8_t input[], uint32_t nr, uint32_t nc)
{
    dlib::matrix<uint8_t> ti = dlib::pointer_to_matrix(input, nr, nc); //dlib::reshape(dlib::mat(std::vector<uint8_t>(input, input+length), 28, 28);
    return net(ti);
}

//----------------------------------------------------------------------------------
//std::vector<float> get_layer1(void)
const float* get_layer1(void)
{
    auto& lo = dlib::layer<1>(net).get_output();
    const float* params = lo.host();
    //std::vector<float> pd(fc10, fc10 + lo5.k());
    return params;
}

//----------------------------------------------------------------------------------
//const float* get_layer12(void)
void get_layer12(struct layer_struct &data)
{
    auto& lo = dlib::layer<12>(net).get_output();

    data.k = lo.k();
    data.n = lo.num_samples();
    data.nr = lo.nr();
    data.nc = lo.nc();
    data.size = lo.size();
    data.params = lo.host();
}

