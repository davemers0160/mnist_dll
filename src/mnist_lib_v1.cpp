#define _CRT_SECURE_NO_WARNINGS

#include <cstdint>
#include <string>
#include <vector>

// Custom includes
#include "mnist_lib.h"
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
unsigned int run_net(unsigned char* input, unsigned int nr, unsigned int nc)
{
    dlib::matrix<uint8_t> ti = dlib::pointer_to_matrix(input, nr, nc); //dlib::reshape(dlib::mat(std::vector<uint8_t>(input, input+length), 28, 28);
    return net(ti);
}

//----------------------------------------------------------------------------------
void get_layer_01(struct layer_struct *data, const float **data_params)
{
    auto& lo = dlib::layer<1>(net).get_output();
    data->k = lo.k();
    data->n = lo.num_samples();
    data->nr = lo.nr();
    data->nc = lo.nc();
    data->size = lo.size();
    *data_params = lo.host();
}

//----------------------------------------------------------------------------------
void get_layer_02(struct layer_struct *data, const float **data_params)
{
    auto& lo = dlib::layer<2>(net).get_output();
    data->k = lo.k();
    data->n = lo.num_samples();
    data->nr = lo.nr();
    data->nc = lo.nc();
    data->size = lo.size();
    *data_params = lo.host();
}

//----------------------------------------------------------------------------------
void get_layer_05(struct layer_struct *data, const float **data_params)
{
    auto& lo = dlib::layer<5>(net).get_output();
    data->k = lo.k();
    data->n = lo.num_samples();
    data->nr = lo.nr();
    data->nc = lo.nc();
    data->size = lo.size();
    *data_params = lo.host();
}

//----------------------------------------------------------------------------------
void get_layer_08(struct layer_struct *data, const float **data_params)
{
    auto& lo = dlib::layer<8>(net).get_output();
    data->k = lo.k();
    data->n = lo.num_samples();
    data->nr = lo.nr();
    data->nc = lo.nc();
    data->size = lo.size();
    *data_params = lo.host();
}

//----------------------------------------------------------------------------------
void get_layer_09(struct layer_struct *data, const float **data_params)
{
    auto& lo = dlib::layer<9>(net).get_output();
    data->k = lo.k();
    data->n = lo.num_samples();
    data->nr = lo.nr();
    data->nc = lo.nc();
    data->size = lo.size();
    *data_params = lo.host();
}

//----------------------------------------------------------------------------------
//void get_layer12(struct layer_struct &data, )
void get_layer_12(struct layer_struct *data, const float **data_params)
{
    auto& lo = dlib::layer<12>(net).get_output();
    data->k = lo.k();
    data->n = lo.num_samples();
    data->nr = lo.nr();
    data->nc = lo.nc();
    data->size = lo.size();
    *data_params = lo.host();
}

