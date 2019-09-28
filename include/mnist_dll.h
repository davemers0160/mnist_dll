#ifndef MNIST_DLL_H
#define MNIST_DLL_H

//#define EXTERN_C
//#include <cstdint>
//#include <string>
//#include <vector>

#ifdef MNIST_DLL_EXPORTS
#define MNIST_DLL_API __declspec(dllexport)
#else
#define MNIST_DLL_API __declspec(dllimport)
#endif

struct layer_struct
{
    unsigned int k;
    unsigned int n;
    unsigned int nr;
    unsigned int nc;
    unsigned int size;
};

#ifdef __cplusplus
extern "C" {
#endif
// This function will initialize the network and load the required weights
MNIST_DLL_API void init_net(const char *net_name);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
// This function will take an grayscale image in std::vector<uint8_t> row major order
// as an input and produce a resulting classification of the image.  The input must be 28*28
MNIST_DLL_API unsigned int run_net(unsigned char input[], unsigned int nr, unsigned int nc);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
    // This function will output a vector of the output layer for the final classification layer
    //MNIST_DLL_API void get_layer_01(struct layer_struct &data, const float* &data_params);
    MNIST_DLL_API void get_layer_01(struct layer_struct *data, const float **data_params);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
    //MNIST_DLL_API void get_layer_02(struct layer_struct &data, const float* &data_params);
    MNIST_DLL_API void get_layer_02(struct layer_struct *data, const float **data_params);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
    //MNIST_DLL_API void get_layer_05(struct layer_struct &data, const float* &data_params);
    MNIST_DLL_API void get_layer_05(struct layer_struct *data, const float **data_params);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
    //MNIST_DLL_API void get_layer_08(struct layer_struct &data, const float* &data_params);
    MNIST_DLL_API void get_layer_08(struct layer_struct *data, const float **data_params);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
    //MNIST_DLL_API void get_layer_09(struct layer_struct &data, const float* &data_params);
    MNIST_DLL_API void get_layer_09(struct layer_struct *data, const float **data_params);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
    //MNIST_DLL_API void get_layer_12(struct layer_struct &data, const float* &data_params);
    MNIST_DLL_API void get_layer_12(struct layer_struct *data, const float **data_params);
#ifdef __cplusplus
}
#endif

#endif  // MNIST_DLL_H
