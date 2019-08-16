#ifndef MNIST_DLL_H
#define MNIST_DLL_H



#ifdef MNIST_DLL_EXPORTS
#define MNIST_DLL_API __declspec(dllexport)
#else
#define MNIST_DLL_API __declspec(dllimport)
#endif


struct layer_struct
{
    uint64_t k;
    uint64_t n;
    uint64_t nr;
    uint64_t nc;
    uint64_t size;
    const float *params;
};



// This function will initialize the network and load the required weights
extern "C" MNIST_DLL_API void init_net(const char *net_name);

// This function will take an grayscale image in std::vector<uint8_t> row major order
// as an input and produce a resulting classification of the image.  The input must be 28*28
extern "C" MNIST_DLL_API uint64_t run_net(uint8_t input[], uint32_t nr, uint32_t nc);

// This function will output a vector of the output layer for the final
// classification layer
//extern "C" MNIST_DLL_API std::vector<float> get_layer1(void);
extern "C" MNIST_DLL_API const float* get_layer1(void);

//extern "C" MNIST_DLL_API const float* get_layer12(void);
extern "C" MNIST_DLL_API void get_layer12(struct layer_struct &data);

#endif  // MNIST_DLL_H
