// Auto generated c++ header file for PSO testing
// Iteration Number: 59
// Population Number: 14

#ifndef NET_DEFINITION_H
#define NET_DEFINITION_H

#include <cstdint>
#include <string>

#include "dlib/dnn.h"
#include "dlib/dnn/core.h"

//#include "dlib_elu.h"
//#include "dlib_srelu.h"

extern const std::string version = "pso_14_59";
//-----------------------------------------------------------------

typedef struct{
    uint32_t iteration;
    uint32_t pop_num;
} pso_struct;
pso_struct pso_info = {59, 14};

//-----------------------------------------------------------------

using net_type = dlib::loss_multiclass_log<
    dlib::fc<10, 
    dlib::sig<    dlib::bn_con<    dlib::fc<461, 
    dlib::htan<    dlib::bn_con<    dlib::fc<656, 
    dlib::max_pool<2, 2, 2, 2,    dlib::htan<    dlib::con<110, 5, 1, 1, 1, 
    dlib::max_pool<2, 2, 2, 2,    dlib::prelu<    dlib::bn_con<    dlib::con<133, 7, 5, 1, 1, 
    dlib::input<dlib::matrix<unsigned char>>
    >>>>>>>>>>>>>>>;

//-----------------------------------------------------------------

inline std::ostream& operator<< (std::ostream& out, const pso_struct& item)
{
    out << "------------------------------------------------------------------" << std::endl;
    out << "PSO Info: " << std::endl;
    out << "  Iteration: " << item.iteration << std::endl;
    out << "  Population Number: " << item.pop_num << std::endl;
    out << "------------------------------------------------------------------" << std::endl;
    return out;
}

#endif 

