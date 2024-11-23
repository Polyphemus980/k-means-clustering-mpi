#include <iostream>

#include "utils.cuh"

namespace Utils
{
    void usage(const char *s)
    {
        std::cout << "Usage: " << s << " data_format computation_method input_file output_file\n";
        std::cout << "data_format - txt or bin\n";
        std::cout << "computation_method - gpu1, gpu2 or cpu\n";
        std::exit(1);
    }
} // Utils