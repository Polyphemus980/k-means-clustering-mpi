#include <cstdio>

namespace Utils
{
    void usage(const char *s)
    {
        fprintf(stderr, "Usage: %s data_format computation_method input_file output_file\n", s);
        fprintf(stderr, "data_format - txt or bin\n");
        fprintf(stderr, "computation_method - gpu1, gpu2 or cpu\n");
        std::exit(1);
    }
} // Utils