#include "gpu_timer.cuh"
#include "utils.cuh"

namespace GpuTimer
{
    void Timer::start()
    {
        CHECK_CUDA(cudaEventCreate(&this->_start));
        CHECK_CUDA(cudaEventCreate(&this->_end));
        CHECK_CUDA(cudaEventRecord(this->_start));
    }

    void Timer::end()
    {
        CHECK_CUDA(cudaEventRecord(this->_end));
        CHECK_CUDA(cudaEventSynchronize(this->_end));
        CHECK_CUDA(cudaEventElapsedTime(&this->_timeInMS, this->_start, this->_end));

        CHECK_CUDA(cudaEventDestroy(this->_start));
        CHECK_CUDA(cudaEventDestroy(this->_end));
    }

    void Timer::printResult(const char *s)
    {
        printf("[GPU TIMER] Step: \"%s\", Time: %f ms\n", s, this->_timeInMS);
    }
} // GpuTimer