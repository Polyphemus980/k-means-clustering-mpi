#include <cstdio>

#include "cpu_timer.cuh"

namespace CpuTimer
{
    void Timer::start()
    {
        this->_start = std::chrono::high_resolution_clock::now();
    }

    void Timer::end()
    {
        this->_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(this->_end - this->_start);
        this->_timeInMS = duration.count();
    }

    void Timer::printResult(const char *s)
    {
        printf("[CPU] Step: \"%s\", Time: %ld ms\n", s, this->_timeInMS);
    }
}