#ifndef GPU_TIMER_H
#define GPU_TIMER_H

#include "base_timer.cuh"

namespace GpuTimer
{

    class Timer : BaseTimer::Timer
    {
    private:
        cudaEvent_t _start{};
        cudaEvent_t _end{};
        float _timeInMS{};

    public:
        void start();
        void end();
        void printResult(const char *s);
    };

} // GpuTimer

#endif // GPU_TIMER_H