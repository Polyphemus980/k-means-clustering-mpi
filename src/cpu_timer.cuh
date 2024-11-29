#ifndef CPU_TIMER_H
#define CPU_TIMER_H

#include <chrono>
#include <cstdint>
#include <string>

#include "base_timer.cuh"

namespace CpuTimer
{
    class Timer : BaseTimer::Timer
    {
    private:
        std::chrono::high_resolution_clock::time_point _start{};
        std::chrono::high_resolution_clock::time_point _end{};
        int64_t _timeInMS{};

    public:
        void start();
        void end();
        void printResult(const char *s);
    };
}

#endif // CPU_TIMER_H