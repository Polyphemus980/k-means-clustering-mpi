#ifndef CPU_TIMER_H
#define CPU_TIMER_H

#include <chrono>
#include <cstdint>
#include <string>

namespace CpuTimer
{
    class Timer
    {
    private:
        std::chrono::high_resolution_clock::time_point _start{};
        std::chrono::high_resolution_clock::time_point _end{};
        int64_t _timeInMS{};

    public:
        void start();
        void end();
        int64_t timeInMS();
        void printResult(const char *s);
    };
}

#endif // CPU_TIMER_H