#ifndef BASE_TIMER_H
#define BASE_TIMER_H

namespace BaseTimer
{
    class Timer
    {
    public:
        void virtual start() = 0;
        void virtual end() = 0;
        void virtual printResult(const char *s) = 0;
    };
} // BaseTimer

#endif // BASE_TIMER_H