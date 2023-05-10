//
// Created by steve on 2/26/2023.
//

#ifndef NANL_LOGGER_CUH
#define NANL_LOGGER_CUH


#include <iomanip>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <stack>
#include <string>
#include <sstream>
#include <limits.h>

namespace NaNL {
    class Logger {
    private:
        std::mutex _print_mutex;
        long ends = 0;
        long starts = 0;

        class LoggerDetails {
        public:
            int long _threadId = -1;
            std::string _function;
            std::string _description;
            std::chrono::time_point<std::chrono::steady_clock> startTimepoint;
            std::chrono::time_point<std::chrono::steady_clock> endTimepoint;
        };

        inline static Logger *instance;
        inline static std::mutex mutex;
        std::unordered_map<int long long, std::deque<LoggerDetails>> activeLogs;
        std::unordered_map<int long long, std::deque<LoggerDetails>> inActiveLogs;

    protected:
        inline Logger() { ; }

    public:
        // remove since singleton
        Logger(Logger &other) = delete;

        void operator=(const Logger&) = delete;

        inline static Logger* GetInstance();
        inline void begin(int64_t _threadId, std::string _function, std::string _description);
        inline void record(int64_t _threadId, std::string _function, std::string _description);
        inline void end(int64_t _threadId);
        inline void log(int64_t _threadId);
    };
}

#include "Logger.cu"

#ifdef PERFORMANCE_LOGGING
#define PERFORMANCE_LOGGING_BEGIN \
std::thread::id threadId = std::this_thread::get_id(); \
unsigned int threadVal = *static_cast<int64_t*>(static_cast<void*>( &threadId)); \
NaNL::Logger::GetInstance()->begin(threadVal, __FUNCTION__, "");

#define PERFORMANCE_LOGGING_START \
std::thread::id threadId = std::this_thread::get_id(); \
unsigned int threadVal = *static_cast<int64_t*>(static_cast<void*>( &threadId)); \
NaNL::Logger::GetInstance()->record(threadVal, __FUNCTION__, typeid(*(this)).name());

#define PERFORMANCE_LOGGING_END \
NaNL::Logger::GetInstance()->end(threadVal);


#define PERFORMANCE_LOGGING_LOG \
NaNL::Logger::GetInstance()->log(threadVal);
#endif // PERFORMANCE_LOGGING
#endif //NANL_LOGGER_CUH


