//
// Created by steve on 4/9/2023.
//

#ifndef NANL_THREADPOOL_CUH
#define NANL_THREADPOOL_CUH

#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

#include "../StaticBlock/StaticBlock.cuh"

namespace NaNL {

    /**
     * ThreadPool is a singleton designed for queuing functions
     * for a pool of threads to complete. Accessed using the
     * getInstance() method; which returns a global instance.
     * Use queue method to start a thread on a function. Returns
     * a future object
     */
    class ThreadPool {
    private:
        inline static ThreadPool *_instance;
        inline static std::mutex _queueMutex;
        inline static std::mutex _instanceMutex;
        inline static std::condition_variable _mutexCondition;

        std::deque<std::thread> _threadPool;
        std::queue<std::function<void()>> _functionQueue;
        bool _shouldStop = false;

        inline void populatePool();
        inline void ThreadLoop();
    protected:
        inline ThreadPool() { ; }
    public:
        void operator=(const ThreadPool&) = delete;
        inline static ThreadPool* getInstance();

        template<class F, class... Args>
        inline auto queue(F&&, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type>;
        inline uint64_t getAllocatedThreads();

    };

} // NaNL

#include "ThreadPool.cu"

#endif //NANL_THREADPOOL_CUH
