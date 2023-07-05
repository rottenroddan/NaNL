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
#include <type_traits>

namespace NaNL {

    class ThreadPool {
    private:
        static ThreadPool* _instance;
        static std::mutex _queueMutex;
        static std::mutex _instanceMutex;
        static std::condition_variable _mutexCondition;

        std::deque<std::thread> _threadPool;
        std::queue<std::function<void()>> _functionQueue;
        bool _shouldStop = false;

        inline void populatePool();
        inline void ThreadLoop();

    protected:
        ThreadPool() = default;

    public:
        void operator=(const ThreadPool&) = delete;
        static ThreadPool* getInstance();

        template<class F, class... Args>
        inline auto queue(F&& f, Args&&... args) -> std::future<decltype(f(args...))> {
            using return_type = decltype(f(args...));

            auto job = std::make_shared<std::packaged_task<return_type()>>(
                    std::bind(std::forward<F>(f), std::forward<Args>(args)...)
            );

            std::future<return_type> result = job->get_future();
            {
                std::unique_lock<std::mutex> lock(_queueMutex);
                this->_functionQueue.emplace([job]() { (*job)(); });
            }
            _mutexCondition.notify_one();
            return result;
        }

        uint64_t getAllocatedThreads();
    };

} // NaNL

#endif // NANL_THREADPOOL_CUH

