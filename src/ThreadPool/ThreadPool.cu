//
// Created by steve on 4/9/2023.
//

#include "ThreadPool.cuh"

#pragma once

NaNL::ThreadPool *NaNL::ThreadPool::getInstance() {
    // lock out other threads
    std::lock_guard<std::mutex> lock(_instanceMutex);

    if (_instance == nullptr) {
        _instance = new ThreadPool();
        _instance->populatePool();
    }
    return _instance;
}

void NaNL::ThreadPool::populatePool() {
    const uint32_t totalThreads = std::thread::hardware_concurrency(); // Max # of threads the system supports
    for (uint32_t i = 0; i < totalThreads; i++) {
        _threadPool.emplace_back(&ThreadPool::ThreadLoop, this);
    }
}

void NaNL::ThreadPool::ThreadLoop() {
    std::thread::id threadId = std::this_thread::get_id();
    unsigned int threadVal = *static_cast<int64_t*>(static_cast<void*>( &threadId));

    {
        std::unique_lock<std::mutex> printLock(_queueMutex);
        std::cout << "[threadId: " << threadVal << "] waiting for jobs." << std::endl;
    }

    while (true) {
        std::function<void()> job;
        {
            std::unique_lock<std::mutex> queueLock(_queueMutex);
            _mutexCondition.wait(queueLock, [this] {
                return !_functionQueue.empty() || _shouldStop;
            });

            // thread pool is requested to end.
            if (_shouldStop) {
                return;
            }

            job = _functionQueue.front();
            _functionQueue.pop();
        }
        job();
    }
}

template<class F, class... Args>
auto NaNL::ThreadPool::queue(F&& f, Args &&... args) -> std::future<typename std::invoke_result<F, Args...>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;

    auto job = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    std::future<return_type> result = job->get_future();
    {
        std::unique_lock<std::mutex> lock(_queueMutex);
        this->_functionQueue.emplace([job]() {(*job)(); });
    }
    _mutexCondition.notify_one();
    return result;
}

size_t NaNL::ThreadPool::getAllocatedThreads() {
    return _threadPool.size();
}
