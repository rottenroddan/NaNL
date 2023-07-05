#pragma once
#include "ThreadPool.cuh"

namespace NaNL {

    ThreadPool* ThreadPool::_instance = nullptr;
    std::mutex ThreadPool::_queueMutex;
    std::mutex ThreadPool::_instanceMutex;
    std::condition_variable ThreadPool::_mutexCondition;

    void ThreadPool::populatePool() {
        const uint32_t totalThreads = std::thread::hardware_concurrency(); // Max # of threads the system supports
        for (uint32_t i = 0; i < totalThreads; i++) {
            _threadPool.emplace_back(&ThreadPool::ThreadLoop, this);
        }
    }

    void ThreadPool::ThreadLoop() {
        std::thread::id threadId = std::this_thread::get_id();
        unsigned int threadVal = *static_cast<int64_t*>(static_cast<void*>(&threadId));

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

    ThreadPool* ThreadPool::getInstance() {
        std::lock_guard<std::mutex> lock(_instanceMutex);

        if (_instance == nullptr) {
            _instance = new ThreadPool();
            _instance->populatePool();
        }
        return _instance;
    }

    uint64_t ThreadPool::getAllocatedThreads() {
        return _threadPool.size();
    }

} // NaNL
