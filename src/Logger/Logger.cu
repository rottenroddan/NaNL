//
// Created by steve on 2/26/2023.
//
#include "Logger.cuh"

#pragma once

namespace NaNL {
    namespace {
        double calculateTimePercentage(std::chrono::time_point<std::chrono::steady_clock> totalStartTime,
                                       std::chrono::time_point<std::chrono::steady_clock> totalEndTime,
                                       std::chrono::time_point<std::chrono::steady_clock> sliceStartTime,
                                       std::chrono::time_point<std::chrono::steady_clock> sliceEndTime) {
            return  ((double)(sliceEndTime - sliceStartTime).count() / (double)(totalEndTime - totalStartTime).count()) * 100.00;
        }
    }

    Logger *NaNL::Logger::GetInstance() {
        // lock out other threads
        std::lock_guard<std::mutex> lock(mutex);
        if (instance == nullptr) {
            instance = new Logger();
        }
        return instance;
    }

    void Logger::begin(int64_t _threadId, std::string _function, std::string _description) {
        LoggerDetails tempLog;
        tempLog._function = _function;
        tempLog._threadId = _threadId;
        tempLog._description = _description;

        std::deque<LoggerDetails> tempDeque;
        tempDeque.push_back(tempLog);

        activeLogs.insert_or_assign(_threadId, tempDeque);
        (activeLogs[_threadId].end() - 1)->startTimepoint = std::chrono::high_resolution_clock::now();
    }

    void Logger::record(int64_t _threadId, std::string _function, std::string _description) {
        starts++;
        LoggerDetails tempLog;
        tempLog._function = _function;
        tempLog._threadId = _threadId;
        tempLog._description = _description;

        activeLogs[_threadId].push_back(tempLog);
        (activeLogs[_threadId].end() - 1)->startTimepoint = std::chrono::high_resolution_clock::now();
    }

    void Logger::end(int64_t _threadId) {
        ends++;
        // immediately set the endTimepoint clock to now.
        (activeLogs[_threadId].end() - 1)->endTimepoint = std::chrono::high_resolution_clock::now();

        if (inActiveLogs.find(_threadId) == inActiveLogs.end()) {
            LoggerDetails tempLog = *(activeLogs[_threadId].end() - 1);

            std::deque<LoggerDetails> tempDeque;
            tempDeque.push_back(tempLog);

            inActiveLogs.insert_or_assign(_threadId, tempDeque);
        } else {
            inActiveLogs[_threadId].emplace_back(*(activeLogs[_threadId].end() - 1));
        }
        activeLogs[_threadId].pop_back();
    }

    void Logger::log(int64_t _threadId) {
        std::mutex _mutex;
        std::lock_guard<std::mutex> lock(_print_mutex);

        auto threadLogs = inActiveLogs[_threadId];

        if (threadLogs.size() != 0) {
            std::stringstream ss;

            std::cout << "[threadId: " << std::right << std::setw(5) << _threadId << "] " << threadLogs[threadLogs.size() - 1]._function << " : "
                      << std::chrono::duration_cast<std::chrono::nanoseconds>(
                              threadLogs[threadLogs.size() - 1].endTimepoint -
                              threadLogs[threadLogs.size() - 1].startTimepoint).count() << "ns "
                      << "(" << (std::chrono::duration_cast<std::chrono::nanoseconds>(
                    threadLogs[threadLogs.size() - 1].endTimepoint -
                    threadLogs[threadLogs.size() - 1].startTimepoint).count() / 1000000000.00) << "s)"
                      << std::endl;

            // don't print last log as that's the top of the stack where logging starts.
            for (int64_t i = 0; i < threadLogs.size() - 1; i++) {
                ss << "  [threadId: " << std::right << std::setw(5) << _threadId << "] " << std::left << std::setw(55)
                   << threadLogs[i]._function << " : "
                   << std::right << std::setw(12) << std::chrono::duration_cast<std::chrono::nanoseconds>(
                        threadLogs[i].endTimepoint - threadLogs[i].startTimepoint).count() << "ns "
                   << "(" << std::setprecision(7) << std::fixed << std::setw(11)
                   << (threadLogs[i].endTimepoint - threadLogs[i].startTimepoint).count() / 1000000000.00
                   << "s) (" << std::setprecision(4) << std::setw(7) << std::fixed
                   << calculateTimePercentage(threadLogs[threadLogs.size() - 1].startTimepoint,
                                              threadLogs[threadLogs.size() - 1].endTimepoint,
                                              threadLogs[i].startTimepoint,
                                              threadLogs[i].endTimepoint) << "%)"
                   << std::endl;

                std::cout << ss.str();

                ss.clear();
                ss = std::stringstream("");
            }
            std::cout << std::endl;
        }

        inActiveLogs[_threadId].clear();
    }

}