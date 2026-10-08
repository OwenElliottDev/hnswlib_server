#ifndef INDEX_CONTEXT_HPP
#define INDEX_CONTEXT_HPP

#include "data_store.hpp"
#include "hnswlib/hnswlib.h"
#include "nlohmann/json.hpp"
#include "wal.hpp"
#include <atomic>
#include <condition_variable>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <map>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#define DEFAULT_INDEX_SIZE 100000
#define DEFAULT_INDEX_RESIZE_HEADROOM 10000
#define INDEX_GROWTH_FACTOR 2.0

struct UtcTime {
  friend std::ostream &operator<<(std::ostream &os, const UtcTime &) {
    std::time_t now = std::time(nullptr);
    std::tm *utc = std::gmtime(&now);
    return os << std::put_time(utc, "%Y-%m-%d %H:%M:%S UTC");
  }
};

#define LOG(src) std::cerr << "[" << UtcTime{} << "][" << src << "] "

// per-index guard to prevent concurrent addPoint with the same label
struct InFlightGuard {
  std::mutex mutex;
  std::condition_variable cv;
  std::unordered_set<int> ids;

  void acquire(int id) {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [&] { return ids.find(id) == ids.end(); });
    ids.insert(id);
  }

  void release(int id) {
    std::lock_guard<std::mutex> lock(mutex);
    ids.erase(id);
    cv.notify_all();
  }
};

struct InFlightLock {
  InFlightGuard &guard;
  int id;
  InFlightLock(InFlightGuard &guard, int id) : guard(guard), id(id) { guard.acquire(id); }
  ~InFlightLock() { guard.release(id); }
  InFlightLock(const InFlightLock &) = delete;
  InFlightLock &operator=(const InFlightLock &) = delete;
};

struct BufferedWrite {
  int id;
  std::vector<float> vector;
  std::map<std::string, FieldValue> metadata;
};

struct IndexContext {
  hnswlib::HierarchicalNSW<float> *index = nullptr;
  nlohmann::json settings;
  DataStore *dataStore = nullptr;
  WriteAheadLog *wal = nullptr;
  InFlightGuard *inFlightGuard = nullptr;

  // MRL (Matryoshka) state. When mrl is enabled the graph is scanned at
  // mrlScanDim leading dimensions and candidates are reranked at full
  // dimensionality with mrlFullDistFunc (owned by the index's MrlSpace).
  bool isMrl = false;
  int mrlScanDim = 0;
  hnswlib::DISTFUNC<float> mrlFullDistFunc = nullptr;
  void *mrlFullDistFuncParam = nullptr;

  std::shared_mutex mutex; // per-index R/W lock
  std::atomic<bool> resizing{false};
  std::vector<BufferedWrite> writeBuffer;
  std::mutex bufferMutex;
  std::thread resizeThread;

  // WAL replay state
  std::atomic<bool> replayingWal{false};
  std::atomic<size_t> walReplayedCount{0};
  std::atomic<size_t> walReplayTotalAdds{0};
  std::string walReplayError; // protected by bufferMutex
  std::thread walReplayThread;
  std::atomic<bool> walReplayCancelled{false};

  ~IndexContext() {
    // cancel and join WAL replay thread first
    walReplayCancelled.store(true, std::memory_order_release);
    if (walReplayThread.joinable()) {
      walReplayThread.join();
    }
    if (resizeThread.joinable()) {
      resizeThread.join();
    }
    if (wal) {
      wal->stopFsyncThread();
      delete wal;
    }
    delete inFlightGuard;
    delete dataStore;
    delete index;
  }
};

#endif // INDEX_CONTEXT_HPP
