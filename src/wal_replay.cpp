#include "wal_replay.hpp"
#include "index_utils.hpp"
#include <algorithm>
#include <chrono>
#include <shared_mutex>

static std::string wal_space_to_string(WalSpaceType spaceType) {
  switch (spaceType) {
  case WalSpaceType::L2:
    return "L2";
  case WalSpaceType::GEODEGREES:
    return "GEODEGREES";
  default:
    return "IP";
  }
}

WalHeader makeWalHeader(const nlohmann::json &settings) {
  WalHeader h;
  h.dimension = settings.at("dimension").get<int32_t>();
  h.M = settings.value("M", 16);
  h.efConstruction = settings.value("efConstruction", 512);
  std::string space = settings.value("spaceType", "IP");
  if (space == "L2")
    h.spaceType = WalSpaceType::L2;
  else if (space == "GEODEGREES")
    h.spaceType = WalSpaceType::GEODEGREES;
  else
    h.spaceType = WalSpaceType::IP;
  h.mrlScanDim = settings.value("mrlScanDim", 0);
  std::string vt = settings.value("vectorType", "FLOAT32");
  if (vt == "FLOAT16")
    h.vectorType = WalVectorType::FLOAT16;
  else if (vt == "BFLOAT16")
    h.vectorType = WalVectorType::BFLOAT16;
  else
    h.vectorType = WalVectorType::FLOAT32;
  return h;
}

std::shared_ptr<IndexContext> contextFromWalHeader(const WalHeader &header, const std::string &indexName) {
  std::string spaceStr = wal_space_to_string(header.spaceType);
  std::string vtStr = "FLOAT32";
  if (header.vectorType == WalVectorType::FLOAT16)
    vtStr = "FLOAT16";
  else if (header.vectorType == WalVectorType::BFLOAT16)
    vtStr = "BFLOAT16";

  BuiltSpace bs = build_space(spaceStr, vtStr, header.dimension, header.mrlScanDim);
  auto *index = new hnswlib::HierarchicalNSW<float>(bs.space, DEFAULT_INDEX_SIZE, header.M, header.efConstruction, 42, true);

  auto ctx = std::make_shared<IndexContext>();
  ctx->index = index;
  ctx->isMrl = bs.isMrl;
  ctx->mrlScanDim = header.mrlScanDim;
  ctx->mrlFullDistFunc = bs.fullDistFunc;
  ctx->mrlFullDistFuncParam = bs.fullDistFuncParam;

  nlohmann::json settings;
  settings["indexName"] = indexName;
  settings["dimension"] = header.dimension;
  settings["spaceType"] = spaceStr;
  settings["vectorType"] = vtStr;
  settings["efConstruction"] = header.efConstruction;
  settings["M"] = header.M;
  settings["mrlScanDim"] = header.mrlScanDim;
  ctx->settings = settings;
  ctx->dataStore = new DataStore();
  return ctx;
}

void startBackgroundWalReplay(IndexContext *ctx, const std::string &indexName, ResolvedWal resolved, const std::string &walPath,
                              int fsyncIntervalMs) {
  ctx->walReplayTotalAdds.store(resolved.adds.size(), std::memory_order_relaxed);
  ctx->walReplayedCount.store(0, std::memory_order_relaxed);

  auto replay = [ctx, indexName, adds = std::move(resolved.adds), deletes = std::move(resolved.deletes),
                 updates = std::move(resolved.updates), walPath, fsyncIntervalMs]() {
    try {
      std::string vectorType = ctx->settings.value("vectorType", "FLOAT32");

      // pre-resize once to fit all adds. resizeIndex reallocates the graph
      // arrays, which is NOT safe to run concurrently with searches (a search
      // reading the old buffers mid-realloc can dereference freed memory and
      // crash later when it traverses a stale link). Hold the exclusive lock so
      // live search traffic during replay is briefly blocked across the resize;
      // the concurrent addPoint phase below stays lock-free (hnswlib supports
      // concurrent add + search once the index is sized).
      size_t needed = ctx->index->cur_element_count + adds.size() + DEFAULT_INDEX_RESIZE_HEADROOM;
      if (needed > ctx->index->max_elements_) {
        size_t newMax = static_cast<size_t>(static_cast<float>(needed) * (1.0f + INDEX_GROWTH_FACTOR) + 1);
        std::unique_lock<std::shared_mutex> resizeLock(ctx->mutex);
        ctx->index->resizeIndex(static_cast<int>(newMax));
      }

      // progress reporter thread
      std::atomic<bool> replayDone{false};
      size_t totalAdds = adds.size();
      std::thread progressThread;
      if (totalAdds >= 1000) {
        progressThread = std::thread([ctx, &replayDone, totalAdds, &indexName]() {
          size_t lastReported = 0;
          while (!replayDone.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            size_t current = ctx->walReplayedCount.load(std::memory_order_relaxed);
            if (current > lastReported) {
              int pct = static_cast<int>(current * 100 / totalAdds);
              LOG("wal") << "index=" << indexName << " replay progress: " << current << "/" << totalAdds << " (" << pct << "%)"
                         << std::endl;
              lastReported = current;
            }
          }
        });
      }

      unsigned numThreads = std::thread::hardware_concurrency();
      if (numThreads == 0)
        numThreads = 4;
      if (numThreads > adds.size())
        numThreads = static_cast<unsigned>(adds.size());

      if (numThreads <= 1 || adds.size() < 100) {
        for (const auto &ra : adds) {
          if (ctx->walReplayCancelled.load(std::memory_order_relaxed))
            break;
          addPointToIndex(ctx->index, vectorType, ra.docId, ra.vector);
          ctx->dataStore->set(ra.docId, ra.metadata);
          ctx->walReplayedCount.fetch_add(1, std::memory_order_relaxed);
        }
      } else {
        std::vector<std::thread> threads;
        threads.reserve(numThreads);
        size_t chunkSize = (adds.size() + numThreads - 1) / numThreads;

        for (unsigned t = 0; t < numThreads; t++) {
          size_t start = t * chunkSize;
          size_t end = std::min(start + chunkSize, adds.size());
          if (start >= end)
            break;
          threads.emplace_back([ctx, &adds, &vectorType, start, end]() {
            for (size_t i = start; i < end; i++) {
              if (ctx->walReplayCancelled.load(std::memory_order_relaxed))
                break;
              const auto &ra = adds[i];
              addPointToIndex(ctx->index, vectorType, ra.docId, ra.vector);
              ctx->dataStore->set(ra.docId, ra.metadata);
              ctx->walReplayedCount.fetch_add(1, std::memory_order_relaxed);
            }
          });
        }
        for (auto &th : threads) {
          th.join();
        }
      }

      replayDone.store(true, std::memory_order_relaxed);
      if (progressThread.joinable())
        progressThread.join();

      if (ctx->walReplayCancelled.load(std::memory_order_relaxed)) {
        LOG("wal") << "index=" << indexName << " replay cancelled" << std::endl;
        ctx->replayingWal.store(false, std::memory_order_release);
        return;
      }

      // true deletion with graph repair; runs after all adds have joined. Holds
      // the exclusive lock because removePoint rewires the graph and is not safe
      // to run concurrently with live searches during replay.
      if (!deletes.empty()) {
        std::unique_lock<std::shared_mutex> delLock(ctx->mutex);
        for (uint32_t docId : deletes) {
          if (ctx->walReplayCancelled.load(std::memory_order_relaxed))
            break;
          try {
            ctx->index->removePoint(docId);
          } catch (...) {
          }
          ctx->dataStore->remove(docId);
        }
      }

      for (const auto &[docId, metadata] : updates) {
        if (ctx->dataStore->contains(docId)) {
          ctx->dataStore->set(docId, metadata);
        }
      }

      LOG("wal") << "index=" << indexName << " replay complete, " << adds.size() << " adds, " << deletes.size() << " deletes, "
                 << updates.size() << " updates" << std::endl;

      // create fresh WAL at same path
      WalHeader wh = makeWalHeader(ctx->settings);
      auto *wal = new WriteAheadLog(walPath, wh);
      wal->startFsyncThread(fsyncIntervalMs);
      ctx->wal = wal;

      ctx->replayingWal.store(false, std::memory_order_release);
    } catch (const std::exception &e) {
      LOG("ERROR") << "index=" << indexName << " WAL replay error: " << e.what() << std::endl;
      {
        std::lock_guard<std::mutex> bufLock(ctx->bufferMutex);
        ctx->walReplayError = e.what();
      }
      ctx->replayingWal.store(false, std::memory_order_release);
    }
  };
  ctx->walReplayThread = std::thread(std::move(replay));
}
