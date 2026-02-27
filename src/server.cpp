#include "crow.h"
#include "data_store.hpp"
#include "filters.hpp"
#include "hnswlib/hnswlib.h"
#include "models.hpp"
#include "nlohmann/json.hpp"
#include "wal.hpp"
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define DEFAULT_INDEX_SIZE 100000
#define DEFAULT_INDEX_RESIZE_HEADROOM 10000
#define INDEX_GROWTH_FACTOR 2.0
#define EXACT_KNN_FILTER_PCT_MATCH_THRESHOLD 0.1

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

  std::shared_mutex mutex; // per-index R/W lock
  std::atomic<bool> resizing{false};
  std::vector<BufferedWrite> writeBuffer;
  std::mutex bufferMutex;
  std::thread resizeThread;

  ~IndexContext() {
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

std::unordered_map<std::string, std::shared_ptr<IndexContext>> contexts;
std::shared_mutex contextMapMutex; // protects the map itself (create/delete/load/list)

int walFsyncIntervalMs = 1000;

std::shared_ptr<IndexContext> getContext(const std::string &indexName) {
  std::shared_lock<std::shared_mutex> lock(contextMapMutex);
  auto it = contexts.find(indexName);
  if (it == contexts.end())
    return nullptr;
  return it->second;
}

hnswlib::SpaceInterface<float> *create_space(const std::string &spaceType, const std::string &vectorType, int dim) {
  if (vectorType == "FLOAT16") {
    if (spaceType == "IP")
      return new hnswlib::InnerProductFloat16Space(dim);
    return new hnswlib::L2Float16Space(dim);
  } else if (vectorType == "BFLOAT16") {
    if (spaceType == "IP")
      return new hnswlib::InnerProductBFloat16Space(dim);
    return new hnswlib::L2BFloat16Space(dim);
  } else {
    if (spaceType == "IP")
      return new hnswlib::InnerProductSpace(dim);
    return new hnswlib::L2Space(dim);
  }
}

std::string get_vector_type(const std::shared_ptr<IndexContext> &ctx) {
  if (ctx->settings.contains("vectorType")) {
    return ctx->settings["vectorType"].get<std::string>();
  }
  return "FLOAT32";
}

std::vector<uint16_t> floats_to_f16(const std::vector<float> &vec) {
  std::vector<uint16_t> result(vec.size());
  for (size_t i = 0; i < vec.size(); i++) {
    result[i] = hnswlib::float_to_half(vec[i]);
  }
  return result;
}

std::vector<uint16_t> floats_to_bf16(const std::vector<float> &vec) {
  std::vector<uint16_t> result(vec.size());
  for (size_t i = 0; i < vec.size(); i++) {
    result[i] = hnswlib::float_to_bfloat16(vec[i]);
  }
  return result;
}

std::vector<float> f16_to_floats(const std::vector<uint16_t> &vec) {
  std::vector<float> result(vec.size());
  for (size_t i = 0; i < vec.size(); i++) {
    result[i] = hnswlib::half_to_float(vec[i]);
  }
  return result;
}

std::vector<float> bf16_to_floats(const std::vector<uint16_t> &vec) {
  std::vector<float> result(vec.size());
  for (size_t i = 0; i < vec.size(); i++) {
    result[i] = hnswlib::bfloat16_to_float(vec[i]);
  }
  return result;
}

// functor to filter results with a bitset of IDs
class FilterIdsInSet : public hnswlib::BaseFilterFunctor {
public:
  const DynamicBitset &ids;
  FilterIdsInSet(const DynamicBitset &ids) : ids(ids) {}
  bool operator()(hnswlib::labeltype label_id) { return ids.test(label_id); }
};

WalHeader makeWalHeader(const nlohmann::json &settings) {
  WalHeader h;
  h.dimension = settings.at("dimension").get<int32_t>();
  h.M = settings.value("M", 16);
  h.efConstruction = settings.value("efConstruction", 512);
  std::string space = settings.value("spaceType", "IP");
  h.spaceType = (space == "L2") ? WalSpaceType::L2 : WalSpaceType::IP;
  std::string vt = settings.value("vectorType", "FLOAT32");
  if (vt == "FLOAT16")
    h.vectorType = WalVectorType::FLOAT16;
  else if (vt == "BFLOAT16")
    h.vectorType = WalVectorType::BFLOAT16;
  else
    h.vectorType = WalVectorType::FLOAT32;
  return h;
}

void remove_index_from_disk(const std::string &indexName) {
  std::filesystem::remove("indices/" + indexName + ".bin");
  std::filesystem::remove("indices/" + indexName + ".json");
  std::filesystem::remove("indices/" + indexName + ".data");
  std::filesystem::remove("indices/" + indexName + ".wal");
  std::filesystem::remove("indices/" + indexName + ".wal.compact");
}

void write_index_to_disk(const std::shared_ptr<IndexContext> &ctx, const std::string &indexName) {
  std::filesystem::create_directories("indices");

  if (!ctx->index) {
    std::cerr << "Error: Index not found: " << indexName << std::endl;
    return;
  }

  try {
    ctx->index->saveIndex("indices/" + indexName + ".bin");
  } catch (const std::exception &e) {
    std::cerr << "Error saving index: " << e.what() << std::endl;
    return;
  }

  std::ofstream settings_file("indices/" + indexName + ".json");
  if (!settings_file) {
    std::cerr << "Error: Unable to open settings file for writing: " << indexName << std::endl;
    return;
  }
  settings_file << ctx->settings.dump();
}

// reads index + settings from disk into a new IndexContext.
// caller must hold exclusive contextMapMutex.
std::shared_ptr<IndexContext> read_index_from_disk(const std::string &indexName) {
  std::string settings_path = "indices/" + indexName + ".json";
  std::ifstream settings_file(settings_path);
  if (!settings_file) {
    throw std::runtime_error("Settings file not found: " + settings_path);
  }
  nlohmann::json indexState;
  settings_file >> indexState;

  int dim = indexState.at("dimension").get<int>();
  std::string space = indexState.value("spaceType", "IP");
  std::string vectorType = indexState.value("vectorType", "FLOAT32");

  hnswlib::SpaceInterface<float> *metricSpace = create_space(space, vectorType, dim);

  std::string index_path = "indices/" + indexName + ".bin";
  auto *index = new hnswlib::HierarchicalNSW<float>(metricSpace, index_path, false, 0, true);

  auto ctx = std::make_shared<IndexContext>();
  ctx->index = index;
  ctx->settings = indexState;
  return ctx;
}

void addPointToIndex(hnswlib::HierarchicalNSW<float> *index, const std::string &vectorType, int id, const std::vector<float> &vec) {
  if (vectorType == "FLOAT16") {
    auto converted = floats_to_f16(vec);
    index->addPoint(converted.data(), id, 0);
  } else if (vectorType == "BFLOAT16") {
    auto converted = floats_to_bf16(vec);
    index->addPoint(converted.data(), id, 0);
  } else {
    index->addPoint(vec.data(), id, 0);
  }
}

void startBackgroundResize(std::shared_ptr<IndexContext> ctx, const std::string &indexName, size_t newMaxElements) {
  ctx->resizeThread = std::thread([ctx, indexName, newMaxElements]() {
    size_t oldMax = ctx->index->max_elements_;
    std::cerr << "[resize] index=" << indexName << " starting resize from " << oldMax << " to " << newMaxElements << std::endl;

    try {
      // exclusive lock: waits for in-flight addPoints/searches to drain
      std::unique_lock<std::shared_mutex> exclusiveLock(ctx->mutex);
      ctx->index->resizeIndex(static_cast<int>(newMaxElements));
      exclusiveLock.unlock();

      // flush buffered writes
      std::lock_guard<std::mutex> bufLock(ctx->bufferMutex);
      size_t bufferedCount = ctx->writeBuffer.size();
      std::cerr << "[resize] index=" << indexName << " resize complete, flushing " << bufferedCount << " buffered writes" << std::endl;

      std::string vectorType = get_vector_type(ctx);
      for (const auto &bw : ctx->writeBuffer) {
        try {
          // resize again if needed during flush
          if (ctx->index->cur_element_count + 1 + DEFAULT_INDEX_RESIZE_HEADROOM > ctx->index->max_elements_) {
            ctx->index->resizeIndex(static_cast<int>(static_cast<float>(ctx->index->max_elements_) * (1.0f + INDEX_GROWTH_FACTOR) + 1));
          }
          addPointToIndex(ctx->index, vectorType, bw.id, bw.vector);
          ctx->dataStore->set(bw.id, bw.metadata);
        } catch (const std::exception &e) {
          std::cerr << "[resize] index=" << indexName << " error flushing id=" << bw.id << ": " << e.what() << std::endl;
        }
      }
      ctx->writeBuffer.clear();
      ctx->resizing.store(false);

      std::cerr << "[resize] index=" << indexName << " flush complete, new max=" << ctx->index->max_elements_
                << ", count=" << ctx->index->cur_element_count << std::endl;
    } catch (const std::exception &e) {
      std::cerr << "[resize] index=" << indexName << " resize FAILED: " << e.what() << std::endl;
      std::lock_guard<std::mutex> bufLock(ctx->bufferMutex);
      ctx->writeBuffer.clear();
      ctx->resizing.store(false);
    }
  });
}

int main() {
  const char *fsyncEnv = std::getenv("WAL_FSYNC_INTERVAL_MS");
  if (fsyncEnv) {
    walFsyncIntervalMs = std::atoi(fsyncEnv);
    if (walFsyncIntervalMs <= 0)
      walFsyncIntervalMs = 1000;
  }

  crow::SimpleApp app;
  app.loglevel(crow::LogLevel::Warning);

  CROW_ROUTE(app, "/health").methods(crow::HTTPMethod::GET)([]() { return "OK"; });

  CROW_ROUTE(app, "/create_index").methods(crow::HTTPMethod::POST)([](const crow::request &req) {
    auto data = nlohmann::json::parse(req.body);
    IndexRequest indexRequest = data.get<IndexRequest>();

    {
      std::unique_lock<std::shared_mutex> mapLock(contextMapMutex);

      if (contexts.find(indexRequest.indexName) != contexts.end()) {
        return crow::response(400, "Index already exists");
      }

      hnswlib::SpaceInterface<float> *space = create_space(indexRequest.spaceType, indexRequest.vectorType, indexRequest.dimension);

      auto ctx = std::make_shared<IndexContext>();
      ctx->index = new hnswlib::HierarchicalNSW<float>(space, DEFAULT_INDEX_SIZE, indexRequest.M, indexRequest.efConstruction, 42, true);

      nlohmann::json settings;
      settings["indexName"] = indexRequest.indexName;
      settings["dimension"] = indexRequest.dimension;
      settings["indexType"] = indexRequest.indexType;
      settings["spaceType"] = indexRequest.spaceType;
      settings["vectorType"] = indexRequest.vectorType;
      settings["efConstruction"] = indexRequest.efConstruction;
      settings["M"] = indexRequest.M;
      ctx->settings = settings;

      ctx->dataStore = new DataStore();
      ctx->inFlightGuard = new InFlightGuard();

      WalHeader wh = makeWalHeader(settings);
      std::string walPath = "indices/" + indexRequest.indexName + ".wal";
      std::filesystem::remove(walPath);
      std::filesystem::remove(walPath + ".compact");
      auto *wal = new WriteAheadLog(walPath, wh);
      wal->startFsyncThread(walFsyncIntervalMs);
      ctx->wal = wal;

      contexts[indexRequest.indexName] = ctx;
    }
    return crow::response(201, "Index created");
  });

  CROW_ROUTE(app, "/load_index").methods(crow::HTTPMethod::POST)([](const crow::request &req) {
    auto data = nlohmann::json::parse(req.body);
    std::string indexName = data["indexName"];
    {
      std::unique_lock<std::shared_mutex> mapLock(contextMapMutex);

      if (contexts.find(indexName) != contexts.end()) {
        return crow::response(400, "Index already exists");
      }

      bool hasSnapshot =
          std::filesystem::exists("indices/" + indexName + ".json") && std::filesystem::exists("indices/" + indexName + ".bin");
      std::string walPath = "indices/" + indexName + ".wal";
      bool hasWal = std::filesystem::exists(walPath);

      if (!hasSnapshot && !hasWal) {
        return crow::response(404, std::string("Index not found on disk"));
      }

      std::shared_ptr<IndexContext> ctx;

      if (hasSnapshot) {
        try {
          ctx = read_index_from_disk(indexName);
        } catch (const std::exception &e) {
          return crow::response(500, std::string("Failed to load index: ") + e.what());
        }

        ctx->dataStore = new DataStore();
        std::string dataPath = "indices/" + indexName + ".data";
        if (std::filesystem::exists(dataPath)) {
          try {
            ctx->dataStore->deserialize(dataPath);
          } catch (const std::exception &e) {
            std::cerr << "Warning: Failed to load data store: " << e.what() << std::endl;
          }
        }
      }

      if (hasWal) {
        try {
          auto [walHeader, entries] = WriteAheadLog::readAll(walPath);

          if (!hasSnapshot) {
            std::string spaceStr = (walHeader.spaceType == WalSpaceType::L2) ? "L2" : "IP";
            std::string vtStr = "FLOAT32";
            if (walHeader.vectorType == WalVectorType::FLOAT16)
              vtStr = "FLOAT16";
            else if (walHeader.vectorType == WalVectorType::BFLOAT16)
              vtStr = "BFLOAT16";

            hnswlib::SpaceInterface<float> *space = create_space(spaceStr, vtStr, walHeader.dimension);
            auto *index = new hnswlib::HierarchicalNSW<float>(space, DEFAULT_INDEX_SIZE, walHeader.M, walHeader.efConstruction, 42, true);

            ctx = std::make_shared<IndexContext>();
            ctx->index = index;

            nlohmann::json settings;
            settings["indexName"] = indexName;
            settings["dimension"] = walHeader.dimension;
            settings["spaceType"] = spaceStr;
            settings["vectorType"] = vtStr;
            settings["efConstruction"] = walHeader.efConstruction;
            settings["M"] = walHeader.M;
            ctx->settings = settings;
            ctx->dataStore = new DataStore();
          }

          std::string vectorType = get_vector_type(ctx);

          std::cerr << "[wal] index=" << indexName << " replaying " << entries.size() << " WAL entries" << std::endl;

          // Resolve final state per docId (last-writer-wins)
          struct ResolvedEntry {
            WalOpType finalOp;
            const std::vector<float> *vector;
            const std::map<std::string, FieldValue> *metadata;
          };
          std::unordered_map<uint32_t, ResolvedEntry> resolved;
          resolved.reserve(entries.size());
          for (const auto &entry : entries) {
            if (entry.opType == WalOpType::ADD) {
              resolved[entry.docId] = {WalOpType::ADD, &entry.vector, &entry.metadata};
            } else if (entry.opType == WalOpType::DELETE) {
              resolved[entry.docId] = {WalOpType::DELETE, nullptr, nullptr};
            }
          }

          // Separate into adds and deletes
          std::vector<std::pair<uint32_t, ResolvedEntry *>> adds;
          std::vector<uint32_t> deletes;
          adds.reserve(resolved.size());
          for (auto &[docId, re] : resolved) {
            if (re.finalOp == WalOpType::ADD) {
              adds.push_back({docId, &re});
            } else {
              deletes.push_back(docId);
            }
          }

          // Pre-resize once to fit all adds
          size_t needed = ctx->index->cur_element_count + adds.size() + DEFAULT_INDEX_RESIZE_HEADROOM;
          if (needed > ctx->index->max_elements_) {
            size_t newMax = static_cast<size_t>(static_cast<float>(needed) * (1.0f + INDEX_GROWTH_FACTOR) + 1);
            ctx->index->resizeIndex(static_cast<int>(newMax));
          }

          // Parallel addPoint
          unsigned numThreads = std::thread::hardware_concurrency();
          if (numThreads == 0)
            numThreads = 4;
          if (numThreads > adds.size())
            numThreads = static_cast<unsigned>(adds.size());

          std::atomic<size_t> replayedCount{0};
          size_t totalAdds = adds.size();

          // progress reporter thread (logs every 10% or every 2s, whichever comes first)
          std::atomic<bool> replayDone{false};
          std::thread progressThread;
          if (totalAdds >= 1000) {
            progressThread = std::thread([&]() {
              size_t lastReported = 0;
              while (!replayDone.load(std::memory_order_relaxed)) {
                std::this_thread::sleep_for(std::chrono::seconds(2));
                size_t current = replayedCount.load(std::memory_order_relaxed);
                if (current > lastReported) {
                  int pct = static_cast<int>(current * 100 / totalAdds);
                  std::cerr << "[wal] index=" << indexName << " replay progress: " << current << "/" << totalAdds << " (" << pct << "%)"
                            << std::endl;
                  lastReported = current;
                }
              }
            });
          }

          if (numThreads <= 1 || adds.size() < 100) {
            for (const auto &[docId, re] : adds) {
              addPointToIndex(ctx->index, vectorType, docId, *re->vector);
              ctx->dataStore->set(docId, *re->metadata);
              replayedCount.fetch_add(1, std::memory_order_relaxed);
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
              threads.emplace_back([&ctx, &adds, &vectorType, &replayedCount, start, end]() {
                for (size_t i = start; i < end; i++) {
                  auto &[docId, re] = adds[i];
                  addPointToIndex(ctx->index, vectorType, docId, *re->vector);
                  ctx->dataStore->set(docId, *re->metadata);
                  replayedCount.fetch_add(1, std::memory_order_relaxed);
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

          // Deletes are fast (just flag marking), do sequentially
          for (uint32_t docId : deletes) {
            try {
              ctx->index->markDelete(docId);
            } catch (...) {
            }
            ctx->dataStore->remove(docId);
          }

          std::cerr << "[wal] index=" << indexName << " replay complete, " << adds.size() << " adds, " << deletes.size()
                    << " deletes (resolved from " << entries.size() << " entries, " << numThreads << " threads)" << std::endl;
        } catch (const std::exception &e) {
          std::cerr << "Warning: WAL replay error: " << e.what() << std::endl;
        }

        WalHeader wh = makeWalHeader(ctx->settings);
        auto *wal = new WriteAheadLog(walPath, wh);
        wal->startFsyncThread(walFsyncIntervalMs);
        ctx->wal = wal;
      }

      if (!ctx->inFlightGuard) {
        ctx->inFlightGuard = new InFlightGuard();
      }

      contexts[indexName] = ctx;
    }

    return crow::response(200, "Index loaded");
  });

  CROW_ROUTE(app, "/save_index").methods(crow::HTTPMethod::POST)([](const crow::request &req) {
    auto data = nlohmann::json::parse(req.body);
    std::string indexName = data["indexName"];

    auto ctx = getContext(indexName);
    if (!ctx) {
      return crow::response(404, "Index not found");
    }

    {
      // exclusive lock waits for any in-flight ops and background resize to finish
      std::unique_lock<std::shared_mutex> lock(ctx->mutex);

      try {
        write_index_to_disk(ctx, indexName);
        ctx->dataStore->serialize("indices/" + indexName + ".data");
        if (ctx->wal) {
          ctx->wal->truncate();
        }
      } catch (const std::exception &e) {
        return crow::response(500, std::string("Failed to save index: ") + e.what());
      }
    }
    return crow::response(200, "Index saved");
  });

  CROW_ROUTE(app, "/delete_index").methods(crow::HTTPMethod::DELETE)([](const crow::request &req) {
    auto data = nlohmann::json::parse(req.body);
    std::string indexName = data["indexName"];

    {
      std::unique_lock<std::shared_mutex> mapLock(contextMapMutex);

      auto it = contexts.find(indexName);
      if (it == contexts.end()) {
        return crow::response(404, "Index not found");
      }

      auto ctx = it->second;
      contexts.erase(it);

      std::filesystem::remove("indices/" + indexName + ".wal");
      std::filesystem::remove("indices/" + indexName + ".wal.compact");
    }
    return crow::response(200, "Index deleted");
  });

  CROW_ROUTE(app, "/delete_index_from_disk").methods(crow::HTTPMethod::DELETE)([](const crow::request &req) {
    auto data = nlohmann::json::parse(req.body);
    std::string indexName = data["indexName"];

    {
      std::unique_lock<std::shared_mutex> mapLock(contextMapMutex);

      if (contexts.find(indexName) != contexts.end()) {
        return crow::response(400, "Index is loaded. Please delete it first");
      }

      remove_index_from_disk(indexName);
    }
    return crow::response(200, "Index deleted from disk");
  });

  CROW_ROUTE(app, "/list_indices").methods(crow::HTTPMethod::GET)([]() {
    nlohmann::json response;
    {
      std::shared_lock<std::shared_mutex> mapLock(contextMapMutex);
      for (auto const &[indexName, _] : contexts) {
        response.push_back(indexName);
      }
    }
    return crow::response(response.dump());
  });

  CROW_ROUTE(app, "/index_status/<string>").methods(crow::HTTPMethod::GET)([](std::string indexName) {
    auto ctx = getContext(indexName);
    if (!ctx) {
      return crow::response(404, "Index not found");
    }

    nlohmann::json resp;
    resp["indexName"] = indexName;
    resp["resizing"] = ctx->resizing.load();
    {
      std::lock_guard<std::mutex> bufLock(ctx->bufferMutex);
      resp["bufferedWrites"] = ctx->writeBuffer.size();
    }
    resp["currentElements"] = (size_t)ctx->index->cur_element_count;
    resp["maxElements"] = (size_t)ctx->index->max_elements_;
    resp["deletedElements"] = (size_t)ctx->index->num_deleted_;
    return crow::response(resp.dump());
  });

  CROW_ROUTE(app, "/add_documents").methods(crow::HTTPMethod::POST)([](const crow::request &req) {
    auto data = nlohmann::json::parse(req.body);
    AddDocumentsRequest addReq = data.get<AddDocumentsRequest>();

    if (addReq.ids.size() != addReq.vectors.size()) {
      return crow::response(400, "Number of IDs does not match number of vectors");
    }

    if (addReq.metadatas.size() > 0 && addReq.metadatas.size() != addReq.ids.size()) {
      return crow::response(400, "Number of metadatas does not match number of IDs");
    }

    auto ctx = getContext(addReq.indexName);
    if (!ctx) {
      return crow::response(404, "Index not found");
    }

    if (ctx->wal) {
      for (size_t i = 0; i < addReq.ids.size(); i++) {
        std::map<std::string, FieldValue> meta;
        if (addReq.metadatas.size()) {
          meta = addReq.metadatas[i];
        }
        ctx->wal->logAdd(static_cast<uint32_t>(addReq.ids[i]), addReq.vectors[i], meta);
      }
    }

    if (ctx->resizing.load()) {
      std::lock_guard<std::mutex> bufLock(ctx->bufferMutex);
      if (ctx->resizing.load()) {
        for (size_t i = 0; i < addReq.ids.size(); i++) {
          BufferedWrite bw;
          bw.id = addReq.ids[i];
          bw.vector = addReq.vectors[i];
          if (addReq.metadatas.size()) {
            bw.metadata = addReq.metadatas[i];
          }
          ctx->writeBuffer.push_back(std::move(bw));
        }
        return crow::response(201, "Documents added");
      }
    }

    if (ctx->index->cur_element_count + addReq.ids.size() + DEFAULT_INDEX_RESIZE_HEADROOM > ctx->index->max_elements_) {
      // try to become the resize initiator
      bool expected = false;
      if (ctx->resizing.compare_exchange_strong(expected, true)) {
        if (ctx->resizeThread.joinable()) {
          ctx->resizeThread.join();
        }

        size_t newMax =
            static_cast<size_t>(static_cast<float>(ctx->index->max_elements_) * (1.0f + INDEX_GROWTH_FACTOR) + addReq.ids.size());

        // buffer writes
        {
          std::lock_guard<std::mutex> bufLock(ctx->bufferMutex);
          for (size_t i = 0; i < addReq.ids.size(); i++) {
            BufferedWrite bw;
            bw.id = addReq.ids[i];
            bw.vector = addReq.vectors[i];
            if (addReq.metadatas.size()) {
              bw.metadata = addReq.metadatas[i];
            }
            ctx->writeBuffer.push_back(std::move(bw));
          }
        }

        startBackgroundResize(ctx, addReq.indexName, newMax);
        return crow::response(201, "Documents added");
      } else {
        // lost the CAS, someone else is resizing, buffer writes
        std::lock_guard<std::mutex> bufLock(ctx->bufferMutex);
        if (ctx->resizing.load()) {
          for (size_t i = 0; i < addReq.ids.size(); i++) {
            BufferedWrite bw;
            bw.id = addReq.ids[i];
            bw.vector = addReq.vectors[i];
            if (addReq.metadatas.size()) {
              bw.metadata = addReq.metadatas[i];
            }
            ctx->writeBuffer.push_back(std::move(bw));
          }
          return crow::response(201, "Documents added");
        }
      }
    }

    // normal write path
    {
      std::shared_lock<std::shared_mutex> lock(ctx->mutex);
      std::string vectorType = get_vector_type(ctx);
      auto *guard = ctx->inFlightGuard;
      for (size_t i = 0; i < addReq.ids.size(); i++) {
        guard->acquire(addReq.ids[i]);
        addPointToIndex(ctx->index, vectorType, addReq.ids[i], addReq.vectors[i]);
        std::map<std::string, FieldValue> meta;
        if (addReq.metadatas.size()) {
          meta = addReq.metadatas[i];
        }
        ctx->dataStore->set(addReq.ids[i], meta);
        guard->release(addReq.ids[i]);
      }

      if (ctx->wal && ctx->wal->hasDeletes() && ctx->wal->approxSize() > WAL_COMPACT_THRESHOLD) {
        ctx->wal->tryCompact();
      }
    }

    return crow::response(201, "Documents added");
  });

  CROW_ROUTE(app, "/delete_documents").methods(crow::HTTPMethod::DELETE)([](const crow::request &req) {
    auto data = nlohmann::json::parse(req.body);
    DeleteDocumentsRequest deleteReq = data.get<DeleteDocumentsRequest>();

    auto ctx = getContext(deleteReq.indexName);
    if (!ctx) {
      return crow::response(404, "Index not found");
    }

    {
      std::shared_lock<std::shared_mutex> lock(ctx->mutex);
      for (int id : deleteReq.ids) {
        ctx->index->markDelete(id);
        ctx->dataStore->remove(id);
        if (ctx->wal) {
          ctx->wal->logDelete(static_cast<uint32_t>(id));
        }
      }
    }

    return crow::response(200, "Documents deleted");
  });

  CROW_ROUTE(app, "/get_document/<string>/<int>")
      .methods(crow::HTTPMethod::GET)([](const crow::request &req, std::string indexName, int id) {
        auto ctx = getContext(indexName);
        if (!ctx) {
          return crow::response(404, "Index not found");
        }

        std::shared_lock<std::shared_mutex> lock(ctx->mutex);

        if (!ctx->dataStore->contains(id)) {
          return crow::response(404, "Document not found");
        }

        auto metadata = ctx->dataStore->get(id);
        std::string vectorType = get_vector_type(ctx);
        std::vector<float> vectorData;
        if (vectorType == "FLOAT16") {
          auto rawData = ctx->index->getDataByLabel<uint16_t>(id);
          vectorData = f16_to_floats(rawData);
        } else if (vectorType == "BFLOAT16") {
          auto rawData = ctx->index->getDataByLabel<uint16_t>(id);
          vectorData = bf16_to_floats(rawData);
        } else {
          vectorData = ctx->index->getDataByLabel<float>(id);
        }
        nlohmann::json response;

        response["id"] = id;
        response["vector"] = vectorData;
        response["metadata"] = nlohmann::json();
        for (const auto &[key, value] : metadata) {
          std::visit([&response, &key](auto &&arg) { response["metadata"][key] = arg; }, value);
        }

        return crow::response(response.dump());
      });

  CROW_ROUTE(app, "/search").methods(crow::HTTPMethod::POST)([](const crow::request &req) {
    auto data = nlohmann::json::parse(req.body);
    SearchRequest searchReq = data.get<SearchRequest>();

    auto ctx = getContext(searchReq.indexName);
    if (!ctx) {
      return crow::response(404, "Index not found");
    }

    std::shared_lock<std::shared_mutex> lock(ctx->mutex);

    ctx->index->setEf(searchReq.efSearch);

    std::string vectorType = get_vector_type(ctx);
    std::vector<uint16_t> queryConverted;
    const void *queryData;
    if (vectorType == "FLOAT16") {
      queryConverted = floats_to_f16(searchReq.queryVector);
      queryData = queryConverted.data();
    } else if (vectorType == "BFLOAT16") {
      queryConverted = floats_to_bf16(searchReq.queryVector);
      queryData = queryConverted.data();
    } else {
      queryData = searchReq.queryVector.data();
    }

    std::priority_queue<std::pair<float, hnswlib::labeltype>> result;

    if (searchReq.filter.size() > 0) {
      std::shared_ptr<FilterASTNode> filters = parseFilters(searchReq.filter);
      DynamicBitset filteredIds = ctx->dataStore->filter(filters);

      FilterIdsInSet filter(filteredIds);

      if (filteredIds.count() < ctx->index->cur_element_count * EXACT_KNN_FILTER_PCT_MATCH_THRESHOLD) {
        result = ctx->index->searchExactKnn(queryData, searchReq.k, &filter);
      } else {
        result = ctx->index->searchKnn(queryData, searchReq.k, &filter);
      }
    } else {
      result = ctx->index->searchKnn(queryData, searchReq.k);
    }

    nlohmann::json response;
    std::vector<int> ids;
    std::vector<float> distances;
    while (!result.empty()) {
      ids.push_back(result.top().second);
      distances.push_back(result.top().first);
      result.pop();
    }

    std::reverse(ids.begin(), ids.end());
    std::reverse(distances.begin(), distances.end());

    response["hits"] = ids;
    response["distances"] = distances;

    if (searchReq.returnMetadata) {
      auto metadatas = ctx->dataStore->getMany(ids);
      response["metadatas"] = nlohmann::json::array();
      for (const auto &metadata : metadatas) {
        nlohmann::json json_metadata;
        for (const auto &[key, value] : metadata) {
          std::visit([&json_metadata, &key](auto &&arg) { json_metadata[key] = arg; }, value);
        }
        response["metadatas"].push_back(json_metadata);
      }
    }

    return crow::response(response.dump());
  });

  std::cout << "Server started on port 8685!" << std::endl;
  std::cout << "Press Ctrl+C to quit" << std::endl;
  std::cout << "All other stdout is suppressed as an optimisation" << std::endl;

  app.port(8685).multithreaded().run();

  // clean up
  {
    std::unique_lock<std::shared_mutex> mapLock(contextMapMutex);
    contexts.clear();
  }
}
