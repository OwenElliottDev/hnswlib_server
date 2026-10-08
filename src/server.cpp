#include "crow.h"
#include "data_store.hpp"
#include "filters.hpp"
#include "index_context.hpp"
#include "index_utils.hpp"
#include "models.hpp"
#include "nlohmann/json.hpp"
#include "wal.hpp"
#include "wal_replay.hpp"
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

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

std::string get_vector_type(const std::shared_ptr<IndexContext> &ctx) {
  if (ctx->settings.contains("vectorType")) {
    return ctx->settings["vectorType"].get<std::string>();
  }
  return "FLOAT32";
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
  int mrlScanDim = indexState.value("mrlScanDim", 0);

  BuiltSpace bs = build_space(space, vectorType, dim, mrlScanDim);

  std::string index_path = "indices/" + indexName + ".bin";
  auto *index = new hnswlib::HierarchicalNSW<float>(bs.space, index_path, false, 0, true);

  auto ctx = std::make_shared<IndexContext>();
  ctx->index = index;
  ctx->settings = indexState;
  ctx->isMrl = bs.isMrl;
  ctx->mrlScanDim = mrlScanDim;
  ctx->mrlFullDistFunc = bs.fullDistFunc;
  ctx->mrlFullDistFuncParam = bs.fullDistFuncParam;
  return ctx;
}

void startBackgroundResize(std::shared_ptr<IndexContext> ctx, const std::string &indexName, size_t newMaxElements) {
  ctx->resizeThread = std::thread([ctx, indexName, newMaxElements]() {
    size_t oldMax = ctx->index->max_elements_;
    LOG("resize") << "index=" << indexName << " starting resize from " << oldMax << " to " << newMaxElements << std::endl;

    try {
      // exclusive lock: waits for in-flight addPoints/searches to drain
      std::unique_lock<std::shared_mutex> exclusiveLock(ctx->mutex);
      ctx->index->resizeIndex(static_cast<int>(newMaxElements));
      exclusiveLock.unlock();

      // flush buffered writes
      std::lock_guard<std::mutex> bufLock(ctx->bufferMutex);
      size_t bufferedCount = ctx->writeBuffer.size();
      LOG("resize") << "index=" << indexName << " resize complete, flushing " << bufferedCount << " buffered writes" << std::endl;

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
          LOG("resize") << "index=" << indexName << " error flushing id=" << bw.id << ": " << e.what() << std::endl;
        }
      }
      ctx->writeBuffer.clear();
      ctx->resizing.store(false);

      LOG("resize") << "index=" << indexName << " flush complete, new max=" << ctx->index->max_elements_
                    << ", count=" << ctx->index->cur_element_count << std::endl;
    } catch (const std::exception &e) {
      LOG("resize") << "index=" << indexName << " resize FAILED: " << e.what() << std::endl;
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

  CROW_ROUTE(app, "/version").methods(crow::HTTPMethod::GET)([]() {
    nlohmann::json response;
    response["version"] = HNSWLIB_VERSION;
    response["tagline"] = "HNSWLib Server: https://github.com/OwenElliottDev/hnswlib_server";
    return crow::response(response.dump());
  });

  CROW_ROUTE(app, "/create_index").methods(crow::HTTPMethod::POST)([](const crow::request &req) {
    auto data = nlohmann::json::parse(req.body);
    IndexRequest indexRequest = data.get<IndexRequest>();

    {
      std::unique_lock<std::shared_mutex> mapLock(contextMapMutex);

      if (contexts.find(indexRequest.indexName) != contexts.end()) {
        return crow::response(400, "Index already exists");
      }

      BuiltSpace bs;
      try {
        bs = build_space(indexRequest.spaceType, indexRequest.vectorType, indexRequest.dimension, indexRequest.mrlScanDim);
      } catch (const std::exception &e) {
        return crow::response(400, std::string("Invalid index configuration: ") + e.what());
      }

      auto ctx = std::make_shared<IndexContext>();
      ctx->index = new hnswlib::HierarchicalNSW<float>(bs.space, DEFAULT_INDEX_SIZE, indexRequest.M, indexRequest.efConstruction, 42, true);
      ctx->isMrl = bs.isMrl;
      ctx->mrlScanDim = indexRequest.mrlScanDim;
      ctx->mrlFullDistFunc = bs.fullDistFunc;
      ctx->mrlFullDistFuncParam = bs.fullDistFuncParam;

      nlohmann::json settings;
      settings["indexName"] = indexRequest.indexName;
      settings["dimension"] = indexRequest.dimension;
      settings["indexType"] = indexRequest.indexType;
      settings["spaceType"] = indexRequest.spaceType;
      settings["vectorType"] = indexRequest.vectorType;
      settings["efConstruction"] = indexRequest.efConstruction;
      settings["M"] = indexRequest.M;
      settings["mrlScanDim"] = indexRequest.mrlScanDim;
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

    std::shared_ptr<IndexContext> ctx;
    ResolvedWal resolved;
    std::string walPath = "indices/" + indexName + ".wal";
    bool hasWalEntries = false;

    {
      std::unique_lock<std::shared_mutex> mapLock(contextMapMutex);

      if (contexts.find(indexName) != contexts.end()) {
        return crow::response(400, "Index already exists");
      }

      bool hasSnapshot =
          std::filesystem::exists("indices/" + indexName + ".json") && std::filesystem::exists("indices/" + indexName + ".bin");
      bool hasWal = std::filesystem::exists(walPath);

      if (!hasSnapshot && !hasWal) {
        return crow::response(404, std::string("Index not found on disk"));
      }

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
            LOG("ERROR") << "Failed to load data store: " << e.what() << std::endl;
          }
        }
      }

      if (hasWal) {
        try {
          auto [walHeader, entries] = WriteAheadLog::readAll(walPath);

          if (!hasSnapshot) {
            ctx = contextFromWalHeader(walHeader, indexName);
          }

          LOG("wal") << "index=" << indexName << " read " << entries.size() << " WAL entries" << std::endl;

          resolved = resolveWalEntries(entries);
          hasWalEntries = !resolved.empty();

        } catch (const std::exception &e) {
          LOG("ERROR") << "WAL read error: " << e.what() << std::endl;
          if (!ctx) {
            return crow::response(500, std::string("Failed to read WAL: ") + e.what());
          }
        }

        // if no WAL entries to replay, create fresh WAL immediately
        if (!hasWalEntries) {
          WalHeader wh = makeWalHeader(ctx->settings);
          auto *wal = new WriteAheadLog(walPath, wh);
          wal->startFsyncThread(walFsyncIntervalMs);
          ctx->wal = wal;
        }
      }

      ctx->inFlightGuard = new InFlightGuard();

      if (hasWalEntries) {
        ctx->replayingWal.store(true, std::memory_order_release);
      }

      contexts[indexName] = ctx;
    } // lock released

    if (hasWalEntries) {
      startBackgroundWalReplay(ctx.get(), indexName, std::move(resolved), walPath, walFsyncIntervalMs);
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

    if (ctx->replayingWal.load()) {
      return crow::response(409, "Index is replaying WAL, try again later");
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
      // ctx destructor cancels+joins walReplayThread, then cleans up resources
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

    bool replaying = ctx->replayingWal.load();
    resp["replayingWal"] = replaying;
    if (replaying) {
      nlohmann::json progress;
      size_t replayed = ctx->walReplayedCount.load(std::memory_order_relaxed);
      size_t total = ctx->walReplayTotalAdds.load(std::memory_order_relaxed);
      progress["replayedAdds"] = replayed;
      progress["totalAdds"] = total;
      progress["percentComplete"] = (total > 0) ? static_cast<int>(replayed * 100 / total) : 100;
      resp["walReplayProgress"] = progress;
    }

    {
      std::lock_guard<std::mutex> bufLock(ctx->bufferMutex);
      resp["bufferedWrites"] = ctx->writeBuffer.size();
      if (!ctx->walReplayError.empty()) {
        resp["walReplayError"] = ctx->walReplayError;
      }
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

    if (ctx->replayingWal.load()) {
      return crow::response(409, "Index is replaying WAL, try again later");
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

  CROW_ROUTE(app, "/update_documents").methods(crow::HTTPMethod::PATCH)([](const crow::request &req) {
    UpdateDocumentsRequest updReq;
    try {
      updReq = nlohmann::json::parse(req.body).get<UpdateDocumentsRequest>();
    } catch (const std::exception &e) {
      return crow::response(400, std::string("Invalid request: ") + e.what());
    }

    if (updReq.metadatas.size() != updReq.ids.size()) {
      return crow::response(400, "Number of metadatas does not match number of IDs");
    }

    auto ctx = getContext(updReq.indexName);
    if (!ctx) {
      return crow::response(404, "Index not found");
    }

    if (ctx->replayingWal.load()) {
      return crow::response(409, "Index is replaying WAL, try again later");
    }

    // Metadata-only, shared lock prevents deletes racing the check.
    // Documents in the buffer are not in the index yet and will report 404.
    nlohmann::json results = nlohmann::json::array();
    bool anyErrors = false;
    {
      std::shared_lock<std::shared_mutex> lock(ctx->mutex);
      for (size_t i = 0; i < updReq.ids.size(); i++) {
        int id = updReq.ids[i];
        InFlightLock idLock(*ctx->inFlightGuard, id);
        if (!ctx->dataStore->contains(id)) {
          results.push_back({{"id", id}, {"status", 404}, {"error", "Document not found"}});
          anyErrors = true;
          continue;
        }
        auto merged = ctx->dataStore->get(id);
        for (const auto &[key, value] : updReq.metadatas[i]) {
          if (value) {
            merged[key] = *value;
          } else {
            merged.erase(key);
          }
        }
        if (ctx->wal) {
          ctx->wal->logUpdate(static_cast<uint32_t>(id), merged);
        }
        ctx->dataStore->set(id, std::move(merged));
        results.push_back({{"id", id}, {"status", 200}});
      }

      if (ctx->wal && ctx->wal->hasDeletes() && ctx->wal->approxSize() > WAL_COMPACT_THRESHOLD) {
        ctx->wal->tryCompact();
      }
    }

    nlohmann::json response;
    response["errors"] = anyErrors;
    response["results"] = std::move(results);
    return crow::response(200, response.dump());
  });

  CROW_ROUTE(app, "/delete_documents").methods(crow::HTTPMethod::DELETE)([](const crow::request &req) {
    auto data = nlohmann::json::parse(req.body);
    DeleteDocumentsRequest deleteReq = data.get<DeleteDocumentsRequest>();

    auto ctx = getContext(deleteReq.indexName);
    if (!ctx) {
      return crow::response(404, "Index not found");
    }

    if (ctx->replayingWal.load()) {
      return crow::response(409, "Index is replaying WAL, try again later");
    }

    {
      // exclusive lock: removePoint repairs the graph and is not thread-safe with
      // concurrent adds/searches/resize.
      std::unique_lock<std::shared_mutex> lock(ctx->mutex);
      for (int id : deleteReq.ids) {
        try {
          ctx->index->removePoint(id);
        } catch (const std::exception &) {
          // id not present in the index; treat as a no-op on the graph
        }
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
        std::vector<float> vectorData = getVectorFromIndex(ctx->index, get_vector_type(ctx), id);
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

    if (searchReq.offset < 0) {
      return crow::response(400, "offset must be non-negative");
    }

    searchReq.efSearch = std::max(searchReq.efSearch, searchReq.k + searchReq.offset);

    std::shared_lock<std::shared_mutex> lock(ctx->mutex);

    ctx->index->setEf(searchReq.efSearch);

    std::vector<uint16_t> queryScratch;
    const void *queryData =
        to_storage_vector(get_vector_type(ctx), searchReq.queryVector.data(), searchReq.queryVector.size(), queryScratch);

    MrlParams mrl{ctx->isMrl, ctx->mrlFullDistFunc, ctx->mrlFullDistFuncParam};
    size_t fetchK = static_cast<size_t>(searchReq.k) + searchReq.offset;
    std::vector<int> ids;
    std::vector<float> distances;
    if (searchReq.filter.size() > 0) {
      std::shared_ptr<FilterASTNode> filters = parseFilters(searchReq.filter);
      DynamicBitset filteredIds = ctx->dataStore->filter(filters);
      std::tie(ids, distances) = knn_search(ctx->index, mrl, queryData, fetchK, searchReq.rerankSize, &filteredIds);
    } else {
      std::tie(ids, distances) = knn_search(ctx->index, mrl, queryData, fetchK, searchReq.rerankSize, nullptr);
    }
    paginate_results(ids, distances, searchReq.offset, searchReq.k);

    nlohmann::json response;
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

  CROW_ROUTE(app, "/similar").methods(crow::HTTPMethod::POST)([](const crow::request &req) {
    auto data = nlohmann::json::parse(req.body);
    SimilarRequest similarReq = data.get<SimilarRequest>();

    auto ctx = getContext(similarReq.indexName);
    if (!ctx) {
      return crow::response(404, "Index not found");
    }

    if (similarReq.offset < 0) {
      return crow::response(400, "offset must be non-negative");
    }

    similarReq.efSearch = std::max(similarReq.efSearch, similarReq.k + similarReq.offset);

    std::shared_lock<std::shared_mutex> lock(ctx->mutex);

    if (!ctx->dataStore->contains(similarReq.docId)) {
      return crow::response(404, "Input document not found in the index");
    }

    ctx->index->setEf(similarReq.efSearch);

    std::vector<float> queryVec = getVectorFromIndex(ctx->index, get_vector_type(ctx), similarReq.docId);
    std::vector<uint16_t> queryScratch;
    const void *queryData = to_storage_vector(get_vector_type(ctx), queryVec.data(), queryVec.size(), queryScratch);

    MrlParams mrl{ctx->isMrl, ctx->mrlFullDistFunc, ctx->mrlFullDistFuncParam};
    size_t fetchK = static_cast<size_t>(similarReq.k) + similarReq.offset + (similarReq.excludeInputDocument ? 1 : 0);
    std::vector<int> ids;
    std::vector<float> distances;
    if (similarReq.filter.size() > 0) {
      std::shared_ptr<FilterASTNode> filters = parseFilters(similarReq.filter);
      DynamicBitset filteredIds = ctx->dataStore->filter(filters);
      std::tie(ids, distances) = knn_search(ctx->index, mrl, queryData, fetchK, similarReq.rerankSize, &filteredIds);
    } else {
      std::tie(ids, distances) = knn_search(ctx->index, mrl, queryData, fetchK, similarReq.rerankSize, nullptr);
    }
    paginate_results(ids, distances, similarReq.offset, similarReq.k, similarReq.excludeInputDocument ? similarReq.docId : -1);

    nlohmann::json response;
    response["hits"] = ids;
    response["distances"] = distances;

    if (similarReq.returnMetadata) {
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

  std::cout << "Welcome to HNSWLib server." << std::endl;
  std::cout << "Server started on port 8685." << std::endl;
  std::cout << "Press Ctrl+C to quit" << std::endl;

  app.port(8685).multithreaded().run();

  // clean up
  {
    std::unique_lock<std::shared_mutex> mapLock(contextMapMutex);
    contexts.clear();
  }
}
