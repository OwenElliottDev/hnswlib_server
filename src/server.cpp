#include "crow.h"
#include "data_store.hpp"
#include "filters.hpp"
#include "hnswlib/hnswlib.h"
#include "models.hpp"
#include "nlohmann/json.hpp"
#include "wal.hpp"
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define DEFAULT_INDEX_SIZE 100000
#define DEFAULT_INDEX_RESIZE_HEADROOM 10000
#define INDEX_GROWTH_FACTOR 2.0
#define EXACT_KNN_FILTER_PCT_MATCH_THRESHOLD 0.1

std::unordered_map<std::string, hnswlib::HierarchicalNSW<float> *> indices;
std::unordered_map<std::string, nlohmann::json> indexSettings;
std::unordered_map<std::string, DataStore *> dataStores;
std::unordered_map<std::string, WriteAheadLog *> walLogs;

std::shared_mutex indexMutex;
std::mutex dataStoreMutex;

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

std::unordered_map<std::string, InFlightGuard *> inFlightGuards;

int walFsyncIntervalMs = 1000;

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

std::string get_vector_type(const std::string &indexName) {
  auto it = indexSettings.find(indexName);
  if (it != indexSettings.end() && it->second.contains("vectorType")) {
    return it->second["vectorType"].get<std::string>();
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

void write_index_to_disk(const std::string &indexName) {
  std::filesystem::create_directories("indices");

  auto *index = indices[indexName];
  if (!index) {
    std::cerr << "Error: Index not found: " << indexName << std::endl;
    return;
  }

  try {
    index->saveIndex("indices/" + indexName + ".bin");
  } catch (const std::exception &e) {
    std::cerr << "Error saving index: " << e.what() << std::endl;
    return;
  }

  std::ofstream settings_file("indices/" + indexName + ".json");
  if (!settings_file) {
    std::cerr << "Error: Unable to open settings file for writing: " << indexName << std::endl;
    return;
  }
  settings_file << indexSettings[indexName].dump();
}

void read_index_from_disk(const std::string &indexName) {
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

  indices[indexName] = index;
  indexSettings[indexName] = indexState;
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
      std::unique_lock<std::shared_mutex> indexLock(indexMutex);
      std::lock_guard<std::mutex> datastoreLock(dataStoreMutex);

      if (indices.find(indexRequest.indexName) != indices.end()) {
        return crow::response(400, "Index already exists");
      }

      hnswlib::SpaceInterface<float> *space = create_space(indexRequest.spaceType, indexRequest.vectorType, indexRequest.dimension);

      auto *index = new hnswlib::HierarchicalNSW<float>(space, DEFAULT_INDEX_SIZE, indexRequest.M, indexRequest.efConstruction, 42, true);

      indices[indexRequest.indexName] = index;
      nlohmann::json settings;
      settings["indexName"] = indexRequest.indexName;
      settings["dimension"] = indexRequest.dimension;
      settings["indexType"] = indexRequest.indexType;
      settings["spaceType"] = indexRequest.spaceType;
      settings["vectorType"] = indexRequest.vectorType;
      settings["efConstruction"] = indexRequest.efConstruction;
      settings["M"] = indexRequest.M;
      indexSettings[indexRequest.indexName] = settings;
      dataStores[indexRequest.indexName] = new DataStore();

      inFlightGuards[indexRequest.indexName] = new InFlightGuard();

      WalHeader wh = makeWalHeader(settings);
      std::string walPath = "indices/" + indexRequest.indexName + ".wal";
      std::filesystem::remove(walPath);
      std::filesystem::remove(walPath + ".compact");
      auto *wal = new WriteAheadLog(walPath, wh);
      wal->startFsyncThread(walFsyncIntervalMs);
      walLogs[indexRequest.indexName] = wal;
    }
    return crow::response(201, "Index created");
  });

  CROW_ROUTE(app, "/load_index").methods(crow::HTTPMethod::POST)([](const crow::request &req) {
    auto data = nlohmann::json::parse(req.body);
    std::string indexName = data["indexName"];
    {
      std::unique_lock<std::shared_mutex> indexLock(indexMutex);
      std::lock_guard<std::mutex> datastoreLock(dataStoreMutex);

      if (indices.find(indexName) != indices.end()) {
        return crow::response(400, "Index already exists");
      }

      bool hasSnapshot =
          std::filesystem::exists("indices/" + indexName + ".json") && std::filesystem::exists("indices/" + indexName + ".bin");
      std::string walPath = "indices/" + indexName + ".wal";
      bool hasWal = std::filesystem::exists(walPath);

      if (!hasSnapshot && !hasWal) {
        return crow::response(404, std::string("Index not found on disk"));
      }

      if (hasSnapshot) {
        try {
          read_index_from_disk(indexName);
        } catch (const std::exception &e) {
          return crow::response(500, std::string("Failed to load index: ") + e.what());
        }

        dataStores[indexName] = new DataStore();
        std::string dataPath = "indices/" + indexName + ".data";
        if (std::filesystem::exists(dataPath)) {
          try {
            dataStores[indexName]->deserialize(dataPath);
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
            indices[indexName] = index;

            nlohmann::json settings;
            settings["indexName"] = indexName;
            settings["dimension"] = walHeader.dimension;
            settings["spaceType"] = spaceStr;
            settings["vectorType"] = vtStr;
            settings["efConstruction"] = walHeader.efConstruction;
            settings["M"] = walHeader.M;
            indexSettings[indexName] = settings;
            dataStores[indexName] = new DataStore();
          }

          auto *index = indices[indexName];
          std::string vectorType = get_vector_type(indexName);

          for (const auto &entry : entries) {
            if (entry.opType == WalOpType::ADD) {
              if (index->cur_element_count + 1 + DEFAULT_INDEX_RESIZE_HEADROOM > index->max_elements_) {
                index->resizeIndex(static_cast<int>(static_cast<float>(index->max_elements_) * (1.0f + INDEX_GROWTH_FACTOR) + 1));
              }
              if (vectorType == "FLOAT16") {
                auto converted = floats_to_f16(entry.vector);
                index->addPoint(converted.data(), entry.docId, 0);
              } else if (vectorType == "BFLOAT16") {
                auto converted = floats_to_bf16(entry.vector);
                index->addPoint(converted.data(), entry.docId, 0);
              } else {
                index->addPoint(entry.vector.data(), entry.docId, 0);
              }
              dataStores[indexName]->set(entry.docId, entry.metadata);
            } else if (entry.opType == WalOpType::DELETE) {
              try {
                index->markDelete(entry.docId);
              } catch (...) {
                // safe to ignore during replay
              }
              dataStores[indexName]->remove(entry.docId);
            }
          }
        } catch (const std::exception &e) {
          std::cerr << "Warning: WAL replay error: " << e.what() << std::endl;
        }

        WalHeader wh = makeWalHeader(indexSettings[indexName]);
        auto *wal = new WriteAheadLog(walPath, wh);
        wal->startFsyncThread(walFsyncIntervalMs);
        walLogs[indexName] = wal;
      }

      if (!inFlightGuards.count(indexName)) {
        inFlightGuards[indexName] = new InFlightGuard();
      }
    }

    return crow::response(200, "Index loaded");
  });

  CROW_ROUTE(app, "/save_index").methods(crow::HTTPMethod::POST)([](const crow::request &req) {
    auto data = nlohmann::json::parse(req.body);
    std::string indexName = data["indexName"];

    {
      std::unique_lock<std::shared_mutex> indexLock(indexMutex);
      std::lock_guard<std::mutex> datastoreLock(dataStoreMutex);

      if (indices.find(indexName) == indices.end()) {
        return crow::response(404, "Index not found");
      }

      try {
        write_index_to_disk(indexName);
        dataStores[indexName]->serialize("indices/" + indexName + ".data");
        if (walLogs.count(indexName) && walLogs[indexName]) {
          walLogs[indexName]->truncate();
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
      std::unique_lock<std::shared_mutex> indexLock(indexMutex);
      std::lock_guard<std::mutex> datastoreLock(dataStoreMutex);

      if (indices.find(indexName) == indices.end()) {
        return crow::response(404, "Index not found");
      }

      delete indices[indexName];
      indices.erase(indexName);
      indexSettings.erase(indexName);
      delete dataStores[indexName];
      dataStores.erase(indexName);
      if (walLogs.count(indexName)) {
        walLogs[indexName]->stopFsyncThread();
        delete walLogs[indexName];
        walLogs.erase(indexName);
        std::filesystem::remove("indices/" + indexName + ".wal");
        std::filesystem::remove("indices/" + indexName + ".wal.compact");
      }
      if (inFlightGuards.count(indexName)) {
        delete inFlightGuards[indexName];
        inFlightGuards.erase(indexName);
      }
    }
    return crow::response(200, "Index deleted");
  });

  CROW_ROUTE(app, "/delete_index_from_disk").methods(crow::HTTPMethod::DELETE)([](const crow::request &req) {
    auto data = nlohmann::json::parse(req.body);
    std::string indexName = data["indexName"];

    {
      std::unique_lock<std::shared_mutex> indexLock(indexMutex);
      std::lock_guard<std::mutex> datastoreLock(dataStoreMutex);

      if (indices.find(indexName) != indices.end()) {
        return crow::response(400, "Index is loaded. Please delete it first");
      }

      remove_index_from_disk(indexName);
    }
    return crow::response(200, "Index deleted from disk");
  });

  CROW_ROUTE(app, "/list_indices").methods(crow::HTTPMethod::GET)([]() {
    nlohmann::json response;
    for (auto const &[indexName, _] : indices) {
      response.push_back(indexName);
    }

    return crow::response(response.dump());
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

    if (indices.find(addReq.indexName) == indices.end()) {
      return crow::response(404, "Index not found");
    }

    auto *index = indices[addReq.indexName];

    if (index->cur_element_count + addReq.ids.size() + DEFAULT_INDEX_RESIZE_HEADROOM > index->max_elements_) {

      std::unique_lock<std::shared_mutex> uniqueLock(indexMutex);

      if (index->cur_element_count + addReq.ids.size() + DEFAULT_INDEX_RESIZE_HEADROOM > index->max_elements_) {
        index->resizeIndex(
            (int)((float)index->max_elements_ + (float)index->max_elements_ * INDEX_GROWTH_FACTOR + (float)addReq.ids.size()));
      }
    }

    {
      std::shared_lock<std::shared_mutex> lock(indexMutex);
      std::string vectorType = get_vector_type(addReq.indexName);
      auto *guard = inFlightGuards[addReq.indexName];
      for (int i = 0; i < addReq.ids.size(); i++) {
        guard->acquire(addReq.ids[i]);
        if (vectorType == "FLOAT16") {
          auto converted = floats_to_f16(addReq.vectors[i]);
          indices[addReq.indexName]->addPoint(converted.data(), addReq.ids[i], 0);
        } else if (vectorType == "BFLOAT16") {
          auto converted = floats_to_bf16(addReq.vectors[i]);
          indices[addReq.indexName]->addPoint(converted.data(), addReq.ids[i], 0);
        } else {
          indices[addReq.indexName]->addPoint(addReq.vectors[i].data(), addReq.ids[i], 0);
        }
        std::map<std::string, FieldValue> meta;
        if (addReq.metadatas.size()) {
          meta = addReq.metadatas[i];
        }
        dataStores[addReq.indexName]->set(addReq.ids[i], meta);
        if (walLogs.count(addReq.indexName) && walLogs[addReq.indexName]) {
          walLogs[addReq.indexName]->logAdd(static_cast<uint32_t>(addReq.ids[i]), addReq.vectors[i], meta);
        }
        guard->release(addReq.ids[i]);
      }

      if (walLogs.count(addReq.indexName) && walLogs[addReq.indexName]) {
        auto *wal = walLogs[addReq.indexName];
        if (wal->hasDeletes() && wal->approxSize() > WAL_COMPACT_THRESHOLD) {
          wal->tryCompact();
        }
      }
    }

    return crow::response(201, "Documents added");
  });

  CROW_ROUTE(app, "/delete_documents").methods(crow::HTTPMethod::DELETE)([](const crow::request &req) {
    auto data = nlohmann::json::parse(req.body);
    DeleteDocumentsRequest deleteReq = data.get<DeleteDocumentsRequest>();

    if (indices.find(deleteReq.indexName) == indices.end()) {
      return crow::response(404, "Index not found");
    }

    auto *index = indices[deleteReq.indexName];

    for (int id : deleteReq.ids) {
      index->markDelete(id);
      dataStores[deleteReq.indexName]->remove(id);
      if (walLogs.count(deleteReq.indexName) && walLogs[deleteReq.indexName]) {
        walLogs[deleteReq.indexName]->logDelete(static_cast<uint32_t>(id));
      }
    }

    return crow::response(200, "Documents deleted");
  });

  CROW_ROUTE(app, "/get_document/<string>/<int>")
      .methods(crow::HTTPMethod::GET)([](const crow::request &req, std::string indexName, int id) {
        if (indices.find(indexName) == indices.end()) {
          return crow::response(404, "Index not found");
        }

        auto *index = indices[indexName];
        auto hasDoc = dataStores[indexName]->contains(id);

        if (!hasDoc) {
          return crow::response(404, "Document not found");
        }

        auto metadata = dataStores[indexName]->get(id);
        std::string vectorType = get_vector_type(indexName);
        std::vector<float> vectorData;
        if (vectorType == "FLOAT16") {
          auto rawData = index->getDataByLabel<uint16_t>(id);
          vectorData = f16_to_floats(rawData);
        } else if (vectorType == "BFLOAT16") {
          auto rawData = index->getDataByLabel<uint16_t>(id);
          vectorData = bf16_to_floats(rawData);
        } else {
          vectorData = index->getDataByLabel<float>(id);
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

    if (indices.find(searchReq.indexName) == indices.end()) {
      return crow::response(404, "Index not found");
    }

    auto *index = indices[searchReq.indexName];
    index->setEf(searchReq.efSearch);

    std::string vectorType = get_vector_type(searchReq.indexName);
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
      DynamicBitset filteredIds = dataStores[searchReq.indexName]->filter(filters);

      FilterIdsInSet filter(filteredIds);

      if (filteredIds.count() < index->cur_element_count * EXACT_KNN_FILTER_PCT_MATCH_THRESHOLD) {
        result = index->searchExactKnn(queryData, searchReq.k, &filter);
      } else {
        result = index->searchKnn(queryData, searchReq.k, &filter);
      }
    } else {
      result = index->searchKnn(queryData, searchReq.k);
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
      auto metadatas = dataStores[searchReq.indexName]->getMany(ids);
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

  for (auto &[name, wal] : walLogs) {
    if (wal) {
      wal->stopFsyncThread();
      delete wal;
    }
  }
  walLogs.clear();

  for (auto &[name, guard] : inFlightGuards) {
    delete guard;
  }
  inFlightGuards.clear();
}
