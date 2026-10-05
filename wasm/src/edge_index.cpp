#include "edge_index.hpp"
#include "hnsw_format.hpp"
#include <algorithm>
#include <cstdio>
#include <stdexcept>

namespace {

constexpr size_t LOAD_HEADROOM = 1024;

void validateSettings(const nlohmann::json &settings) {
  if (!settings.contains("dimension") || !settings["dimension"].is_number_integer() || settings["dimension"].get<int>() <= 0) {
    throw std::invalid_argument("settings.dimension must be a positive integer");
  }
  std::string space = settings.value("spaceType", "IP");
  if (space != "IP" && space != "L2" && space != "GEODEGREES") {
    throw std::invalid_argument("settings.spaceType must be one of IP, L2, GEODEGREES");
  }
  std::string vt = settings.value("vectorType", "FLOAT32");
  if (vt != "FLOAT32" && vt != "FLOAT16" && vt != "BFLOAT16") {
    throw std::invalid_argument("settings.vectorType must be one of FLOAT32, FLOAT16, BFLOAT16");
  }
}

} // namespace

void EdgeIndex::init(const nlohmann::json &settings) {
  validateSettings(settings);
  settings_ = settings;
  dimension_ = settings_["dimension"].get<int>();
  vectorType_ = settings_.value("vectorType", "FLOAT32");
  // normalise so saved settings always carry every field the server reads
  settings_["spaceType"] = settings_.value("spaceType", "IP");
  settings_["vectorType"] = vectorType_;
  settings_["M"] = settings_.value("M", 16);
  settings_["efConstruction"] = settings_.value("efConstruction", 512);
  settings_["mrlScanDim"] = settings_.value("mrlScanDim", 0);
  space_ = build_space(settings_["spaceType"], vectorType_, dimension_, settings_["mrlScanDim"]);
}

EdgeIndex::EdgeIndex(const nlohmann::json &settings, size_t initialCapacity) {
  init(settings);
  index_ =
      std::make_unique<hnswlib::HierarchicalNSW<float>>(space_.space, std::max<size_t>(initialCapacity, 1), settings_["M"].get<size_t>(),
                                                        settings_["efConstruction"].get<size_t>(), 42, true);
}

EdgeIndex::~EdgeIndex() {
  index_.reset();
  delete space_.space;
}

std::unique_ptr<EdgeIndex> EdgeIndex::load(const std::string &binPath, const nlohmann::json &settings, const std::string &dataPath,
                                           const std::string &scratchDir) {
  std::unique_ptr<EdgeIndex> idx(new EdgeIndex());
  idx->init(settings);

  std::string nativePath = scratchDir + "/edge_index_load.native.bin";
  try {
    size_t count = hnsw_format::portableToNative(binPath, nativePath);
    // size to the stored elements rather than the server's (often much larger)
    // capacity; reserve() grows geometrically if documents are added later
    idx->index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(idx->space_.space, nativePath, false, count + LOAD_HEADROOM, true);
  } catch (...) {
    std::remove(nativePath.c_str());
    throw;
  }
  std::remove(nativePath.c_str());

  // loadIndex trusts the caller's space, so catch settings that don't match the file
  if (idx->index_->label_offset_ - idx->index_->offsetData_ != idx->space_.space->get_data_size()) {
    throw std::runtime_error("Index settings do not match the index file (check dimension and vectorType)");
  }

  if (!dataPath.empty()) {
    idx->dataStore_.deserialize(dataPath);
  }
  return idx;
}

void EdgeIndex::save(const std::string &binPath, const std::string &dataPath, const std::string &scratchDir) {
  std::string nativePath = scratchDir + "/edge_index_save.native.bin";
  try {
    index_->saveIndex(nativePath);
    hnsw_format::nativeToPortable(nativePath, binPath);
  } catch (...) {
    std::remove(nativePath.c_str());
    throw;
  }
  std::remove(nativePath.c_str());
  dataStore_.serialize(dataPath);
}

void EdgeIndex::reserve(size_t additional) {
  size_t needed = index_->cur_element_count + additional;
  if (needed > index_->max_elements_) {
    index_->resizeIndex(std::max(needed, index_->max_elements_ * 2));
  }
}

void EdgeIndex::addDocuments(const std::vector<int> &ids, const float *vectors, size_t n, const std::vector<Metadata> &metadatas) {
  if (ids.size() != n) {
    throw std::invalid_argument("Number of IDs does not match number of vectors");
  }
  if (!metadatas.empty() && metadatas.size() != n) {
    throw std::invalid_argument("Number of metadatas does not match number of IDs");
  }
  for (int id : ids) {
    if (id < 0) {
      throw std::invalid_argument("Document IDs must be non-negative integers");
    }
  }

  reserve(n);
  for (size_t i = 0; i < n; i++) {
    addPointToIndex(index_.get(), vectorType_, ids[i], vectors + i * dimension_, dimension_);
    dataStore_.set(ids[i], metadatas.empty() ? Metadata{} : metadatas[i]);
  }
}

void EdgeIndex::deleteDocuments(const std::vector<int> &ids) {
  for (int id : ids) {
    try {
      index_->removePoint(id);
    } catch (const std::exception &) {
      // id not present in the index; treat as a no-op on the graph
    }
    dataStore_.remove(id);
  }
}

SearchResult EdgeIndex::search(const float *query, size_t queryDim, const SearchOptions &options) {
  if (queryDim != static_cast<size_t>(dimension_)) {
    throw std::invalid_argument("Query vector has dimension " + std::to_string(queryDim) + ", expected " + std::to_string(dimension_));
  }
  if (options.k == 0) {
    throw std::invalid_argument("k must be positive");
  }

  index_->setEf(std::max<size_t>(options.efSearch, options.k));

  std::vector<uint16_t> scratch;
  const void *queryData = to_storage_vector(vectorType_, query, queryDim, scratch);
  MrlParams mrl{space_.isMrl, space_.fullDistFunc, space_.fullDistFuncParam};

  SearchResult result;
  if (!options.filter.empty()) {
    DynamicBitset filteredIds = dataStore_.filter(parseFilters(options.filter));
    std::tie(result.hits, result.distances) = knn_search(index_.get(), mrl, queryData, options.k, options.rerankSize, &filteredIds);
  } else {
    std::tie(result.hits, result.distances) = knn_search(index_.get(), mrl, queryData, options.k, options.rerankSize, nullptr);
  }

  if (options.returnMetadata) {
    result.metadatas = dataStore_.getMany(result.hits);
  }
  return result;
}

bool EdgeIndex::contains(int id) { return dataStore_.contains(id); }

std::vector<float> EdgeIndex::getVector(int id) { return getVectorFromIndex(index_.get(), vectorType_, id); }

Metadata EdgeIndex::getMetadata(int id) { return dataStore_.get(id); }

nlohmann::json EdgeIndex::status() const {
  nlohmann::json s;
  s["currentElements"] = static_cast<size_t>(index_->cur_element_count);
  s["maxElements"] = static_cast<size_t>(index_->max_elements_);
  s["deletedElements"] = static_cast<size_t>(index_->num_deleted_);
  return s;
}
