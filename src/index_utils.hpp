#ifndef INDEX_UTILS_HPP
#define INDEX_UTILS_HPP

// Index helpers shared by the HTTP server and the WASM build: metric space
// construction, reduced-precision vector conversion and filtered kNN dispatch.

#include "dynamic_bitset.hpp"
#include "hnswlib/hnswlib.h"
#include <algorithm>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define EXACT_KNN_FILTER_PCT_MATCH_THRESHOLD 0.1

inline hnswlib::SpaceInterface<float> *create_base_space(const std::string &spaceType, const std::string &vectorType, int dim) {
  // geodegrees is great-circle distance over (lat, lon) in float32; it ignores vectorType
  if (spaceType == "GEODEGREES") {
    return new hnswlib::GeoDegreesSpace(dim);
  }
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

// Result of building an index's metric space. For MRL indexes the space is an
// MrlSpace and the full-dimension distance function is exposed for reranking.
struct BuiltSpace {
  hnswlib::SpaceInterface<float> *space = nullptr;
  bool isMrl = false;
  hnswlib::DISTFUNC<float> fullDistFunc = nullptr;
  void *fullDistFuncParam = nullptr;
};

// Builds the metric space for an index, wrapping it in an MrlSpace when mrlScanDim > 0.
// Throws std::runtime_error on invalid configuration (e.g. geodegrees dim != 2,
// or mrlScanDim >= dim).
inline BuiltSpace build_space(const std::string &spaceType, const std::string &vectorType, int dim, int mrlScanDim) {
  BuiltSpace bs;
  if (mrlScanDim > 0) {
    if (spaceType == "GEODEGREES") {
      throw std::runtime_error("MRL is not supported for the geodegrees space");
    }
    if (mrlScanDim >= dim) {
      throw std::runtime_error("mrlScanDim must be smaller than dimension");
    }
    auto *scanSpace = create_base_space(spaceType, vectorType, mrlScanDim);
    auto *fullSpace = create_base_space(spaceType, vectorType, dim);
    auto *mrl = new hnswlib::MrlSpace(scanSpace, fullSpace);
    bs.space = mrl;
    bs.isMrl = true;
    bs.fullDistFunc = mrl->get_full_dist_func();
    bs.fullDistFuncParam = mrl->get_full_dist_func_param();
  } else {
    bs.space = create_base_space(spaceType, vectorType, dim);
  }
  return bs;
}

inline std::vector<uint16_t> floats_to_f16(const float *vec, size_t n) {
  std::vector<uint16_t> result(n);
  for (size_t i = 0; i < n; i++) {
    result[i] = hnswlib::float_to_half(vec[i]);
  }
  return result;
}

inline std::vector<uint16_t> floats_to_bf16(const float *vec, size_t n) {
  std::vector<uint16_t> result(n);
  for (size_t i = 0; i < n; i++) {
    result[i] = hnswlib::float_to_bfloat16(vec[i]);
  }
  return result;
}

inline std::vector<uint16_t> floats_to_f16(const std::vector<float> &vec) { return floats_to_f16(vec.data(), vec.size()); }

inline std::vector<uint16_t> floats_to_bf16(const std::vector<float> &vec) { return floats_to_bf16(vec.data(), vec.size()); }

inline std::vector<float> f16_to_floats(const std::vector<uint16_t> &vec) {
  std::vector<float> result(vec.size());
  for (size_t i = 0; i < vec.size(); i++) {
    result[i] = hnswlib::half_to_float(vec[i]);
  }
  return result;
}

inline std::vector<float> bf16_to_floats(const std::vector<uint16_t> &vec) {
  std::vector<float> result(vec.size());
  for (size_t i = 0; i < vec.size(); i++) {
    result[i] = hnswlib::bfloat16_to_float(vec[i]);
  }
  return result;
}

// Converts a float32 vector into the index's storage type. `scratch` owns the
// converted data for reduced-precision types; the returned pointer is valid
// for as long as both `vec` and `scratch` are.
inline const void *to_storage_vector(const std::string &vectorType, const float *vec, size_t n, std::vector<uint16_t> &scratch) {
  if (vectorType == "FLOAT16") {
    scratch = floats_to_f16(vec, n);
    return scratch.data();
  } else if (vectorType == "BFLOAT16") {
    scratch = floats_to_bf16(vec, n);
    return scratch.data();
  }
  return vec;
}

inline void addPointToIndex(hnswlib::HierarchicalNSW<float> *index, const std::string &vectorType, int id, const float *vec, size_t n) {
  std::vector<uint16_t> scratch;
  index->addPoint(to_storage_vector(vectorType, vec, n, scratch), id, true);
}

inline void addPointToIndex(hnswlib::HierarchicalNSW<float> *index, const std::string &vectorType, int id, const std::vector<float> &vec) {
  addPointToIndex(index, vectorType, id, vec.data(), vec.size());
}

// Copies a stored vector out by label. hnswlib's getDataByLabel sizes the copy
// from the distance function's dimension, which for MRL indexes is mrlScanDim,
// so this uses the full stored size instead.
template <typename T> std::vector<T> getStoredVector(hnswlib::HierarchicalNSW<float> *index, int id) {
  std::unique_lock<std::mutex> lockLabel(index->getLabelOpMutex(id));
  std::unique_lock<std::mutex> lockTable(index->label_lookup_lock);
  auto it = index->label_lookup_.find(id);
  if (it == index->label_lookup_.end() || index->isMarkedDeleted(it->second)) {
    throw std::runtime_error("Label not found");
  }
  const T *data = reinterpret_cast<const T *>(index->getDataByInternalId(it->second));
  return std::vector<T>(data, data + index->data_size_ / sizeof(T));
}

// Reads a stored vector back out as float32 regardless of storage type.
inline std::vector<float> getVectorFromIndex(hnswlib::HierarchicalNSW<float> *index, const std::string &vectorType, int id) {
  if (vectorType == "FLOAT16") {
    return f16_to_floats(getStoredVector<uint16_t>(index, id));
  } else if (vectorType == "BFLOAT16") {
    return bf16_to_floats(getStoredVector<uint16_t>(index, id));
  }
  return getStoredVector<float>(index, id);
}

// functor to filter results with a bitset of IDs
class FilterIdsInSet : public hnswlib::BaseFilterFunctor {
public:
  const DynamicBitset &ids;
  FilterIdsInSet(const DynamicBitset &ids) : ids(ids) {}
  bool operator()(hnswlib::labeltype label_id) { return ids.test(label_id); }
};

// MRL state needed at query time to rerank candidates at full dimensionality.
struct MrlParams {
  bool isMrl = false;
  hnswlib::DISTFUNC<float> fullDistFunc = nullptr;
  void *fullDistFuncParam = nullptr;
};

// Dispatches a kNN search honoring the (optional) filter bitset and MRL reranking.
// For MRL indexes with rerankSize > 0 we scan at mrlScanDim dims and rerank the
// best rerankSize candidates at full dimensionality; rerankSize == 0 returns the
// scan-dim ranking directly. Highly selective filters fall back to an exact scan,
// which is skipped for MRL since its scan distance is truncated.
// Results are returned as (hits, distances), nearest first.
inline std::pair<std::vector<int>, std::vector<float>> knn_search(hnswlib::HierarchicalNSW<float> *index, const MrlParams &mrl,
                                                                  const void *queryData, size_t k, int rerankSize,
                                                                  const DynamicBitset *filteredIds) {
  bool useMrlRerank = mrl.isMrl && rerankSize > 0;
  auto approxSearch = [&](hnswlib::BaseFilterFunctor *f) {
    if (useMrlRerank) {
      return index->searchKnnMrl(queryData, k, static_cast<size_t>(rerankSize), mrl.fullDistFunc, mrl.fullDistFuncParam, f);
    }
    return index->searchKnn(queryData, k, f);
  };

  std::priority_queue<std::pair<float, hnswlib::labeltype>> result;
  if (filteredIds) {
    FilterIdsInSet filter(*filteredIds);
    if (!mrl.isMrl && filteredIds->count() < index->cur_element_count * EXACT_KNN_FILTER_PCT_MATCH_THRESHOLD) {
      result = index->searchExactKnn(queryData, k, &filter);
    } else {
      result = approxSearch(&filter);
    }
  } else {
    result = approxSearch(nullptr);
  }

  std::vector<int> ids(result.size());
  std::vector<float> distances(result.size());
  for (size_t i = result.size(); i > 0; i--) {
    ids[i - 1] = static_cast<int>(result.top().second);
    distances[i - 1] = result.top().first;
    result.pop();
  }
  return {std::move(ids), std::move(distances)};
}

// Applies pagination to knn_search results in place: removes `excludeId` (when
// non-negative), skips the first `offset` hits, then keeps at most `k`. Callers
// should search for k + offset (+ 1 when excluding) so a full page survives.
inline void paginate_results(std::vector<int> &ids, std::vector<float> &distances, size_t offset, size_t k, int excludeId = -1) {
  if (excludeId >= 0) {
    auto it = std::find(ids.begin(), ids.end(), excludeId);
    if (it != ids.end()) {
      distances.erase(distances.begin() + (it - ids.begin()));
      ids.erase(it);
    }
  }
  size_t skip = std::min(offset, ids.size());
  ids.erase(ids.begin(), ids.begin() + skip);
  distances.erase(distances.begin(), distances.begin() + skip);
  if (ids.size() > k) {
    ids.resize(k);
    distances.resize(k);
  }
}

#endif // INDEX_UTILS_HPP
