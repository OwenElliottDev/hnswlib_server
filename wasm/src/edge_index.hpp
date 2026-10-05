#ifndef EDGE_INDEX_HPP
#define EDGE_INDEX_HPP

// A single in-process vector index (hnswlib graph + metadata DataStore) with the
// same semantics as one index in the HTTP server, minus WAL, threads and HTTP.
// Index files are interchangeable with the server's indices/<name>.{bin,json,data}.

#include "data_store.hpp"
#include "index_utils.hpp"
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using Metadata = std::map<std::string, FieldValue>;

struct SearchOptions {
  size_t k = 10;
  int efSearch = 512;
  std::string filter;
  bool returnMetadata = false;
  int rerankSize = 0;
};

struct SearchResult {
  std::vector<int> hits;
  std::vector<float> distances;
  std::vector<Metadata> metadatas; // populated when returnMetadata is set
};

class EdgeIndex {
public:
  // Creates an empty index. `settings` takes the same fields as the server's
  // /create_index body: dimension (required), spaceType, vectorType, M,
  // efConstruction, mrlScanDim.
  explicit EdgeIndex(const nlohmann::json &settings, size_t initialCapacity = 1024);
  ~EdgeIndex();

  EdgeIndex(const EdgeIndex &) = delete;
  EdgeIndex &operator=(const EdgeIndex &) = delete;

  // Loads an index saved by the server (or by save()). `binPath` is the
  // portable 64-bit layout .bin file; `dataPath` may be empty when there is no
  // metadata. `scratchDir` is used for the translated native-layout file.
  static std::unique_ptr<EdgeIndex> load(const std::string &binPath, const nlohmann::json &settings, const std::string &dataPath,
                                         const std::string &scratchDir);

  // Writes the server-compatible .bin and .data files. Settings are available
  // from settings().
  void save(const std::string &binPath, const std::string &dataPath, const std::string &scratchDir);

  // `vectors` is n * dimension floats, row-major. `metadatas` is empty or has n entries.
  void addDocuments(const std::vector<int> &ids, const float *vectors, size_t n, const std::vector<Metadata> &metadatas);
  void deleteDocuments(const std::vector<int> &ids);
  SearchResult search(const float *query, size_t queryDim, const SearchOptions &options);

  bool contains(int id);
  std::vector<float> getVector(int id);
  Metadata getMetadata(int id);

  const nlohmann::json &settings() const { return settings_; }
  nlohmann::json status() const;
  int dimension() const { return dimension_; }

private:
  EdgeIndex() = default;
  void init(const nlohmann::json &settings);
  void reserve(size_t additional);

  nlohmann::json settings_;
  int dimension_ = 0;
  std::string vectorType_;
  BuiltSpace space_;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;
  DataStore dataStore_;
};

#endif // EDGE_INDEX_HPP
