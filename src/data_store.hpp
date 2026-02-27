#ifndef DATA_STORE_HPP
#define DATA_STORE_HPP

#include <filesystem>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "dynamic_bitset.hpp"
#include "field_value.hpp"
#include "filters.hpp"

// Type for data stores data
using KeyValueStore = std::unordered_map<int, std::map<std::string, FieldValue>>;

// Comparator for variants
struct VariantComparator {
  bool operator()(const FieldValue &lhs, const FieldValue &rhs) const;
};

// Alias for field index structure
using FieldIndex = std::unordered_map<std::string, std::map<FieldValue, std::vector<int>, VariantComparator>>;

struct Facets {
  std::unordered_map<std::string, std::unordered_map<std::string, int>> counts;
  std::unordered_map<std::string, std::tuple<int, int>> ranges;
};

class DataStore {
private:
  std::mutex mutex;

  FieldIndex fieldIndex;
  size_t maxId_ = 0;
  DynamicBitset allIds_;

  template <typename T>
  void filterByType(DynamicBitset &result, const std::string &field, const std::string &type, const FieldValue &value);

public:
  KeyValueStore data;

  DataStore() = default;
  void set(int id, std::map<std::string, FieldValue> record);
  std::map<std::string, FieldValue> get(int id);
  std::vector<std::map<std::string, FieldValue>> getMany(const std::vector<int> &ids);
  bool contains(int id);
  bool matchesFilter(int id, std::shared_ptr<FilterASTNode> filters);
  void remove(int id);
  DynamicBitset filter(std::shared_ptr<FilterASTNode> filters);
  Facets get_facets(const std::vector<int> &ids);
  size_t maxId() const { return maxId_; }
  void serialize(const std::string &filename);
  void deserialize(const std::string &filename);
};

#endif // DATA_STORE_HPP
