// data_store.hpp
#ifndef DATA_STORE_HPP
#define DATA_STORE_HPP

#include <unordered_map>
#include <map>
#include <unordered_set>
#include <string>
#include <vector>
#include <stdexcept>
#include <filesystem>
#include <mutex>

#include "filters.hpp"
#include "field_value.hpp"

// Type for data stores data
using KeyValueStore = std::unordered_map<int, std::map<std::string, FieldValue>>;

// Comparator for variants
struct VariantComparator {
    bool operator()(const FieldValue& lhs, const FieldValue& rhs) const;
};

// Alias for field index structure
using FieldIndex = std::unordered_map<std::string, std::map<FieldValue, std::unordered_set<int>, VariantComparator>>;

struct Facets {
    std::unordered_map<std::string, std::unordered_map<std::string, int>> counts;
    std::unordered_map<std::string, std::tuple<int, int>> ranges;
};

class DataStore {
private:
    std::mutex mutex;

    FieldIndex fieldIndex;

    template<typename T>
    void filterByType(std::unordered_set<int>& result, const std::string& field, const std::string& type, const FieldValue& value);

public:
    KeyValueStore data;
    std::unordered_set<int> ids;

    DataStore() = default;
    void set(int id, std::map<std::string, FieldValue> record);
    std::map<std::string, FieldValue> get(int id);
    std::vector<std::map<std::string, FieldValue>> getMany(const std::vector<int>& ids);
    bool contains(int id);
    bool matchesFilter(int id, std::shared_ptr<FilterASTNode> filters);
    void remove(int id);
    std::unordered_set<int> filter(std::shared_ptr<FilterASTNode> filters);
    Facets get_facets(const std::vector<int>& ids);
    void serialize(const std::string &filename);
    void deserialize(const std::string &filename);
};

#endif // DATA_STORE_HPP
