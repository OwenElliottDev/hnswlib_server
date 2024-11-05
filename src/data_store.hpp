// data_store.hpp
#ifndef DATA_STORE_HPP
#define DATA_STORE_HPP

#include <unordered_map>
#include <map>
#include <set>
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
using FieldIndex = std::unordered_map<std::string, std::map<FieldValue, std::set<int>, VariantComparator>>;

class DataStore {
private:
    std::mutex mutex;

    FieldIndex fieldIndex;

    template<typename T>
    bool compare(const T& a, const T& b, const std::string& type);

    template<typename T>
    void filterByType(std::set<int>& result, const std::string& field, const std::string& type, const FieldValue& value);

public:
    KeyValueStore data;
    std::set<int> ids;

    DataStore() = default;
    void set(int id, std::map<std::string, FieldValue> record);
    std::map<std::string, FieldValue> get(int id);
    void remove(int id);
    std::set<int> filter(std::shared_ptr<FilterASTNode> filters);
    void serialize(const std::string &filename);
    void deserialize(const std::string &filename);
};

#endif // DATA_STORE_HPP
