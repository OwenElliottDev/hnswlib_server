// data_store.hpp
#ifndef DATA_STORE_HPP
#define DATA_STORE_HPP

#include <unordered_map>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <variant>
#include <stdexcept>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <typeindex>

using FieldValue = std::variant<int, long, float, double, std::string>;

using KeyValueStore = std::unordered_map<int, std::map<std::string, FieldValue>>;

struct VariantComparator {
    bool operator()(const FieldValue& lhs, const FieldValue& rhs) const {
        return lhs < rhs;
    }
};

using FieldIndex = std::map<std::string, std::map<FieldValue, std::set<int>, VariantComparator>>;

struct Filter {
    std::string field;
    std::string type;
    FieldValue value;
};

class DataStore {
private:
    FieldIndex field_index;

    template<typename T>
    bool compare(const T& a, const T& b, std::string type) {
        if (type=="EQUAL") return a == b;
        if (type=="NOT_EQUAL") return a != b;
        if (type=="GREATER_THAN") return a > b;
        if (type=="LESS_THAN") return a < b;
        if (type=="GREATER_THAN_EQUAL") return a >= b;
        if (type=="LESS_THAN_EQUAL") return a <= b;
        throw std::runtime_error("Unsupported comparison type");
    }

    template<typename T>
    std::set<int> filterByType(const std::map<FieldValue, std::set<int>, VariantComparator>& fieldData, std::string type, const FieldValue& value) {
        std::set<int> result;
        for (const auto& [fieldValue, ids] : fieldData) {
            if (auto castedFieldValue = std::get_if<T>(&fieldValue)) {
                if (compare(*castedFieldValue, std::get<T>(value), type)) {
                    result.insert(ids.begin(), ids.end());
                }
            }
        }
        return result;
    }

public:
    KeyValueStore data;

    DataStore() = default;

    void set(int id, std::map<std::string, FieldValue> record) {
        data[id] = record;
        for (const auto& [field, value] : record) {
            field_index[field][value].insert(id);
        }
    }

    std::map<std::string, FieldValue> get(int id) {
        return data[id];
    }

    void remove(int id) {
        auto record = data[id];
        for (const auto& [field, value] : record) {
            field_index[field][value].erase(id);
        }
        data.erase(id);
    }

    std::set<int> filter(const std::vector<Filter>& filters) {
        std::set<int> result;
        bool firstFilter = true;

        for (const auto& filter : filters) {
            const auto& field = filter.field;
            const auto& value = filter.value;
            auto type = filter.type;

            if (field_index.find(field) == field_index.end()) {
                continue;
            }

            const auto& fieldData = field_index[field];
            std::set<int> currentResult;

            if (std::holds_alternative<int>(value)) {
                currentResult = filterByType<int>(fieldData, type, value);
            } else if (std::holds_alternative<long>(value)) {
                currentResult = filterByType<long>(fieldData, type, value);
            } else if (std::holds_alternative<float>(value)) {
                currentResult = filterByType<float>(fieldData, type, value);
            } else if (std::holds_alternative<double>(value)) {
                currentResult = filterByType<double>(fieldData, type, value);
            } else if (std::holds_alternative<std::string>(value)) {
                currentResult = filterByType<std::string>(fieldData, type, value);
            } else {
                continue;
            }

            if (firstFilter) {
                result = std::move(currentResult);
                firstFilter = false;
            } else {
                std::set<int> intersection;
                std::set_intersection(result.begin(), result.end(),
                                      currentResult.begin(), currentResult.end(),
                                      std::inserter(intersection, intersection.begin()));
                result = std::move(intersection);
            }
        }

        return result;
    }

    void serialize(const std::string &filename) {
        size_t size = data.size();
    }

    void deserialize(const std::string &filename) {
        // Read data from disk
    }
};

#endif // DATA_STORE_HPP