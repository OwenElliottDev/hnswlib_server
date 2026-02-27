// data_store.cpp
#include "data_store.hpp"
#include <fstream>
#include <algorithm>
#include <set>
#include <typeindex>
#include <climits>

bool VariantComparator::operator()(const FieldValue& lhs, const FieldValue& rhs) const {
    return lhs < rhs;
}

template<typename T>
void DataStore::filterByType(DynamicBitset& result, const std::string& field, const std::string& type, const FieldValue& value) {
    auto& fieldData = fieldIndex[field];

    if (fieldData.empty()) return;

    if (type == "=") {
        auto upper_bound = fieldData.upper_bound(value);
        auto lower_bound = fieldData.lower_bound(value);
        for (auto it = lower_bound; it != upper_bound; ++it) {
            for (int id : it->second) result.set(id);
        }
    } else if (type == "!=") {
        for (const auto& [fieldValue, ids] : fieldData) {
            if (fieldValue != value) {
                for (int id : ids) result.set(id);
            }
        }
    } else if (type == ">") {
        auto it = fieldData.upper_bound(value);
        for (; it != fieldData.end(); ++it) {
            for (int id : it->second) result.set(id);
        }
    } else if (type == "<") {
        auto upper_bound = fieldData.upper_bound(value);
        for (auto it = fieldData.begin(); it != upper_bound; ++it) {
            for (int id : it->second) result.set(id);
        }
    } else if (type == ">=") {
        auto lower_bound = fieldData.lower_bound(value);
        for (auto it = lower_bound; it != fieldData.end(); ++it) {
            for (int id : it->second) result.set(id);
        }
    } else if (type == "<=") {
        auto upper_bound = fieldData.upper_bound(value);
        for (auto it = fieldData.begin(); it != upper_bound; ++it) {
            for (int id : it->second) result.set(id);
        }
    } else {
        throw std::runtime_error("Unsupported comparison type");
    }
}

void DataStore::set(int id, std::map<std::string, FieldValue> record) {
    std::lock_guard<std::mutex> lock(mutex);
    data[id] = std::move(record);
    if (static_cast<size_t>(id) >= maxId_) {
        maxId_ = static_cast<size_t>(id);
    }
    allIds_.resize(maxId_ + 1);
    allIds_.set(id);
    for (const auto& [field, value] : data[id]) {
        fieldIndex[field][value].push_back(id);
    }
}

std::map<std::string, FieldValue> DataStore::get(int id) {
    return data.at(id);
}

bool DataStore::contains(int id) {
    return data.find(id) != data.end();
}


std::vector<std::map<std::string, FieldValue>> DataStore::getMany(const std::vector<int>& ids) {
    std::vector<std::map<std::string, FieldValue>> result;
    for (int id : ids) {
        result.push_back(data.at(id));
    }
    return result;
}

bool DataStore::matchesFilter(int id, std::shared_ptr<FilterASTNode> filters) {
    if (filters == nullptr) {
        return true;
    }

    auto record = data[id];
    switch (filters->type) {
        case NodeType::Comparison: {
            auto filter = filters->filter;
            auto field = filter.field;
            auto value = filter.value;
            auto type = filter.type;

            if (record.find(field) == record.end()) {
                return false;
            }

            auto recordValue = record[field];
            if (std::holds_alternative<long>(value)) {
                return std::get<long>(recordValue) == std::get<long>(value);
            } else if (std::holds_alternative<double>(value)) {
                return std::get<double>(recordValue) == std::get<double>(value);
            } else if (std::holds_alternative<std::string>(value)) {
                return std::get<std::string>(recordValue) == std::get<std::string>(value);
            }
            break;
        }
        case NodeType::BooleanOp: {
            auto left = matchesFilter(id, filters->left);
            auto right = matchesFilter(id, filters->right);

            if (filters->booleanOp == BooleanOp::And) {
                return left && right;
            } else {
                return left || right;
            }
            break;
        }
        case NodeType::Not: {
            return !matchesFilter(id, filters->child);
        }
    }

    return false;
}

void DataStore::remove(int id) {
    std::lock_guard<std::mutex> lock(mutex);

    if (data.find(id) == data.end()) return;
    auto record = data[id];
    for (const auto& [field, value] : record) {
        auto& vec = fieldIndex[field][value];
        vec.erase(std::remove(vec.begin(), vec.end(), id), vec.end());
    }
    data.erase(id);
    allIds_.clear(id);
}

DynamicBitset DataStore::filter(std::shared_ptr<FilterASTNode> filters) {
    if (filters == nullptr) {
        return DynamicBitset();
    }

    switch (filters->type) {
        case NodeType::Comparison: {
            auto filter = filters->filter;
            DynamicBitset result(maxId_ + 1);

            if (std::holds_alternative<long>(filter.value)) {
                filterByType<long>(result, filter.field, filter.type, filter.value);
            } else if (std::holds_alternative<double>(filter.value)) {
                filterByType<double>(result, filter.field, filter.type, filter.value);
            } else if (std::holds_alternative<std::string>(filter.value)) {
                filterByType<std::string>(result, filter.field, filter.type, filter.value);
            }
            return result;
        }
        case NodeType::BooleanOp: {
            auto left = filter(filters->left);
            auto right = filter(filters->right);

            if (filters->booleanOp == BooleanOp::And) {
                left &= right;
            } else {
                left |= right;
            }
            return left;
        }
        case NodeType::Not: {
            auto child = filter(filters->child);
            return allIds_.andNot(child);
        }
    }

    return DynamicBitset();
}

Facets DataStore::get_facets(const std::vector<int>& ids) {
    Facets facets;

    for (int id : ids) {
        const auto& document = data[id];

        for (const auto& [field, value] : document) {
            if (std::holds_alternative<long>(value) ||
                std::holds_alternative<double>(value))
            {
                auto [it, inserted] = facets.ranges.try_emplace(field, INT_MAX, INT_MIN);
                auto& range = it->second;

                int v = std::holds_alternative<long>(value)
                      ? static_cast<int>(std::get<long>(value))
                      : static_cast<int>(std::get<double>(value));

                std::get<0>(range) = std::min(std::get<0>(range), v); // min
                std::get<1>(range) = std::max(std::get<1>(range), v); // max
            }
            else if (std::holds_alternative<std::string>(value)) {
                const std::string& s = std::get<std::string>(value);
                facets.counts[field][s]++;
            }
        }
    }

    return facets;
}


namespace {
    void serializeFieldValue(std::ofstream& outFile, const FieldValue& value) {
        int index = value.index();
        outFile.write(reinterpret_cast<const char*>(&index), sizeof(index));

        switch (index) {
            case 0: // long
                {
                    long v = std::get<long>(value);
                    outFile.write(reinterpret_cast<const char*>(&v), sizeof(v));
                }
                break;
            case 1: // double
                {
                    double v = std::get<double>(value);
                    outFile.write(reinterpret_cast<const char*>(&v), sizeof(v));
                }
                break;
            case 2: // std::string
                {
                    const std::string& str = std::get<std::string>(value);
                    size_t size = str.size();
                    outFile.write(reinterpret_cast<const char*>(&size), sizeof(size));
                    outFile.write(str.data(), size);
                }
                break;
            default:
                throw std::runtime_error("Unknown variant index during serialization");
        }
    }

    FieldValue deserializeFieldValue(std::ifstream& inFile) {
        int index;
        inFile.read(reinterpret_cast<char*>(&index), sizeof(index));

        switch (index) {
            case 0: // long
                {
                    long v;
                    inFile.read(reinterpret_cast<char*>(&v), sizeof(v));
                    return v;
                }
            case 1: // double
                {
                    double v;
                    inFile.read(reinterpret_cast<char*>(&v), sizeof(v));
                    return v;
                }
            case 2: // std::string
                {
                    size_t size;
                    inFile.read(reinterpret_cast<char*>(&size), sizeof(size));
                    std::string str(size, '\0');
                    inFile.read(&str[0], size);
                    return str;
                }
            default:
                throw std::runtime_error("Unknown variant index during deserialization");
        }
    }
}

void DataStore::serialize(const std::string &filename) {
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        throw std::runtime_error("Failed to open file for serialization.");
    }

    size_t recordCount = data.size();
    outFile.write(reinterpret_cast<const char*>(&recordCount), sizeof(recordCount));

    for (const auto& [id, record] : data) {
        outFile.write(reinterpret_cast<const char*>(&id), sizeof(id));
        size_t fieldCount = record.size();
        outFile.write(reinterpret_cast<const char*>(&fieldCount), sizeof(fieldCount));

        for (const auto& [field, value] : record) {
            size_t fieldLength = field.size();
            outFile.write(reinterpret_cast<const char*>(&fieldLength), sizeof(fieldLength));
            outFile.write(field.data(), fieldLength);

            // Serialize the FieldValue object
            serializeFieldValue(outFile, value);
        }
    }
}

void DataStore::deserialize(const std::string &filename) {
    std::lock_guard<std::mutex> lock(mutex);

    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        throw std::runtime_error("Failed to open file for deserialization.");
    }

    size_t recordCount;
    inFile.read(reinterpret_cast<char*>(&recordCount), sizeof(recordCount));

    for (size_t i = 0; i < recordCount; ++i) {
        int id;
        inFile.read(reinterpret_cast<char*>(&id), sizeof(id));

        size_t fieldCount;
        inFile.read(reinterpret_cast<char*>(&fieldCount), sizeof(fieldCount));

        std::map<std::string, FieldValue> record;
        for (size_t j = 0; j < fieldCount; ++j) {
            size_t fieldLength;
            inFile.read(reinterpret_cast<char*>(&fieldLength), sizeof(fieldLength));

            std::string field(fieldLength, '\0');
            inFile.read(field.data(), fieldLength);

            // Deserialize the FieldValue object
            FieldValue value = deserializeFieldValue(inFile);

            record[field] = value;

            // Update the field index
            fieldIndex[field][value].push_back(id);
        }

        data[id] = record;
        if (static_cast<size_t>(id) >= maxId_) {
            maxId_ = static_cast<size_t>(id);
        }
        allIds_.resize(maxId_ + 1);
        allIds_.set(id);
    }
}
