// models.hpp
#ifndef MODELS_HPP
#define MODELS_HPP

#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "data_store.hpp"

struct IndexRequest {
    std::string index_name;
    int dimension;
    std::string index_type = "Approximate";
    std::string space_type = "IP";
    int ef_construction = 512;
    int M = 16;
};

struct AddDocumentsRequest {
    std::string index_name;
    std::vector<int> ids;
    std::vector<std::vector<float>> vectors;
    std::vector<std::map<std::string, FieldValue>> metadatas = {};
};

struct DeleteDocumentsRequest {
    std::string index_name;
    std::vector<int> ids;
};

struct SearchRequest {
    std::string index_name;
    std::vector<float> query_vector;
    int k;
    int ef_search = 512;
    std::vector<Filter> filters = {};
};

// Custom serialization for AddDocumentsRequest
inline void to_json(nlohmann::json& j, const AddDocumentsRequest& req) {
    j["index_name"] = req.index_name;
    j["ids"] = req.ids;
    j["vectors"] = req.vectors;

    // Convert metadatas manually
    j["metadatas"] = nlohmann::json::array();
    for (const auto& metadata_map : req.metadatas) {
        nlohmann::json json_metadata_map;
        for (const auto& [key, value] : metadata_map) {
            // Visit the variant to assign the appropriate type to JSON
            std::visit([&json_metadata_map, &key](auto&& arg) {
                json_metadata_map[key] = arg;
            }, value);
        }
        j["metadatas"].push_back(json_metadata_map);
    }
}

inline void from_json(const nlohmann::json& j, AddDocumentsRequest& req) {
    j.at("index_name").get_to(req.index_name);
    j.at("ids").get_to(req.ids);
    j.at("vectors").get_to(req.vectors);

    // Convert metadatas manually
    req.metadatas.clear();
    for (const auto& json_metadata_map : j.at("metadatas")) {
        std::map<std::string, FieldValue> metadata_map;
        for (const auto& [key, json_value] : json_metadata_map.items()) {
            FieldValue field_value;

            if (json_value.is_number_integer()) {
                field_value = json_value.get<int>();
            } else if (json_value.is_number_float()) {
                field_value = json_value.get<double>();
            } else if (json_value.is_string()) {
                field_value = json_value.get<std::string>();
            } else {
                throw std::invalid_argument("Unsupported type in metadatas");
            }

            metadata_map[key] = field_value;
        }
        req.metadatas.push_back(metadata_map);
    }
}

// JSON Serialization helpers for FieldValue
inline void to_json(nlohmann::json& j, const FieldValue& value) {
    std::visit([&j](auto&& arg) { j = arg; }, value);
}

inline void from_json(const nlohmann::json& j, FieldValue& value) {
    if (j.is_number_integer()) {
        // Use int64_t to safely capture larger integers
        value = j.get<int64_t>();
    } else if (j.is_number_float()) {
        value = j.get<double>();
    } else if (j.is_string()) {
        value = j.get<std::string>();
    } else {
        throw std::invalid_argument("Unsupported type for FieldValue");
    }
}

// JSON Serialization helpers for Filter
inline void to_json(nlohmann::json& j, const Filter& filter) {
    // Create the JSON object explicitly
    j["field"] = filter.field;
    j["type"] = filter.type;

    // Handle the variant value field explicitly
    std::visit([&j](const auto& arg) {
        j["value"] = arg;
    }, filter.value);
}

inline void from_json(const nlohmann::json& j, Filter& filter) {
    // Extract the simple fields
    j.at("field").get_to(filter.field);
    j.at("type").get_to(filter.type);

    // Handle the variant value field
    if (j.at("value").is_number_integer()) {
        filter.value = j.at("value").get<int>();
    } else if (j.at("value").is_number_float()) {
        filter.value = j.at("value").get<double>();
    } else if (j.at("value").is_string()) {
        filter.value = j.at("value").get<std::string>();
    } else {
        throw std::invalid_argument("Unsupported type for Filter value");
    }
}

// JSON Serialization for other structures
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(IndexRequest, index_name, dimension, index_type, space_type, ef_construction, M)
// NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(AddDocumentsRequest, index_name, ids, vectors, metadatas)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DeleteDocumentsRequest, index_name, ids)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SearchRequest, index_name, query_vector, k, ef_search, filters)

#endif // MODELS_HPP
