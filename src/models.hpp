// models.hpp
#ifndef MODELS_HPP
#define MODELS_HPP

#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "data_store.hpp"

struct IndexRequest {
    std::string indexName;
    int dimension;
    std::string indexType;
    std::string spaceType;
    int efConstruction;
    int M;
};

struct AddDocumentsRequest {
    std::string indexName;
    std::vector<int> ids;
    std::vector<std::vector<float>> vectors;
    std::vector<std::map<std::string, FieldValue>> metadatas;
};

struct DeleteDocumentsRequest {
    std::string indexName;
    std::vector<int> ids;
};

struct SearchRequest {
    std::string indexName;
    std::vector<float> queryVector;
    int k;
    int efSearch;
    std::string filter;
    bool returnMetadata;
};

// Custom serialization for AddDocumentsRequest
inline void to_json(nlohmann::json& j, const AddDocumentsRequest& req) {
    j["indexName"] = req.indexName;
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
    j.at("indexName").get_to(req.indexName);
    j.at("ids").get_to(req.ids);
    j.at("vectors").get_to(req.vectors);

    // Convert metadatas manually
    req.metadatas.clear();
    for (const auto& json_metadata_map : j.at("metadatas")) {
        std::map<std::string, FieldValue> metadata_map;
        for (const auto& [key, json_value] : json_metadata_map.items()) {
            FieldValue field_value;

            if (json_value.is_number_integer()) {
                field_value = json_value.get<long>();
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
        value = j.get<long>();
    } else if (j.is_number_float()) {
        value = j.get<double>();
    } else if (j.is_string()) {
        value = j.get<std::string>();
    } else {
        throw std::invalid_argument("Unsupported type for FieldValue");
    }
}

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(IndexRequest, indexName, dimension, indexType, spaceType, efConstruction, M)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DeleteDocumentsRequest, indexName, ids)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SearchRequest, indexName, queryVector, k, efSearch, filter, returnMetadata)

#endif // MODELS_HPP