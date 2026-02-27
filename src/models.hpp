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
    std::string indexType = "APPROXIMATE"; // default is approximate index
    std::string spaceType = "IP"; // default is Inner Product space
    std::string vectorType = "FLOAT32"; // "FLOAT32", "FLOAT16", or "BFLOAT16"
    int efConstruction = 512; // default value for efConstruction
    int M = 16; // default value for M
};

inline void from_json(const nlohmann::json& j, IndexRequest& req) {
    j.at("indexName").get_to(req.indexName);
    j.at("dimension").get_to(req.dimension);
    // Defaults
    req.indexType = j.value("indexType", req.indexType);
    req.spaceType = j.value("spaceType", req.spaceType);
    req.vectorType = j.value("vectorType", req.vectorType);
    req.efConstruction = j.value("efConstruction", req.efConstruction);
    req.M = j.value("M", req.M);
}

struct AddDocumentsRequest {
    std::string indexName;
    std::vector<int> ids;
    std::vector<std::vector<float>> vectors;
    std::vector<std::map<std::string, FieldValue>> metadatas = {}; // default is empty metadata
};

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
    if (!j.contains("metadatas")) {
        return; // No metadata provided
    }
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


struct DeleteDocumentsRequest {
    std::string indexName;
    std::vector<int> ids;
};

struct SearchRequest {
    std::string indexName;
    std::vector<float> queryVector;
    int k;
    int efSearch = 512; // default value
    std::string filter = ""; // filter string, default is empty (no filter)
    bool returnMetadata = false; // whether to return metadata or not, default is false
};

inline void from_json(const nlohmann::json& j, SearchRequest& req) {
    j.at("indexName").get_to(req.indexName);
    j.at("queryVector").get_to(req.queryVector);
    j.at("k").get_to(req.k);
    // defaults
    req.efSearch = j.value("efSearch", req.efSearch);
    req.filter = j.value("filter", req.filter);
    req.returnMetadata = j.value("returnMetadata", req.returnMetadata);
}

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DeleteDocumentsRequest, indexName, ids)

#endif // MODELS_HPP