#ifndef MODELS_HPP
#define MODELS_HPP

#include "data_store.hpp"
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

struct IndexRequest {
  std::string indexName;
  int dimension;
  std::string indexType = "APPROXIMATE";
  std::string spaceType = "IP";
  std::string vectorType = "FLOAT32"; // "FLOAT32", "FLOAT16", or "BFLOAT16"
  int efConstruction = 512;
  int M = 16;
  int mrlScanDim = 0; // Matryoshka (MRL): build/scan the graph at this many leading
                      // dimensions while storing full vectors. 0 disables MRL.
};

inline void from_json(const nlohmann::json &j, IndexRequest &req) {
  j.at("indexName").get_to(req.indexName);
  j.at("dimension").get_to(req.dimension);
  req.indexType = j.value("indexType", req.indexType);
  req.spaceType = j.value("spaceType", req.spaceType);
  req.vectorType = j.value("vectorType", req.vectorType);
  req.efConstruction = j.value("efConstruction", req.efConstruction);
  req.M = j.value("M", req.M);
  req.mrlScanDim = j.value("mrlScanDim", req.mrlScanDim);
}

inline void metadatas_from_json(const nlohmann::json &j, std::vector<std::map<std::string, FieldValue>> &metadatas) {
  for (const auto &json_metadata_map : j.at("metadatas")) {
    std::map<std::string, FieldValue> metadata_map;
    for (const auto &[key, json_value] : json_metadata_map.items()) {
      FieldValue field_value;

      if (json_value.is_number_integer()) {
        field_value = json_value.get<long>();
      } else if (json_value.is_number_float()) {
        field_value = json_value.get<double>();
      } else if (json_value.is_string()) {
        field_value = json_value.get<std::string>();
      } else if (json_value.is_array()) {
        if (json_value.empty() || json_value[0].is_string()) {
          std::vector<std::string> arr;
          for (const auto &el : json_value)
            arr.push_back(el.get<std::string>());
          field_value = arr;
        } else if (json_value[0].is_number_integer()) {
          std::vector<long> arr;
          for (const auto &el : json_value)
            arr.push_back(el.get<long>());
          field_value = arr;
        } else if (json_value[0].is_number_float()) {
          std::vector<double> arr;
          for (const auto &el : json_value)
            arr.push_back(el.get<double>());
          field_value = arr;
        } else {
          throw std::invalid_argument("Unsupported array element type in metadatas");
        }
      } else {
        throw std::invalid_argument("Unsupported type in metadatas");
      }

      metadata_map[key] = field_value;
    }
    metadatas.push_back(metadata_map);
  }
}

struct AddDocumentsRequest {
  std::string indexName;
  std::vector<int> ids;
  std::vector<std::vector<float>> vectors;
  std::vector<std::map<std::string, FieldValue>> metadatas = {};
};

inline void to_json(nlohmann::json &j, const AddDocumentsRequest &req) {
  j["indexName"] = req.indexName;
  j["ids"] = req.ids;
  j["vectors"] = req.vectors;

  j["metadatas"] = nlohmann::json::array();
  for (const auto &metadata_map : req.metadatas) {
    nlohmann::json json_metadata_map;
    for (const auto &[key, value] : metadata_map) {
      std::visit([&json_metadata_map, &key](auto &&arg) { json_metadata_map[key] = arg; }, value);
    }
    j["metadatas"].push_back(json_metadata_map);
  }
}

inline void from_json(const nlohmann::json &j, AddDocumentsRequest &req) {
  j.at("indexName").get_to(req.indexName);
  j.at("ids").get_to(req.ids);
  j.at("vectors").get_to(req.vectors);

  if (!j.contains("metadatas")) {
    return;
  }
  metadatas_from_json(j, req.metadatas);
}

struct UpdateDocumentsRequest {
  std::string indexName;
  std::vector<int> ids;
  std::vector<std::map<std::string, std::optional<FieldValue>>> metadatas;
};

inline void from_json(const nlohmann::json &j, UpdateDocumentsRequest &req) {
  j.at("indexName").get_to(req.indexName);
  j.at("ids").get_to(req.ids);

  nlohmann::json toSet = {{"metadatas", nlohmann::json::array()}};
  for (const auto &patch : j.at("metadatas")) {
    if (!patch.is_object()) {
      throw std::invalid_argument("Each metadatas entry must be an object");
    }
    nlohmann::json fields = nlohmann::json::object();
    for (const auto &[key, value] : patch.items()) {
      if (!value.is_null()) {
        fields[key] = value;
      }
    }
    toSet["metadatas"].push_back(std::move(fields));
  }
  std::vector<std::map<std::string, FieldValue>> parsed;
  metadatas_from_json(toSet, parsed);

  const auto &patches = j.at("metadatas");
  for (size_t i = 0; i < parsed.size(); i++) {
    std::map<std::string, std::optional<FieldValue>> patch(parsed[i].begin(), parsed[i].end());
    for (const auto &[key, value] : patches[i].items()) {
      if (value.is_null()) {
        patch[key] = std::nullopt;
      }
    }
    req.metadatas.push_back(std::move(patch));
  }
}

// JSON Serialization helpers for FieldValue
inline void to_json(nlohmann::json &j, const FieldValue &value) {
  std::visit([&j](auto &&arg) { j = arg; }, value);
}

inline void from_json(const nlohmann::json &j, FieldValue &value) {
  if (j.is_number_integer()) {
    value = j.get<long>();
  } else if (j.is_number_float()) {
    value = j.get<double>();
  } else if (j.is_string()) {
    value = j.get<std::string>();
  } else if (j.is_array()) {
    if (j.empty() || j[0].is_string()) {
      std::vector<std::string> arr;
      for (const auto &el : j)
        arr.push_back(el.get<std::string>());
      value = arr;
    } else if (j[0].is_number_integer()) {
      std::vector<long> arr;
      for (const auto &el : j)
        arr.push_back(el.get<long>());
      value = arr;
    } else if (j[0].is_number_float()) {
      std::vector<double> arr;
      for (const auto &el : j)
        arr.push_back(el.get<double>());
      value = arr;
    } else {
      throw std::invalid_argument("Unsupported array element type for FieldValue");
    }
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
  int offset = 0;
  int efSearch = 512;
  std::string filter = "";
  bool returnMetadata = false;
  int rerankSize = 0; // MRL only: rerank the best rerankSize scan-dim candidates at full
                      // dimensionality and return the top k. 0 means no reranking.
};

inline void from_json(const nlohmann::json &j, SearchRequest &req) {
  j.at("indexName").get_to(req.indexName);
  j.at("queryVector").get_to(req.queryVector);
  j.at("k").get_to(req.k);
  req.offset = j.value("offset", req.offset);
  req.efSearch = j.value("efSearch", req.efSearch);
  req.filter = j.value("filter", req.filter);
  req.returnMetadata = j.value("returnMetadata", req.returnMetadata);
  req.rerankSize = j.value("rerankSize", req.rerankSize);
}

struct SimilarRequest {
  std::string indexName;
  int docId;
  int k;
  int offset = 0;
  bool excludeInputDocument = true;
  int efSearch = 512;
  std::string filter = "";
  bool returnMetadata = false;
  int rerankSize = 0; // MRL only: rerank the best rerankSize scan-dim candidates at full
                      // dimensionality and return the top k. 0 means no reranking.
};

inline void from_json(const nlohmann::json &j, SimilarRequest &req) {
  j.at("indexName").get_to(req.indexName);
  j.at("docId").get_to(req.docId);
  j.at("k").get_to(req.k);
  req.offset = j.value("offset", req.offset);
  req.excludeInputDocument = j.value("excludeInputDocument", req.excludeInputDocument);
  req.efSearch = j.value("efSearch", req.efSearch);
  req.filter = j.value("filter", req.filter);
  req.returnMetadata = j.value("returnMetadata", req.returnMetadata);
  req.rerankSize = j.value("rerankSize", req.rerankSize);
}

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DeleteDocumentsRequest, indexName, ids)

#endif // MODELS_HPP