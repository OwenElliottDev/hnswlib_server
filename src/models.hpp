// models.hpp
#ifndef MODELS_HPP
#define MODELS_HPP

#include <string>
#include <vector>
#include <nlohmann/json.hpp>

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
};

// JSON Serialization
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(IndexRequest, index_name, dimension, index_type, space_type, ef_construction, M)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(AddDocumentsRequest, index_name, ids, vectors)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DeleteDocumentsRequest, index_name, ids)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SearchRequest, index_name, query_vector, k, ef_search)

#endif // MODELS_HPP
