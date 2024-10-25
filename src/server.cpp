#include "crow.h"
#include "hnswlib/hnswlib.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <unordered_map>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <set>
#include "data_store.hpp"
#include "models.hpp"
#include "filters.hpp"

#define INDEX_GROWTH_FACTOR 0.4

std::unordered_map<std::string, hnswlib::HierarchicalNSW<float>*> indices;
std::unordered_map<std::string, nlohmann::json> index_settings;
std::unordered_map<std::string, DataStore*> data_stores;

// Functor to filter results with a set of IDs
class FilterIdsInSet : public hnswlib::BaseFilterFunctor {
    public:
    std::set<int>& ids;
    FilterIdsInSet(std::set<int>& ids) : ids(ids) {}
    bool operator()(hnswlib::labeltype label_id) {
        return ids.find(label_id) != ids.end();
    }
};

void remove_index_from_disk(const std::string &index_name) {
    std::filesystem::remove("indices/" + index_name + ".bin");
    std::filesystem::remove("indices/" + index_name + ".json");
}


void write_index_to_disk(const std::string &index_name) {
    std::filesystem::create_directories("indices");

    auto *index = indices[index_name];
    if (!index) {
        std::cerr << "Error: Index not found: " << index_name << std::endl;
        return;
    }

    try {
        index->saveIndex("indices/" + index_name + ".bin");
    } catch (const std::exception &e) {
        std::cerr << "Error saving index: " << e.what() << std::endl;
        return;
    }

    std::ofstream settings_file("indices/" + index_name + ".json");
    if (!settings_file) {
        std::cerr << "Error: Unable to open settings file for writing: " << index_name << std::endl;
        return;
    }
    settings_file << index_settings[index_name].dump();
}

void read_index_from_disk(const std::string &index_name) {
    std::ifstream settings_file("indices/" + index_name + ".json");
    nlohmann::json index_state;
    settings_file >> index_state;

    int dim = index_state["dimension"];
    std::string space = index_state["space_type"];
    int ef_construction = index_state["ef_construction"];
    int M = index_state["M"];

    hnswlib::SpaceInterface<float>* metric_space = (space == "IP")
        ? static_cast<hnswlib::SpaceInterface<float>*>(new hnswlib::InnerProductSpace(dim))
        : static_cast<hnswlib::SpaceInterface<float>*>(new hnswlib::L2Space(dim));

    auto *index = new hnswlib::HierarchicalNSW<float>(
        metric_space,
        10000,
        M,
        ef_construction,
        42,
        true
    );
    std::string index_path = "indices/" + index_name + ".bin";
    index->loadIndex(index_path, metric_space, 10000);

    indices[index_name] = index;
    index_settings[index_name] = index_state;
}

int main() {
    crow::SimpleApp app;
    app.loglevel(crow::LogLevel::Warning);

    CROW_ROUTE(app, "/health").methods(crow::HTTPMethod::GET)
    ([]() {
        return "OK";
    });

    CROW_ROUTE(app, "/create_index").methods(crow::HTTPMethod::POST)
    ([](const crow::request &req) {
        auto data = nlohmann::json::parse(req.body);
        IndexRequest index_request = data.get<IndexRequest>();

        if (indices.find(index_request.index_name) != indices.end()) {
            return crow::response(400, "Index already exists");
        }

        hnswlib::SpaceInterface<float>* space = (index_request.space_type == "IP")
            ? static_cast<hnswlib::SpaceInterface<float>*>(new hnswlib::InnerProductSpace(index_request.dimension))
            : static_cast<hnswlib::SpaceInterface<float>*>(new hnswlib::L2Space(index_request.dimension));

        auto *index = new hnswlib::HierarchicalNSW<float>(
            space,
            10000,
            index_request.M,
            index_request.ef_construction,
            42,
            true
        );

        indices[index_request.index_name] = index;
        index_settings[index_request.index_name] = data;
        data_stores[index_request.index_name] = new DataStore();

        return crow::response(200, "Index created");
    });

    CROW_ROUTE(app, "/load_index").methods(crow::HTTPMethod::POST)
    ([](const crow::request &req) {
        auto data = nlohmann::json::parse(req.body);
        std::string index_name = data["index_name"];

        if (indices.find(index_name) != indices.end()) {
            return crow::response(400, "Index already exists");
        }

        read_index_from_disk(index_name);

        data_stores[index_name] = new DataStore();
        data_stores[index_name]->deserialize("indices/" + index_name + ".data");

        return crow::response(200, "Index loaded");
    });

    CROW_ROUTE(app, "/save_index").methods(crow::HTTPMethod::POST)
    ([](const crow::request &req) {
        auto data = nlohmann::json::parse(req.body);
        std::string index_name = data["index_name"];

        if (indices.find(index_name) == indices.end()) {
            return crow::response(404, "Index not found");
        }

        write_index_to_disk(index_name);
        data_stores[index_name]->serialize("indices/" + index_name + ".data");

        return crow::response(200, "Index saved");
    });

    CROW_ROUTE(app, "/delete_index").methods(crow::HTTPMethod::POST)
    ([](const crow::request &req) {
        auto data = nlohmann::json::parse(req.body);
        std::string index_name = data["index_name"];

        if (indices.find(index_name) == indices.end()) {
            return crow::response(404, "Index not found");
        }

        delete indices[index_name];
        indices.erase(index_name);
        index_settings.erase(index_name);
        delete data_stores[index_name];
        data_stores.erase(index_name);

        return crow::response(200, "Index deleted");
    });

    CROW_ROUTE(app, "/delete_index_from_disk").methods(crow::HTTPMethod::POST)
    ([](const crow::request &req) {
        auto data = nlohmann::json::parse(req.body);
        std::string index_name = data["index_name"];

        if (indices.find(index_name) != indices.end()) {
            return crow::response(400, "Index is loaded. Please unload it first");
        }

        remove_index_from_disk(index_name);

        return crow::response(200, "Index deleted from disk");
    });

    CROW_ROUTE(app, "/list_indices").methods(crow::HTTPMethod::GET)
    ([]() {
        nlohmann::json response;
        for (auto const& [index_name, _] : indices) {
            response.push_back(index_name);
        }

        return crow::response(response.dump());
    });

    CROW_ROUTE(app, "/add_documents").methods(crow::HTTPMethod::POST)
    ([](const crow::request &req) {
        auto data = nlohmann::json::parse(req.body);
        AddDocumentsRequest add_req = data.get<AddDocumentsRequest>();

        // if missing any keys, return a 400 with the details
        // If vectors is not in the request, return a 400


        if (add_req.ids.size() != add_req.vectors.size()) {
            return crow::response(400, "Number of IDs does not match number of vectors");
        }

        if (add_req.metadatas.size() > 0 && add_req.metadatas.size() != add_req.ids.size()) {
            return crow::response(400, "Number of metadatas does not match number of IDs");
        }

        if (indices.find(add_req.index_name) == indices.end()) {
            return crow::response(404, "Index not found");
        }

        auto *index = indices[add_req.index_name];

        int current_size = index->cur_element_count;

        if (current_size + add_req.ids.size() > index->max_elements_) {
            index->resizeIndex(current_size + current_size*INDEX_GROWTH_FACTOR + add_req.ids.size());
        }

        for (int i = 0; i < add_req.ids.size(); i++) {
            std::vector<float> vec_data(add_req.vectors[i].begin(), add_req.vectors[i].end());
            index->addPoint(vec_data.data(), add_req.ids[i], 0);
            if (add_req.metadatas.size()) {
                data_stores[add_req.index_name]->set(add_req.ids[i], add_req.metadatas[i]);
            }
        }

        return crow::response(200, "Documents added");
    });

    CROW_ROUTE(app, "/delete_documents").methods(crow::HTTPMethod::POST)
    ([](const crow::request &req) {
        auto data = nlohmann::json::parse(req.body);
        DeleteDocumentsRequest delete_req = data.get<DeleteDocumentsRequest>();

        if (indices.find(delete_req.index_name) == indices.end()) {
            return crow::response(404, "Index not found");
        }

        auto *index = indices[delete_req.index_name];

        for (int id : delete_req.ids) {
            index->markDelete(id);
            data_stores[delete_req.index_name]->remove(id);
        }

        return crow::response(200, "Documents deleted");
    });

    CROW_ROUTE(app, "/search").methods(crow::HTTPMethod::POST)
    ([](const crow::request &req) {
        auto data = nlohmann::json::parse(req.body);
        SearchRequest search_req = data.get<SearchRequest>();

        if (indices.find(search_req.index_name) == indices.end()) {
            return crow::response(404, "Index not found");
        }

        auto *index = indices[search_req.index_name];
        std::vector<float> query_vec(search_req.query_vector.begin(), search_req.query_vector.end());
        index->setEf(search_req.ef_search);


        std::priority_queue<std::pair<float, hnswlib::labeltype>> result;

        if (search_req.filter.size() > 0) {
            std::shared_ptr<FilterASTNode> filters = parseFilters(search_req.filter);
            std::set<int> filtered_ids = data_stores[search_req.index_name]->filter(filters);
            FilterIdsInSet filter(filtered_ids);
            result = index->searchKnn(query_vec.data(), search_req.k, &filter);
        } else {
            result = index->searchKnn(query_vec.data(), search_req.k);
        }
         
        nlohmann::json response;
        std::vector<int> ids;
        std::vector<float> distances;
        while (!result.empty()) {
            ids.push_back(result.top().second);
            distances.push_back(result.top().first);
            result.pop();
        }

        std::reverse(ids.begin(), ids.end());
        std::reverse(distances.begin(), distances.end());

        response["hits"] = ids;
        response["distances"] = distances;

        return crow::response(response.dump());
    });

    std::cout << "Server started on port 8685!" << std::endl;
    std::cout << "Press Ctrl+C to quit" << std::endl;
    std::cout << "All other stdout is suppressed as an optimisation" << std::endl;

    // Start the server
    app.port(8685).multithreaded().run();
}
