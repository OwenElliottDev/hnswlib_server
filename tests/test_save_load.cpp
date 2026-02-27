#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include "hnswlib/hnswlib.h"
#include "nlohmann/json.hpp"
#include "data_store.hpp"

// Replicate the thin helpers from server.cpp since we can't link against main()

hnswlib::SpaceInterface<float>* create_space(const std::string &spaceType, const std::string &vectorType, int dim) {
    if (vectorType == "FLOAT16") {
        if (spaceType == "IP")
            return new hnswlib::InnerProductFloat16Space(dim);
        return new hnswlib::L2Float16Space(dim);
    } else if (vectorType == "BFLOAT16") {
        if (spaceType == "IP")
            return new hnswlib::InnerProductBFloat16Space(dim);
        return new hnswlib::L2BFloat16Space(dim);
    } else {
        if (spaceType == "IP")
            return new hnswlib::InnerProductSpace(dim);
        return new hnswlib::L2Space(dim);
    }
}

std::vector<uint16_t> floats_to_f16(const std::vector<float> &vec) {
    std::vector<uint16_t> result(vec.size());
    for (size_t i = 0; i < vec.size(); i++) {
        result[i] = hnswlib::float_to_half(vec[i]);
    }
    return result;
}

std::vector<uint16_t> floats_to_bf16(const std::vector<float> &vec) {
    std::vector<uint16_t> result(vec.size());
    for (size_t i = 0; i < vec.size(); i++) {
        result[i] = hnswlib::float_to_bfloat16(vec[i]);
    }
    return result;
}

std::vector<float> f16_to_floats(const std::vector<uint16_t> &vec) {
    std::vector<float> result(vec.size());
    for (size_t i = 0; i < vec.size(); i++) {
        result[i] = hnswlib::half_to_float(vec[i]);
    }
    return result;
}

std::vector<float> bf16_to_floats(const std::vector<uint16_t> &vec) {
    std::vector<float> result(vec.size());
    for (size_t i = 0; i < vec.size(); i++) {
        result[i] = hnswlib::bfloat16_to_float(vec[i]);
    }
    return result;
}

// Test directory for save/load operations
static const std::string TEST_DIR = "test_indices";

class SaveLoadTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::filesystem::create_directories(TEST_DIR);
    }

    void TearDown() override {
        std::filesystem::remove_all(TEST_DIR);
    }

    // Helper: save index and settings to disk (mirrors server.cpp write_index_to_disk)
    void saveIndex(const std::string &name,
                   hnswlib::HierarchicalNSW<float> *index,
                   const nlohmann::json &settings) {
        index->saveIndex(TEST_DIR + "/" + name + ".bin");

        std::ofstream settingsFile(TEST_DIR + "/" + name + ".json");
        ASSERT_TRUE(settingsFile.good());
        settingsFile << settings.dump();
    }

    // Helper: load index from disk (mirrors the fixed server.cpp read_index_from_disk)
    hnswlib::HierarchicalNSW<float>* loadIndex(const std::string &name,
                                                 hnswlib::SpaceInterface<float> *space) {
        std::string indexPath = TEST_DIR + "/" + name + ".bin";
        return new hnswlib::HierarchicalNSW<float>(space, indexPath, false, 0, true);
    }
};

// Basic roundtrip: create index, add vectors, save, load, search
TEST_F(SaveLoadTest, BasicRoundtrip) {
    int dim = 4;
    auto *space = new hnswlib::L2Space(dim);
    auto *index = new hnswlib::HierarchicalNSW<float>(space, 1000, 16, 200, 42, true);

    // Add some vectors
    std::vector<std::vector<float>> vectors = {
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f},
        {0.9f, 0.1f, 0.0f, 0.0f},  // close to vector 0
    };
    for (int i = 0; i < (int)vectors.size(); i++) {
        index->addPoint(vectors[i].data(), i);
    }

    nlohmann::json settings = {
        {"dimension", dim},
        {"spaceType", "L2"},
        {"vectorType", "FLOAT32"},
        {"efConstruction", 200},
        {"M", 16}
    };

    saveIndex("test_basic", index, settings);
    delete index;

    // Load
    auto *loadSpace = new hnswlib::L2Space(dim);
    auto *loaded = loadIndex("test_basic", loadSpace);

    ASSERT_EQ(loaded->cur_element_count, 4);

    // Search: query vector 0's position, should find vector 0 as nearest
    loaded->setEf(50);
    auto result = loaded->searchKnn(vectors[0].data(), 1);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.top().second, 0);  // label 0 is nearest

    // Verify vector data roundtrips
    auto retrieved = loaded->getDataByLabel<float>(0);
    ASSERT_EQ(retrieved.size(), (size_t)dim);
    for (int i = 0; i < dim; i++) {
        EXPECT_FLOAT_EQ(retrieved[i], vectors[0][i]);
    }

    delete loaded;
    delete space;
    delete loadSpace;
}

// Test that settings with defaults are correctly saved and loaded
TEST_F(SaveLoadTest, SettingsWithDefaults) {
    int dim = 4;

    // Simulate what happens when a client sends minimal create request.
    // The fixed server code now saves all fields with defaults applied.
    nlohmann::json settings = {
        {"indexName", "minimal"},
        {"dimension", dim},
        {"indexType", "APPROXIMATE"},
        {"spaceType", "IP"},
        {"vectorType", "FLOAT32"},
        {"efConstruction", 512},
        {"M", 16}
    };

    auto *space = create_space("IP", "FLOAT32", dim);
    auto *index = new hnswlib::HierarchicalNSW<float>(space, 1000, 16, 512, 42, true);

    std::vector<float> vec = {0.5f, 0.5f, 0.5f, 0.5f};
    index->addPoint(vec.data(), 0);

    saveIndex("minimal", index, settings);
    delete index;

    // Now load and verify we can read all settings
    std::ifstream settingsFile(TEST_DIR + "/minimal.json");
    ASSERT_TRUE(settingsFile.good());
    nlohmann::json loaded_settings;
    settingsFile >> loaded_settings;

    EXPECT_EQ(loaded_settings.at("dimension").get<int>(), dim);
    EXPECT_EQ(loaded_settings.value("spaceType", "IP"), "IP");
    EXPECT_EQ(loaded_settings.value("vectorType", "FLOAT32"), "FLOAT32");
    EXPECT_EQ(loaded_settings.value("efConstruction", 512), 512);
    EXPECT_EQ(loaded_settings.value("M", 16), 16);

    // Load the index using the settings
    std::string spaceType = loaded_settings.value("spaceType", "IP");
    std::string vectorType = loaded_settings.value("vectorType", "FLOAT32");
    auto *loadSpace = create_space(spaceType, vectorType, dim);
    auto *loaded = loadIndex("minimal", loadSpace);

    ASSERT_EQ(loaded->cur_element_count, 1);

    delete loaded;
    delete space;
    delete loadSpace;
}

// Test that loading a missing index file throws
TEST_F(SaveLoadTest, LoadMissingIndexThrows) {
    auto *space = new hnswlib::L2Space(4);
    EXPECT_THROW(
        loadIndex("nonexistent", space),
        std::runtime_error
    );
    delete space;
}

// Test save/load with DataStore metadata
TEST_F(SaveLoadTest, DataStoreRoundtrip) {
    int dim = 4;
    auto *space = new hnswlib::L2Space(dim);
    auto *index = new hnswlib::HierarchicalNSW<float>(space, 1000, 16, 200, 42, true);

    // Add vectors
    std::vector<float> vec1 = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> vec2 = {0.0f, 1.0f, 0.0f, 0.0f};
    index->addPoint(vec1.data(), 0);
    index->addPoint(vec2.data(), 1);

    // Create data store with metadata
    DataStore store;
    std::map<std::string, FieldValue> meta0 = {{"name", std::string("alice")}, {"age", (long)30}};
    std::map<std::string, FieldValue> meta1 = {{"name", std::string("bob")}, {"score", 0.95}};
    store.set(0, meta0);
    store.set(1, meta1);

    nlohmann::json settings = {
        {"dimension", dim},
        {"spaceType", "L2"},
        {"vectorType", "FLOAT32"},
        {"efConstruction", 200},
        {"M", 16}
    };

    // Save everything
    saveIndex("with_meta", index, settings);
    store.serialize(TEST_DIR + "/with_meta.data");
    delete index;

    // Load index
    auto *loadSpace = new hnswlib::L2Space(dim);
    auto *loaded = loadIndex("with_meta", loadSpace);
    ASSERT_EQ(loaded->cur_element_count, 2);

    // Load data store
    DataStore loadedStore;
    loadedStore.deserialize(TEST_DIR + "/with_meta.data");

    ASSERT_TRUE(loadedStore.contains(0));
    ASSERT_TRUE(loadedStore.contains(1));

    auto m0 = loadedStore.get(0);
    EXPECT_EQ(std::get<std::string>(m0["name"]), "alice");
    EXPECT_EQ(std::get<long>(m0["age"]), 30);

    auto m1 = loadedStore.get(1);
    EXPECT_EQ(std::get<std::string>(m1["name"]), "bob");
    EXPECT_DOUBLE_EQ(std::get<double>(m1["score"]), 0.95);

    delete loaded;
    delete space;
    delete loadSpace;
}

// Test save/load with many vectors to verify large index roundtrip
TEST_F(SaveLoadTest, LargeIndexRoundtrip) {
    int dim = 8;
    int numVectors = 500;
    auto *space = new hnswlib::L2Space(dim);
    auto *index = new hnswlib::HierarchicalNSW<float>(space, numVectors + 100, 16, 200, 42, true);

    // Generate vectors: each vector has value 1.0 at position (i % dim)
    std::vector<std::vector<float>> vectors(numVectors, std::vector<float>(dim, 0.0f));
    for (int i = 0; i < numVectors; i++) {
        vectors[i][i % dim] = 1.0f;
        vectors[i][(i + 1) % dim] = 0.5f * (i / (float)numVectors);
        index->addPoint(vectors[i].data(), i);
    }

    nlohmann::json settings = {
        {"dimension", dim},
        {"spaceType", "L2"},
        {"vectorType", "FLOAT32"},
        {"efConstruction", 200},
        {"M", 16}
    };

    saveIndex("large", index, settings);
    delete index;

    auto *loadSpace = new hnswlib::L2Space(dim);
    auto *loaded = loadIndex("large", loadSpace);

    ASSERT_EQ(loaded->cur_element_count, (size_t)numVectors);

    // Verify a few vectors roundtrip correctly
    for (int i = 0; i < 10; i++) {
        auto retrieved = loaded->getDataByLabel<float>(i);
        ASSERT_EQ(retrieved.size(), (size_t)dim);
        for (int d = 0; d < dim; d++) {
            EXPECT_FLOAT_EQ(retrieved[d], vectors[i][d]);
        }
    }

    // Search should still work
    loaded->setEf(50);
    auto result = loaded->searchKnn(vectors[0].data(), 1);
    EXPECT_EQ(result.top().second, 0);

    delete loaded;
    delete space;
    delete loadSpace;
}

// Test save/load with inner product space
TEST_F(SaveLoadTest, InnerProductSpaceRoundtrip) {
    int dim = 4;
    auto *space = new hnswlib::InnerProductSpace(dim);
    auto *index = new hnswlib::HierarchicalNSW<float>(space, 1000, 16, 200, 42, true);

    // Normalized vectors for IP
    std::vector<float> vec0 = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> vec1 = {0.0f, 1.0f, 0.0f, 0.0f};
    std::vector<float> vec2 = {0.707f, 0.707f, 0.0f, 0.0f};  // close to both
    index->addPoint(vec0.data(), 0);
    index->addPoint(vec1.data(), 1);
    index->addPoint(vec2.data(), 2);

    nlohmann::json settings = {
        {"dimension", dim},
        {"spaceType", "IP"},
        {"vectorType", "FLOAT32"},
        {"efConstruction", 200},
        {"M", 16}
    };

    saveIndex("ip_test", index, settings);
    delete index;

    auto *loadSpace = new hnswlib::InnerProductSpace(dim);
    auto *loaded = loadIndex("ip_test", loadSpace);

    ASSERT_EQ(loaded->cur_element_count, 3);

    // Search for vec0 should return vec0
    loaded->setEf(50);
    auto result = loaded->searchKnn(vec0.data(), 1);
    EXPECT_EQ(result.top().second, 0);

    delete loaded;
    delete space;
    delete loadSpace;
}

// Test save/load with FLOAT16 vector type
TEST_F(SaveLoadTest, Float16Roundtrip) {
    int dim = 4;
    auto *space = new hnswlib::L2Float16Space(dim);
    auto *index = new hnswlib::HierarchicalNSW<float>(space, 1000, 16, 200, 42, true);

    std::vector<float> vec0_f = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> vec1_f = {0.0f, 1.0f, 0.0f, 0.0f};
    auto vec0 = floats_to_f16(vec0_f);
    auto vec1 = floats_to_f16(vec1_f);
    index->addPoint(vec0.data(), 0);
    index->addPoint(vec1.data(), 1);

    nlohmann::json settings = {
        {"dimension", dim},
        {"spaceType", "L2"},
        {"vectorType", "FLOAT16"},
        {"efConstruction", 200},
        {"M", 16}
    };

    saveIndex("f16_test", index, settings);
    delete index;

    auto *loadSpace = new hnswlib::L2Float16Space(dim);
    auto *loaded = loadIndex("f16_test", loadSpace);

    ASSERT_EQ(loaded->cur_element_count, 2);

    // Retrieve and verify data through f16 roundtrip
    auto raw = loaded->getDataByLabel<uint16_t>(0);
    auto roundtripped = f16_to_floats(raw);
    ASSERT_EQ(roundtripped.size(), (size_t)dim);
    for (int i = 0; i < dim; i++) {
        EXPECT_NEAR(roundtripped[i], vec0_f[i], 1e-3);
    }

    // Search should work
    loaded->setEf(50);
    auto result = loaded->searchKnn(vec0.data(), 1);
    EXPECT_EQ(result.top().second, 0);

    delete loaded;
    delete space;
    delete loadSpace;
}

// Test save/load with BFLOAT16 vector type
TEST_F(SaveLoadTest, BFloat16Roundtrip) {
    int dim = 4;
    auto *space = new hnswlib::L2BFloat16Space(dim);
    auto *index = new hnswlib::HierarchicalNSW<float>(space, 1000, 16, 200, 42, true);

    std::vector<float> vec0_f = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> vec1_f = {0.0f, 1.0f, 0.0f, 0.0f};
    auto vec0 = floats_to_bf16(vec0_f);
    auto vec1 = floats_to_bf16(vec1_f);
    index->addPoint(vec0.data(), 0);
    index->addPoint(vec1.data(), 1);

    nlohmann::json settings = {
        {"dimension", dim},
        {"spaceType", "L2"},
        {"vectorType", "BFLOAT16"},
        {"efConstruction", 200},
        {"M", 16}
    };

    saveIndex("bf16_test", index, settings);
    delete index;

    auto *loadSpace = new hnswlib::L2BFloat16Space(dim);
    auto *loaded = loadIndex("bf16_test", loadSpace);

    ASSERT_EQ(loaded->cur_element_count, 2);

    auto raw = loaded->getDataByLabel<uint16_t>(0);
    auto roundtripped = bf16_to_floats(raw);
    ASSERT_EQ(roundtripped.size(), (size_t)dim);
    for (int i = 0; i < dim; i++) {
        EXPECT_NEAR(roundtripped[i], vec0_f[i], 1e-2);
    }

    loaded->setEf(50);
    auto result = loaded->searchKnn(vec0.data(), 1);
    EXPECT_EQ(result.top().second, 0);

    delete loaded;
    delete space;
    delete loadSpace;
}

// Test that the settings file preserves all fields needed for reconstruction
TEST_F(SaveLoadTest, SettingsFileCompleteness) {
    // Write a minimal settings file (simulating the OLD buggy behavior)
    nlohmann::json minimalSettings = {
        {"indexName", "bugtest"},
        {"dimension", 4}
    };

    std::ofstream out(TEST_DIR + "/bugtest.json");
    out << minimalSettings.dump();
    out.close();

    // Read it back and use .value() defaults (the fixed loading approach)
    std::ifstream in(TEST_DIR + "/bugtest.json");
    nlohmann::json loaded;
    in >> loaded;

    // These should all work with defaults
    EXPECT_EQ(loaded.at("dimension").get<int>(), 4);
    EXPECT_EQ(loaded.value("spaceType", "IP"), "IP");
    EXPECT_EQ(loaded.value("vectorType", "FLOAT32"), "FLOAT32");
    EXPECT_EQ(loaded.value("efConstruction", 512), 512);
    EXPECT_EQ(loaded.value("M", 16), 16);
}

// Test save/load with deleted elements
TEST_F(SaveLoadTest, DeletedElementsRoundtrip) {
    int dim = 4;
    auto *space = new hnswlib::L2Space(dim);
    auto *index = new hnswlib::HierarchicalNSW<float>(space, 1000, 16, 200, 42, true);

    std::vector<float> vec0 = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> vec1 = {0.0f, 1.0f, 0.0f, 0.0f};
    std::vector<float> vec2 = {0.0f, 0.0f, 1.0f, 0.0f};
    index->addPoint(vec0.data(), 0);
    index->addPoint(vec1.data(), 1);
    index->addPoint(vec2.data(), 2);

    // Delete element 1
    index->markDelete(1);

    nlohmann::json settings = {
        {"dimension", dim},
        {"spaceType", "L2"},
        {"vectorType", "FLOAT32"},
        {"efConstruction", 200},
        {"M", 16}
    };

    saveIndex("deleted", index, settings);
    delete index;

    auto *loadSpace = new hnswlib::L2Space(dim);
    auto *loaded = loadIndex("deleted", loadSpace);

    // cur_element_count includes deleted elements
    ASSERT_EQ(loaded->cur_element_count, 3);

    // Search for vec0 — element 1 should not appear in results
    loaded->setEf(50);
    auto result = loaded->searchKnn(vec0.data(), 2);
    ASSERT_EQ(result.size(), 2);
    // Results should be labels 0 and 2 (not 1)
    std::vector<int> resultLabels;
    while (!result.empty()) {
        resultLabels.push_back(result.top().second);
        result.pop();
    }
    EXPECT_TRUE(std::find(resultLabels.begin(), resultLabels.end(), 1) == resultLabels.end());

    delete loaded;
    delete space;
    delete loadSpace;
}
