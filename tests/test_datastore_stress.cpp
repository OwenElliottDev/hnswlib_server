#include <gtest/gtest.h>
#include "data_store.hpp"
#include <memory>
#include <string>
#include <chrono>

class DataStoreStressTest : public ::testing::Test {
protected:
    DataStore dataStore;

    std::shared_ptr<FilterASTNode> makeComparisonFilter(
        const std::string field, 
        const std::string op, 
        FieldValue value
    ) {
        Filter filter = {field, op, value};
        return std::make_shared<FilterASTNode>(filter);
    }

    void populateDataStore(int numRecords) {
        for (int i = 0; i < numRecords; ++i) {
            dataStore.set(i, {{"name", "Name" + std::to_string(i)}, {"age", i % 100}});
        }
    }

    void benchmarkFilter(const std::string& description, std::shared_ptr<FilterASTNode> filterNode) {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = dataStore.filter(filterNode);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << description << ": Filtering took " << duration << " ms"
                  << " and matched " << result.count() << " records." << std::endl;
    }
};

TEST_F(DataStoreStressTest, FilterWithGreaterSelectors) {
    int numRecords = 10000000;  // 1 million records
    populateDataStore(numRecords);

    // Test filters with different selectivity rates
    benchmarkFilter("100% match (age >= 0)", makeComparisonFilter("age", ">=", 0L));  // Matches all records
    benchmarkFilter("75% match (age >= 25)", makeComparisonFilter("age", ">=", 25L));  // Matches 75% of records
    benchmarkFilter("50% match (age >= 50)", makeComparisonFilter("age", ">=", 50L));  // Matches 50% of records
    benchmarkFilter("25% match (age >= 75)", makeComparisonFilter("age", ">=", 75L));  // Matches 25% of records
}


TEST_F(DataStoreStressTest, FilterWithEqualSelector) {
    int numRecords = 10000000;  // 1 million records
    populateDataStore(numRecords);

    // Test filters with different selectivity rates
    benchmarkFilter("match (age = 50)", makeComparisonFilter("age", "=", 50L));
    benchmarkFilter("match (age = 500) - no records", makeComparisonFilter("age", "=", 500L));
}

TEST_F(DataStoreStressTest, FilterWithStringEqualSelector) {
    int numRecords = 10000000;  // 1 million records
    populateDataStore(numRecords);

    // Test filters with different selectivity rates
    benchmarkFilter("match (name = Name500)", makeComparisonFilter("name", "=", "Name500"));
    benchmarkFilter("match (name = Name5000)", makeComparisonFilter("name", "=", "Name5000"));
}