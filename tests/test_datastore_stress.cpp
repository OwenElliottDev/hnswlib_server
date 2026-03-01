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

    void populateDataStoreWithArrays(int numRecords) {
        std::vector<std::string> allTags = {"python", "cpp", "rust", "java", "go",
                                            "javascript", "typescript", "ruby", "swift", "kotlin"};
        for (int i = 0; i < numRecords; ++i) {
            // Each record gets 2-3 tags based on its index
            std::vector<std::string> tags;
            tags.push_back(allTags[i % 10]);
            tags.push_back(allTags[(i * 3 + 1) % 10]);
            if (i % 3 == 0) tags.push_back(allTags[(i * 7 + 2) % 10]);

            dataStore.set(i, {
                {"name", "Name" + std::to_string(i)},
                {"age", i % 100},
                {"tags", tags}
            });
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

    void benchmarkFilterString(const std::string& description, const std::string& filterString) {
        auto ast = parseFilters(filterString);
        benchmarkFilter(description, ast);
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

TEST_F(DataStoreStressTest, FilterWithIN) {
    int numRecords = 100000;
    populateDataStore(numRecords);

    // IN with small array (2 values)
    benchmarkFilterString("IN 2 values (name IN [\"Name50\",\"Name500\"])",
        R"(name IN ["Name50","Name500"])");

    // IN with medium array (5 values)
    benchmarkFilterString("IN 5 values (age IN [10,20,30,40,50])",
        "age IN [10,20,30,40,50]");

    // IN combined with AND
    benchmarkFilterString("IN + AND (name IN [...] AND age >= 50)",
        R"(name IN ["Name50","Name500","Name5000"] AND age >= 50)");
}

TEST_F(DataStoreStressTest, FilterWithCONTAINS_Substring) {
    int numRecords = 100000;
    populateDataStore(numRecords);

    // Substring match - broad (many matches)
    benchmarkFilterString("CONTAINS substring broad (name CONTAINS \"Name1\")",
        R"(name CONTAINS "Name1")");

    // Substring match - narrow (few matches)
    benchmarkFilterString("CONTAINS substring narrow (name CONTAINS \"Name9999\")",
        R"(name CONTAINS "Name9999")");
}

TEST_F(DataStoreStressTest, FilterWithCONTAINS_ArrayElement) {
    int numRecords = 100000;
    populateDataStoreWithArrays(numRecords);

    // Array element containment - single tag
    benchmarkFilterString("CONTAINS array element (tags CONTAINS \"python\")",
        R"(tags CONTAINS "python")");

    // Array element containment + AND
    benchmarkFilterString("CONTAINS array + AND (tags CONTAINS \"python\" AND age >= 50)",
        R"(tags CONTAINS "python" AND age >= 50)");

    // Array element containment - rare tag
    benchmarkFilterString("CONTAINS array element (tags CONTAINS \"swift\")",
        R"(tags CONTAINS "swift")");
}