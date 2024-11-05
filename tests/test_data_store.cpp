#include <gtest/gtest.h>
#include "data_store.hpp"
#include <memory>
#include <string>

class DataStoreTest : public ::testing::Test {
protected:
    DataStore dataStore;

    // Utility function to help with setting up filter nodes
    std::shared_ptr<FilterASTNode> makeComparisonFilter(
        const std::string field, 
        const std::string op, 
        FieldValue value
    ) {
        Filter filter = {field, op, value};  
        return std::make_shared<FilterASTNode>(filter);
    }
};

TEST_F(DataStoreTest, SetAndGetRecord) {
    std::map<std::string, FieldValue> record = {{"name", "Alice"}, {"age", 30L}};
    dataStore.set(1, record);
    auto retrieved = dataStore.get(1);

    EXPECT_EQ(std::get<std::string>(retrieved["name"]), "Alice");
    EXPECT_EQ(std::get<long>(retrieved["age"]), 30L);
}

TEST_F(DataStoreTest, UpdateRecord) {
    std::map<std::string, FieldValue> record1 = {{"name", "Bob"}, {"age", 25L}};
    dataStore.set(2, record1);
    std::map<std::string, FieldValue> record2 = {{"name", "Bob"}, {"age", 26L}};
    dataStore.set(2, record2);

    auto retrieved = dataStore.get(2);
    EXPECT_EQ(std::get<long>(retrieved["age"]), 26L);
}

TEST_F(DataStoreTest, RemoveRecord) {
    std::map<std::string, FieldValue> record = {{"name", "Charlie"}, {"age", 40L}};
    dataStore.set(3, record);
    dataStore.remove(3);

    EXPECT_THROW(dataStore.get(3), std::out_of_range);
}

TEST_F(DataStoreTest, FilterByComparison) {
    dataStore.set(4, {{"name", "David"}, {"age", 28L}});
    dataStore.set(5, {{"name", "Eve"}, {"age", 30L}});
    dataStore.set(6, {{"name", "Frank"}, {"age", 28L}});

    auto filter = makeComparisonFilter("age", "=", 28L);
    auto result = dataStore.filter(filter);

    std::set<int> expected = {4, 6};
    EXPECT_EQ(result, expected);
}

TEST_F(DataStoreTest, FilterWithBooleanOp) {
    dataStore.set(7, {{"name", "Grace"}, {"age", 35L}});
    dataStore.set(8, {{"name", "Heidi"}, {"age", 40L}});
    dataStore.set(9, {{"name", "Ivan"}, {"age", 45L}});

    auto ageFilter = makeComparisonFilter("age", ">=", 35L);
    auto nameFilter = makeComparisonFilter("age", "=", "Grace");

    auto root = std::make_shared<FilterASTNode>(BooleanOp::And, ageFilter, nameFilter);

    auto result = dataStore.filter(root);

    std::set<int> expected = {7};
    EXPECT_EQ(result, expected);
}

TEST_F(DataStoreTest, SerializationAndDeserialization) {
    std::string filename = "datastore_test.bin";

    dataStore.set(10, {{"name", "Jack"}, {"age", 32L}});
    dataStore.set(11, {{"name", "Karen"}, {"age", 29L}});
    dataStore.serialize(filename);

    DataStore newDataStore;
    newDataStore.deserialize(filename);

    auto retrieved = newDataStore.get(10);
    EXPECT_EQ(std::get<std::string>(retrieved["name"]), "Jack");
    EXPECT_EQ(std::get<long>(retrieved["age"]), 32L);

    retrieved = newDataStore.get(11);
    EXPECT_EQ(std::get<std::string>(retrieved["name"]), "Karen");
    EXPECT_EQ(std::get<long>(retrieved["age"]), 29L);
}

TEST_F(DataStoreTest, TestEqualLongFilter) {
    dataStore.set(12, {{"name", "Liam"}, {"age", 25L}});
    dataStore.set(13, {{"name", "Mia"}, {"age", 25L}});
    dataStore.set(14, {{"name", "Noah"}, {"age", 30L}});
    dataStore.set(15, {{"name", "Olivia"}, {"age", 30L}});

    std::string filterString = "age = 25";

    auto ast = parseFilters(filterString);

    auto result = dataStore.filter(ast);

    std::set<int> expected = {12, 13};
    EXPECT_EQ(result, expected);
}

TEST_F(DataStoreTest, TestEqualStringFilter) {
    dataStore.set(16, {{"name", "Sophia"}, {"age", 25L}});
    dataStore.set(17, {{"name", "James"}, {"age", 30L}});
    dataStore.set(18, {{"name", "James"}, {"age", 40L}});

    std::string filterString = "name = \"Sophia\"";

    auto ast = parseFilters(filterString);

    auto result = dataStore.filter(ast);

    std::set<int> expected = {16};
    EXPECT_EQ(result, expected);
}