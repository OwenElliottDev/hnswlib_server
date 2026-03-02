#include <gtest/gtest.h>
#include "data_store.hpp"
#include "dynamic_bitset.hpp"
#include <memory>
#include <string>
#include <algorithm>

class DataStoreTest : public ::testing::Test {
protected:
    DataStore dataStore;

    // utility function to help with setting up filter nodes
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

    std::vector<int> expected = {4, 6};
    EXPECT_EQ(result.to_vector(), expected);
}

TEST_F(DataStoreTest, FilterWithBooleanOp) {
    dataStore.set(7, {{"name", "Grace"}, {"age", 35L}});
    dataStore.set(8, {{"name", "Heidi"}, {"age", 40L}});
    dataStore.set(9, {{"name", "Ivan"}, {"age", 45L}});

    auto ageFilter = makeComparisonFilter("age", ">=", 35L);
    auto nameFilter = makeComparisonFilter("name", "=", "Grace");

    auto root = std::make_shared<FilterASTNode>(BooleanOp::And, ageFilter, nameFilter);

    auto result = dataStore.filter(root);

    std::vector<int> expected = {7};
    EXPECT_EQ(result.to_vector(), expected);
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

    std::vector<int> expected = {12, 13};
    EXPECT_EQ(result.to_vector(), expected);
}

TEST_F(DataStoreTest, TestEqualStringFilter) {
    dataStore.set(16, {{"name", "Sophia"}, {"age", 25L}});
    dataStore.set(17, {{"name", "James"}, {"age", 30L}});
    dataStore.set(18, {{"name", "James"}, {"age", 40L}});

    std::string filterString = "name = \"Sophia\"";

    auto ast = parseFilters(filterString);

    auto result = dataStore.filter(ast);

    std::vector<int> expected = {16};
    EXPECT_EQ(result.to_vector(), expected);
}

TEST_F(DataStoreTest, TestFilterFloatRange) {
    dataStore.set(19, {{"name", "Ava"}, {"age", 25.5}});
    dataStore.set(20, {{"name", "Logan"}, {"age", 30.5}});
    dataStore.set(21, {{"name", "Logan"}, {"age", 40.5}});

    std::string filterString = "age >= 30.0";
    auto ast = parseFilters(filterString);
    auto result = dataStore.filter(ast);
    std::vector<int> expected = {20, 21};
    EXPECT_EQ(result.to_vector(), expected);

    filterString = "age < 30.0";
    ast = parseFilters(filterString);
    result = dataStore.filter(ast);
    expected = {19};
    EXPECT_EQ(result.to_vector(), expected);
}


TEST_F(DataStoreTest, TestCountFacets) {
    dataStore.set(22, {{"name", "Emma"}, {"age", 22L}});
    dataStore.set(23, {{"name", "Oliver"}, {"age", 22L}});
    dataStore.set(24, {{"name", "Ava"}, {"age", 30L}});
    dataStore.set(25, {{"name", "Ava"}, {"age", 20L}});

    std::vector<int> ids = {22, 23, 24, 25};
    auto facets = dataStore.get_facets(ids);

    EXPECT_EQ(facets.counts["name"]["Emma"], 1);
    EXPECT_EQ(facets.counts["name"]["Oliver"], 1);
    EXPECT_EQ(facets.counts["name"]["Ava"], 2);
    EXPECT_EQ(std::get<0>(facets.ranges["age"]), 20);
    EXPECT_EQ(std::get<1>(facets.ranges["age"]), 30);
}

TEST_F(DataStoreTest, SetAndGetArrayFields) {
    std::vector<std::string> tags = {"python", "cpp", "rust"};
    std::map<std::string, FieldValue> record = {{"name", "Alice"}, {"tags", tags}};
    dataStore.set(100, record);
    auto retrieved = dataStore.get(100);

    EXPECT_EQ(std::get<std::string>(retrieved["name"]), "Alice");
    auto &retrievedTags = std::get<std::vector<std::string>>(retrieved["tags"]);
    ASSERT_EQ(retrievedTags.size(), 3);
    EXPECT_EQ(retrievedTags[0], "python");
    EXPECT_EQ(retrievedTags[1], "cpp");
    EXPECT_EQ(retrievedTags[2], "rust");
}

TEST_F(DataStoreTest, FilterIN) {
    dataStore.set(101, {{"name", "alice"}, {"age", 25L}});
    dataStore.set(102, {{"name", "bob"}, {"age", 30L}});
    dataStore.set(103, {{"name", "charlie"}, {"age", 35L}});

    std::string filterString = R"(name IN ["alice","bob"])";
    auto ast = parseFilters(filterString);
    auto result = dataStore.filter(ast);

    std::vector<int> expected = {101, 102};
    EXPECT_EQ(result.to_vector(), expected);
}

TEST_F(DataStoreTest, FilterINLong) {
    dataStore.set(104, {{"name", "alice"}, {"age", 25L}});
    dataStore.set(105, {{"name", "bob"}, {"age", 30L}});
    dataStore.set(106, {{"name", "charlie"}, {"age", 35L}});

    std::string filterString = "age IN [25,30]";
    auto ast = parseFilters(filterString);
    auto result = dataStore.filter(ast);

    std::vector<int> expected = {104, 105};
    EXPECT_EQ(result.to_vector(), expected);
}

TEST_F(DataStoreTest, FilterContainsSubstring) {
    dataStore.set(107, {{"name", "alice"}, {"age", 25L}});
    dataStore.set(108, {{"name", "bob"}, {"age", 30L}});
    dataStore.set(109, {{"name", "charlie"}, {"age", 35L}});

    std::string filterString = R"(name CONTAINS "lic")";
    auto ast = parseFilters(filterString);
    auto result = dataStore.filter(ast);

    std::vector<int> expected = {107};
    EXPECT_EQ(result.to_vector(), expected);
}

TEST_F(DataStoreTest, FilterContainsArrayElement) {
    std::vector<std::string> tags1 = {"python", "cpp"};
    std::vector<std::string> tags2 = {"java", "rust"};
    std::vector<std::string> tags3 = {"python", "javascript"};
    dataStore.set(110, {{"name", "alice"}, {"tags", tags1}});
    dataStore.set(111, {{"name", "bob"}, {"tags", tags2}});
    dataStore.set(112, {{"name", "charlie"}, {"tags", tags3}});

    std::string filterString = R"(tags CONTAINS "python")";
    auto ast = parseFilters(filterString);
    auto result = dataStore.filter(ast);

    std::vector<int> expected = {110, 112};
    EXPECT_EQ(result.to_vector(), expected);
}

TEST_F(DataStoreTest, SerializationWithArrayFields) {
    std::string filename = "datastore_array_test.bin";

    std::vector<std::string> tags = {"python", "cpp"};
    std::vector<long> scores = {10L, 20L, 30L};
    dataStore.set(113, {{"name", "Alice"}, {"tags", tags}, {"scores", scores}});
    dataStore.serialize(filename);

    DataStore newDataStore;
    newDataStore.deserialize(filename);

    auto retrieved = newDataStore.get(113);
    EXPECT_EQ(std::get<std::string>(retrieved["name"]), "Alice");

    auto &retrievedTags = std::get<std::vector<std::string>>(retrieved["tags"]);
    ASSERT_EQ(retrievedTags.size(), 2);
    EXPECT_EQ(retrievedTags[0], "python");
    EXPECT_EQ(retrievedTags[1], "cpp");

    auto &retrievedScores = std::get<std::vector<long>>(retrieved["scores"]);
    ASSERT_EQ(retrievedScores.size(), 3);
    EXPECT_EQ(retrievedScores[0], 10L);
    EXPECT_EQ(retrievedScores[1], 20L);
    EXPECT_EQ(retrievedScores[2], 30L);

    std::string filterString = R"(tags CONTAINS "python")";
    auto ast = parseFilters(filterString);
    auto result = newDataStore.filter(ast);
    std::vector<int> expected = {113};
    EXPECT_EQ(result.to_vector(), expected);

    std::remove(filename.c_str());
}

TEST_F(DataStoreTest, RemoveWithArrayFields) {
    std::vector<std::string> tags = {"python", "cpp"};
    dataStore.set(114, {{"name", "alice"}, {"tags", tags}});
    dataStore.set(115, {{"name", "bob"}, {"tags", std::vector<std::string>{"python", "java"}}});

    dataStore.remove(114);

    // alice should no longer match
    std::string filterString = R"(tags CONTAINS "cpp")";
    auto ast = parseFilters(filterString);
    auto result = dataStore.filter(ast);
    EXPECT_EQ(result.to_vector().size(), 0);

    // but python still matches bob
    filterString = R"(tags CONTAINS "python")";
    ast = parseFilters(filterString);
    result = dataStore.filter(ast);
    std::vector<int> expected = {115};
    EXPECT_EQ(result.to_vector(), expected);
}

TEST_F(DataStoreTest, MatchesFilterIN) {
    dataStore.set(116, {{"name", "alice"}});
    dataStore.set(117, {{"name", "bob"}});

    std::string filterString = R"(name IN ["alice","bob"])";
    auto ast = parseFilters(filterString);

    EXPECT_TRUE(dataStore.matchesFilter(116, ast));
    EXPECT_TRUE(dataStore.matchesFilter(117, ast));
}

TEST_F(DataStoreTest, MatchesFilterContainsSubstring) {
    dataStore.set(118, {{"name", "alice"}});
    dataStore.set(119, {{"name", "bob"}});

    std::string filterString = R"(name CONTAINS "lic")";
    auto ast = parseFilters(filterString);

    EXPECT_TRUE(dataStore.matchesFilter(118, ast));
    EXPECT_FALSE(dataStore.matchesFilter(119, ast));
}

TEST_F(DataStoreTest, SetAndGetStringWithSpaces) {
    std::map<std::string, FieldValue> record = {{"city", std::string("New York")}, {"age", 30L}};
    dataStore.set(200, record);
    auto retrieved = dataStore.get(200);

    EXPECT_EQ(std::get<std::string>(retrieved["city"]), "New York");
    EXPECT_EQ(std::get<long>(retrieved["age"]), 30L);
}

TEST_F(DataStoreTest, FilterEqualStringWithSpaces) {
    dataStore.set(201, {{"city", std::string("New York")}, {"age", 25L}});
    dataStore.set(202, {{"city", std::string("San Francisco")}, {"age", 30L}});
    dataStore.set(203, {{"city", std::string("New York")}, {"age", 35L}});

    std::string filterString = R"(city = "New York")";
    auto ast = parseFilters(filterString);
    auto result = dataStore.filter(ast);

    std::vector<int> expected = {201, 203};
    EXPECT_EQ(result.to_vector(), expected);
}

TEST_F(DataStoreTest, FilterContainsSubstringWithSpaces) {
    dataStore.set(204, {{"city", std::string("New York")}});
    dataStore.set(205, {{"city", std::string("New Orleans")}});
    dataStore.set(206, {{"city", std::string("Boston")}});

    std::string filterString = R"(city CONTAINS "New ")";
    auto ast = parseFilters(filterString);
    auto result = dataStore.filter(ast);

    std::vector<int> expected = {204, 205};
    EXPECT_EQ(result.to_vector(), expected);
}

TEST_F(DataStoreTest, FilterINStringWithSpaces) {
    dataStore.set(207, {{"city", std::string("New York")}});
    dataStore.set(208, {{"city", std::string("San Francisco")}});
    dataStore.set(209, {{"city", std::string("Boston")}});

    std::string filterString = R"(city IN ["New York","San Francisco"])";
    auto ast = parseFilters(filterString);
    auto result = dataStore.filter(ast);

    std::vector<int> expected = {207, 208};
    EXPECT_EQ(result.to_vector(), expected);
}

TEST_F(DataStoreTest, MatchesFilterStringWithSpaces) {
    dataStore.set(210, {{"city", std::string("New York")}});
    dataStore.set(211, {{"city", std::string("Boston")}});

    std::string filterString = R"(city = "New York")";
    auto ast = parseFilters(filterString);

    EXPECT_TRUE(dataStore.matchesFilter(210, ast));
    EXPECT_FALSE(dataStore.matchesFilter(211, ast));
}

TEST_F(DataStoreTest, MatchesFilterContainsArrayElement) {
    std::vector<std::string> tags1 = {"python", "cpp"};
    std::vector<std::string> tags2 = {"java", "rust"};
    dataStore.set(120, {{"tags", tags1}});
    dataStore.set(121, {{"tags", tags2}});

    std::string filterString = R"(tags CONTAINS "python")";
    auto ast = parseFilters(filterString);

    EXPECT_TRUE(dataStore.matchesFilter(120, ast));
    EXPECT_FALSE(dataStore.matchesFilter(121, ast));
}
