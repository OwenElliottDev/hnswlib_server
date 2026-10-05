#include <gtest/gtest.h>
#include <string>
#include "filters.hpp"

TEST(FiltersTest, TestBasicTokenize) {
    std::string filterString = "age = 30 AND name = \"Alice\"";
    auto tokens = tokenize(filterString);
    ASSERT_EQ(tokens.size(), 7);
    ASSERT_EQ(tokens[0].value, "age");
    ASSERT_EQ(tokens[0].type, "IDENTIFIER");
    ASSERT_EQ(tokens[1].value, "=");
    ASSERT_EQ(tokens[1].type, "COMPARATOR");
    ASSERT_EQ(tokens[2].value, "30");
    ASSERT_EQ(tokens[2].type, "LONG");
    ASSERT_EQ(tokens[3].value, "AND");
    ASSERT_EQ(tokens[3].type, "BOOLEAN_OP");
    ASSERT_EQ(tokens[4].value, "name");
    ASSERT_EQ(tokens[4].type, "IDENTIFIER");
    ASSERT_EQ(tokens[5].value, "=");
    ASSERT_EQ(tokens[5].type, "COMPARATOR");
    ASSERT_EQ(tokens[6].value, "Alice");
    ASSERT_EQ(tokens[6].type, "STRING");
}

TEST(FilterTest, TestTokenizeGroups) {
    std::string filterString = "(age = 30 OR age = 31) AND name = \"Alice\"";
    auto tokens = tokenize(filterString);
    ASSERT_EQ(tokens.size(), 13);
    ASSERT_EQ(tokens[0].value, "(");
    ASSERT_EQ(tokens[0].type, "LPAREN");
    ASSERT_EQ(tokens[1].value, "age");
    ASSERT_EQ(tokens[1].type, "IDENTIFIER");
    ASSERT_EQ(tokens[2].value, "=");
    ASSERT_EQ(tokens[2].type, "COMPARATOR");
    ASSERT_EQ(tokens[3].value, "30");
    ASSERT_EQ(tokens[3].type, "LONG");
    ASSERT_EQ(tokens[4].value, "OR");
    ASSERT_EQ(tokens[4].type, "BOOLEAN_OP");
    ASSERT_EQ(tokens[5].value, "age");
    ASSERT_EQ(tokens[5].type, "IDENTIFIER");
    ASSERT_EQ(tokens[6].value, "=");
    ASSERT_EQ(tokens[6].type, "COMPARATOR");
    ASSERT_EQ(tokens[7].value, "31");
    ASSERT_EQ(tokens[7].type, "LONG");
    ASSERT_EQ(tokens[8].value, ")");
    ASSERT_EQ(tokens[8].type, "RPAREN");
    ASSERT_EQ(tokens[9].value, "AND");
    ASSERT_EQ(tokens[9].type, "BOOLEAN_OP");
    ASSERT_EQ(tokens[10].value, "name");
    ASSERT_EQ(tokens[10].type, "IDENTIFIER");
    ASSERT_EQ(tokens[11].value, "=");
    ASSERT_EQ(tokens[11].type, "COMPARATOR");
    ASSERT_EQ(tokens[12].value, "Alice");
    ASSERT_EQ(tokens[12].type, "STRING");
}

TEST(FilterTest, TestTokenizeNot) {
    std::string filterString = "NOT age = 30";
    auto tokens = tokenize(filterString);
    ASSERT_EQ(tokens.size(), 4);
    ASSERT_EQ(tokens[0].value, "NOT");
    ASSERT_EQ(tokens[0].type, "BOOLEAN_OP");
    ASSERT_EQ(tokens[1].value, "age");
    ASSERT_EQ(tokens[1].type, "IDENTIFIER");
    ASSERT_EQ(tokens[2].value, "=");
    ASSERT_EQ(tokens[2].type, "COMPARATOR");
    ASSERT_EQ(tokens[3].value, "30");
    ASSERT_EQ(tokens[3].type, "LONG");
}

TEST(FilterTest, TestASTConstruction) {
    std::string filterString = "NOT age = 30";
    auto ast = parseFilters(filterString);

    ASSERT_EQ(ast->type, NodeType::Not);
    ASSERT_EQ(ast->child->type, NodeType::Comparison);
    ASSERT_EQ(ast->child->filter.field, "age");
    ASSERT_EQ(ast->child->filter.type, "=");
    ASSERT_EQ(std::get<long>(ast->child->filter.value), 30);
}

TEST(FilterTest, TestASTConstructionWithAnd) {
    std::string filterString = "age = 30 AND name = \"Alice\"";
    auto ast = parseFilters(filterString);

    ASSERT_EQ(ast->type, NodeType::BooleanOp);
    ASSERT_EQ(ast->booleanOp, BooleanOp::And);
    ASSERT_EQ(ast->left->type, NodeType::Comparison);
    ASSERT_EQ(ast->right->type, NodeType::Comparison);
    ASSERT_EQ(ast->left->filter.field, "age");
    ASSERT_EQ(ast->left->filter.type, "=");
    ASSERT_EQ(std::get<long>(ast->left->filter.value), 30);
    ASSERT_EQ(ast->right->filter.field, "name");
    ASSERT_EQ(ast->right->filter.type, "=");
    ASSERT_EQ(std::get<std::string>(ast->right->filter.value), "Alice");
}

TEST(FilterTest, TestASTConstructionWithOr) {
    std::string filterString = "age = 30 OR name = \"Alice\"";
    auto ast = parseFilters(filterString);

    ASSERT_EQ(ast->type, NodeType::BooleanOp);
    ASSERT_EQ(ast->booleanOp, BooleanOp::Or);
    ASSERT_EQ(ast->left->type, NodeType::Comparison);
    ASSERT_EQ(ast->right->type, NodeType::Comparison);
    ASSERT_EQ(ast->left->filter.field, "age");
    ASSERT_EQ(ast->left->filter.type, "=");
    ASSERT_EQ(std::get<long>(ast->left->filter.value), 30);
    ASSERT_EQ(ast->right->filter.field, "name");
    ASSERT_EQ(ast->right->filter.type, "=");
    ASSERT_EQ(std::get<std::string>(ast->right->filter.value), "Alice");
}

TEST(FilterTest, TestTokenizeArrayStringLiteral) {
    std::string filterString = R"(name IN ["alice","bob"])";
    auto tokens = tokenize(filterString);
    ASSERT_EQ(tokens.size(), 3);
    ASSERT_EQ(tokens[0].value, "name");
    ASSERT_EQ(tokens[0].type, "IDENTIFIER");
    ASSERT_EQ(tokens[1].value, "IN");
    ASSERT_EQ(tokens[1].type, "COMPARATOR");
    ASSERT_EQ(tokens[2].type, "ARRAY_STRING");
    ASSERT_EQ(tokens[2].value, R"(["alice","bob"])");
}

TEST(FilterTest, TestTokenizeArrayWithSpaces) {
    std::string filterString = R"(name IN ["alice", "bob", "charlie"])";
    auto tokens = tokenize(filterString);
    ASSERT_EQ(tokens.size(), 3);
    ASSERT_EQ(tokens[2].type, "ARRAY_STRING");
    ASSERT_EQ(tokens[2].value, R"(["alice", "bob", "charlie"])");
}

TEST(FilterTest, TestTokenizeContains) {
    std::string filterString = R"(name CONTAINS "lic")";
    auto tokens = tokenize(filterString);
    ASSERT_EQ(tokens.size(), 3);
    ASSERT_EQ(tokens[0].value, "name");
    ASSERT_EQ(tokens[0].type, "IDENTIFIER");
    ASSERT_EQ(tokens[1].value, "CONTAINS");
    ASSERT_EQ(tokens[1].type, "COMPARATOR");
    ASSERT_EQ(tokens[2].value, "lic");
    ASSERT_EQ(tokens[2].type, "STRING");
}

TEST(FilterTest, TestASTINConstruction) {
    std::string filterString = R"(name IN ["alice","bob"])";
    auto ast = parseFilters(filterString);

    ASSERT_EQ(ast->type, NodeType::Comparison);
    ASSERT_EQ(ast->filter.field, "name");
    ASSERT_EQ(ast->filter.type, "IN");
    auto &arr = std::get<std::vector<std::string>>(ast->filter.value);
    ASSERT_EQ(arr.size(), 2);
    ASSERT_EQ(arr[0], "alice");
    ASSERT_EQ(arr[1], "bob");
}

TEST(FilterTest, TestASTContainsConstruction) {
    std::string filterString = R"(name CONTAINS "lic")";
    auto ast = parseFilters(filterString);

    ASSERT_EQ(ast->type, NodeType::Comparison);
    ASSERT_EQ(ast->filter.field, "name");
    ASSERT_EQ(ast->filter.type, "CONTAINS");
    ASSERT_EQ(std::get<std::string>(ast->filter.value), "lic");
}

TEST(FilterTest, TestTokenizeArrayLong) {
    std::string filterString = "age IN [25,30,35]";
    auto tokens = tokenize(filterString);
    ASSERT_EQ(tokens.size(), 3);
    ASSERT_EQ(tokens[2].type, "ARRAY_LONG");
}

TEST(FilterTest, TestTokenizeStringWithSpaces) {
    std::string filterString = R"(city = "New York")";
    auto tokens = tokenize(filterString);
    ASSERT_EQ(tokens.size(), 3);
    ASSERT_EQ(tokens[0].value, "city");
    ASSERT_EQ(tokens[0].type, "IDENTIFIER");
    ASSERT_EQ(tokens[1].value, "=");
    ASSERT_EQ(tokens[1].type, "COMPARATOR");
    ASSERT_EQ(tokens[2].value, "New York");
    ASSERT_EQ(tokens[2].type, "STRING");
}

TEST(FilterTest, TestASTStringWithSpaces) {
    std::string filterString = R"(city = "San Francisco")";
    auto ast = parseFilters(filterString);

    ASSERT_EQ(ast->type, NodeType::Comparison);
    ASSERT_EQ(ast->filter.field, "city");
    ASSERT_EQ(ast->filter.type, "=");
    ASSERT_EQ(std::get<std::string>(ast->filter.value), "San Francisco");
}

TEST(FilterTest, TestTokenizeContainsStringWithSpaces) {
    std::string filterString = R"(name CONTAINS "New York")";
    auto tokens = tokenize(filterString);
    ASSERT_EQ(tokens.size(), 3);
    ASSERT_EQ(tokens[0].value, "name");
    ASSERT_EQ(tokens[1].value, "CONTAINS");
    ASSERT_EQ(tokens[1].type, "COMPARATOR");
    ASSERT_EQ(tokens[2].value, "New York");
    ASSERT_EQ(tokens[2].type, "STRING");
}

TEST(FilterTest, TestTokenizeINArrayWithSpaces) {
    std::string filterString = R"(city IN ["New York","San Francisco"])";
    auto tokens = tokenize(filterString);
    ASSERT_EQ(tokens.size(), 3);
    ASSERT_EQ(tokens[0].value, "city");
    ASSERT_EQ(tokens[1].value, "IN");
    ASSERT_EQ(tokens[2].type, "ARRAY_STRING");
}

TEST(FilterTest, TestASTINWithSpaces) {
    std::string filterString = R"(city IN ["New York","San Francisco"])";
    auto ast = parseFilters(filterString);

    ASSERT_EQ(ast->type, NodeType::Comparison);
    ASSERT_EQ(ast->filter.field, "city");
    ASSERT_EQ(ast->filter.type, "IN");
    auto &arr = std::get<std::vector<std::string>>(ast->filter.value);
    ASSERT_EQ(arr.size(), 2);
    ASSERT_EQ(arr[0], "New York");
    ASSERT_EQ(arr[1], "San Francisco");
}

TEST(FilterTest, TestASTStringWithSpacesAndBooleanOp) {
    std::string filterString = R"(city = "New York" AND name = "Alice Smith")";
    auto ast = parseFilters(filterString);

    ASSERT_EQ(ast->type, NodeType::BooleanOp);
    ASSERT_EQ(ast->booleanOp, BooleanOp::And);
    ASSERT_EQ(std::get<std::string>(ast->left->filter.value), "New York");
    ASSERT_EQ(std::get<std::string>(ast->right->filter.value), "Alice Smith");
}

TEST(FilterTest, TestASTConstructionWithGroup) {
    std::string filterString = "(age = 30 OR age = 31) AND name = \"Alice\"";
    auto ast = parseFilters(filterString);

    ASSERT_EQ(ast->type, NodeType::BooleanOp);
    ASSERT_EQ(ast->booleanOp, BooleanOp::And);
    ASSERT_EQ(ast->left->type, NodeType::BooleanOp);
    ASSERT_EQ(ast->right->type, NodeType::Comparison);
    ASSERT_EQ(ast->left->booleanOp, BooleanOp::Or);
    ASSERT_EQ(ast->left->left->type, NodeType::Comparison);
    ASSERT_EQ(ast->left->right->type, NodeType::Comparison);
    ASSERT_EQ(ast->left->left->filter.field, "age");
    ASSERT_EQ(ast->left->left->filter.type, "=");
    ASSERT_EQ(std::get<long>(ast->left->left->filter.value), 30);
    ASSERT_EQ(ast->left->right->filter.field, "age");
    ASSERT_EQ(ast->left->right->filter.type, "=");
    ASSERT_EQ(std::get<long>(ast->left->right->filter.value), 31);
    ASSERT_EQ(ast->right->filter.field, "name");
    ASSERT_EQ(ast->right->filter.type, "=");
    ASSERT_EQ(std::get<std::string>(ast->right->filter.value), "Alice");
}

TEST(FilterTest, TestAndBindsTighterThanOr) {
    auto ast = parseFilters("a = 1 OR b = 2 AND c = 3");

    ASSERT_EQ(ast->type, NodeType::BooleanOp);
    ASSERT_EQ(ast->booleanOp, BooleanOp::Or);
    ASSERT_EQ(ast->left->type, NodeType::Comparison);
    ASSERT_EQ(ast->left->filter.field, "a");
    ASSERT_EQ(ast->right->type, NodeType::BooleanOp);
    ASSERT_EQ(ast->right->booleanOp, BooleanOp::And);
    ASSERT_EQ(ast->right->left->filter.field, "b");
    ASSERT_EQ(ast->right->right->filter.field, "c");
}

TEST(FilterTest, TestAndBindsTighterThanOrOnLeft) {
    auto ast = parseFilters("a = 1 AND b = 2 OR c = 3");

    ASSERT_EQ(ast->booleanOp, BooleanOp::Or);
    ASSERT_EQ(ast->left->booleanOp, BooleanOp::And);
    ASSERT_EQ(ast->left->left->filter.field, "a");
    ASSERT_EQ(ast->left->right->filter.field, "b");
    ASSERT_EQ(ast->right->filter.field, "c");
}

TEST(FilterTest, TestSameOperatorIsLeftAssociative) {
    auto ast = parseFilters("a = 1 OR b = 2 OR c = 3");

    ASSERT_EQ(ast->booleanOp, BooleanOp::Or);
    ASSERT_EQ(ast->left->booleanOp, BooleanOp::Or);
    ASSERT_EQ(ast->left->left->filter.field, "a");
    ASSERT_EQ(ast->left->right->filter.field, "b");
    ASSERT_EQ(ast->right->filter.field, "c");
}

TEST(FilterTest, TestNotBindsTighterThanAnd) {
    auto ast = parseFilters("NOT a = 1 AND b = 2");

    ASSERT_EQ(ast->booleanOp, BooleanOp::And);
    ASSERT_EQ(ast->left->type, NodeType::Not);
    ASSERT_EQ(ast->left->child->filter.field, "a");
    ASSERT_EQ(ast->right->filter.field, "b");
}

TEST(FilterTest, TestNotAppliesToGroup) {
    auto ast = parseFilters("NOT (a = 1 OR b = 2)");

    ASSERT_EQ(ast->type, NodeType::Not);
    ASSERT_EQ(ast->child->type, NodeType::BooleanOp);
    ASSERT_EQ(ast->child->booleanOp, BooleanOp::Or);
    ASSERT_EQ(ast->child->left->filter.field, "a");
    ASSERT_EQ(ast->child->right->filter.field, "b");
}

TEST(FilterTest, TestDoubleNot) {
    auto ast = parseFilters("NOT NOT a = 1");

    ASSERT_EQ(ast->type, NodeType::Not);
    ASSERT_EQ(ast->child->type, NodeType::Not);
    ASSERT_EQ(ast->child->child->filter.field, "a");
}

TEST(FilterTest, TestNestedGroups) {
    auto ast = parseFilters("((a = 1 OR b = 2) AND (c = 3 OR NOT d = 4))");

    ASSERT_EQ(ast->booleanOp, BooleanOp::And);
    ASSERT_EQ(ast->left->booleanOp, BooleanOp::Or);
    ASSERT_EQ(ast->right->booleanOp, BooleanOp::Or);
    ASSERT_EQ(ast->right->right->type, NodeType::Not);
}

TEST(FilterTest, TestNegativeNumbers) {
    auto ast = parseFilters("a > -1 AND b <= -2.5");

    ASSERT_EQ(std::get<long>(ast->left->filter.value), -1);
    ASSERT_DOUBLE_EQ(std::get<double>(ast->right->filter.value), -2.5);
}

TEST(FilterTest, TestNegativeArrays) {
    auto longs = parseFilters("a IN [-1, 2, -3]");
    std::vector<long> expectedLongs = {-1, 2, -3};
    ASSERT_EQ(std::get<std::vector<long>>(longs->filter.value), expectedLongs);

    auto doubles = parseFilters("a IN [-1.5, 2.5]");
    std::vector<double> expectedDoubles = {-1.5, 2.5};
    ASSERT_EQ(std::get<std::vector<double>>(doubles->filter.value), expectedDoubles);
}

TEST(FilterTest, TestEmptyFilterHasNoAst) {
    ASSERT_EQ(parseFilters(""), nullptr);
    ASSERT_EQ(parseFilters("   "), nullptr);
}

TEST(FilterTest, TestMalformedFiltersThrow) {
    const std::vector<std::string> malformed = {
        "a",
        "a =",
        "a = 1 AND",
        "a = 1 OR",
        "NOT",
        "NOT NOT",
        "(a = 1",
        "a = 1)",
        "()",
        "(",
        ")",
        "AND a = 1",
        "a = 1 b = 2",
        "a = 1 NOT b = 2",
        "= 1",
        "a = b",
        "a 1",
        "NOT (a = 1",
    };
    for (const auto &filter : malformed) {
        EXPECT_THROW(parseFilters(filter), std::runtime_error) << filter;
    }
}

TEST(FilterTest, TestDeepNestingIsRejected) {
    std::string deepParens = std::string(100000, '(') + "a = 1" + std::string(100000, ')');
    EXPECT_THROW(parseFilters(deepParens), std::runtime_error);

    std::string deepNots;
    for (int i = 0; i < 100000; i++) {
        deepNots += "NOT ";
    }
    EXPECT_THROW(parseFilters(deepNots + "a = 1"), std::runtime_error);

    std::string allowed = std::string(100, '(') + "a = 1" + std::string(100, ')');
    EXPECT_NO_THROW(parseFilters(allowed));
    EXPECT_NO_THROW(parseFilters("a = 1"));
}

