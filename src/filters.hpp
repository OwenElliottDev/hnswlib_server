#ifndef FILTERS_HPP
#define FILTERS_HPP

#include "field_value.hpp"
#include <memory>
#include <string>
#include <vector>

struct Filter {
  std::string field;
  std::string type; // Comparison type: =, !=, >, <, >=, <=
  FieldValue value;
};

enum class NodeType { Comparison, BooleanOp, Not };

enum class BooleanOp { And, Or };

struct Token {
  std::string value;
  std::string type;
};

class FilterASTNode {
public:
  FilterASTNode(Filter filter);
  FilterASTNode(BooleanOp op, std::shared_ptr<FilterASTNode> left, std::shared_ptr<FilterASTNode> right);
  FilterASTNode(NodeType nodeType, std::shared_ptr<FilterASTNode> child); // For NOT operations
  NodeType type;
  Filter filter;
  BooleanOp booleanOp;
  std::shared_ptr<FilterASTNode> left;
  std::shared_ptr<FilterASTNode> right;
  std::shared_ptr<FilterASTNode> child;
  std::string toString();
};

std::vector<Token> tokenize(const std::string &filterString);
std::shared_ptr<FilterASTNode> parseTerm(int &index, const std::vector<Token> &tokens);
std::shared_ptr<FilterASTNode> parseFactor(int &index, const std::vector<Token> &tokens);
std::shared_ptr<FilterASTNode> parseExpression(int &index, const std::vector<Token> &tokens);
std::shared_ptr<FilterASTNode> parseFilters(const std::string &filterString);
FieldValue convertValue(const std::string &value, const std::string &type);

#endif // FILTERS_HPP
