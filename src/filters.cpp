#include "filters.hpp"
#include <regex>
#include <sstream>

FilterASTNode::FilterASTNode(Filter filter) : type(NodeType::Comparison), filter(filter) {}

FilterASTNode::FilterASTNode(BooleanOp op, std::shared_ptr<FilterASTNode> left, std::shared_ptr<FilterASTNode> right)
    : type(NodeType::BooleanOp), booleanOp(op), left(left), right(right) {}

FilterASTNode::FilterASTNode(NodeType nodeType, std::shared_ptr<FilterASTNode> child) : type(nodeType), child(child) {}

std::string FilterASTNode::toString() {
  std::string value;
  switch (type) {
  case NodeType::Comparison:
    if (std::holds_alternative<long>(filter.value)) {
      value = std::to_string(std::get<long>(filter.value));
    } else if (std::holds_alternative<double>(filter.value)) {
      value = std::to_string(std::get<double>(filter.value));
    } else if (std::holds_alternative<std::string>(filter.value)) {
      value = std::get<std::string>(filter.value);
    }
    return filter.field + " " + filter.type + " " + value;
  case NodeType::BooleanOp:
    return left->toString() + " " + (booleanOp == BooleanOp::And ? "AND" : "OR") + " " + right->toString();
  case NodeType::Not:
    return "NOT " + child->toString();
  }
  return "Invalid node type";
}

FieldValue convertType(const std::string &value, const std::string &type) {
  if (type == "STRING") {
    return value;
  } else if (type == "LONG") {
    return std::stol(value);
  } else if (type == "DOUBLE") {
    return std::stod(value);
  } else {
    throw std::runtime_error("Unsupported type: " + type);
  }
}

const std::regex LPAREN(R"(\()");
const std::regex RPAREN(R"(\))");
const std::regex STRING("\"([^\"]*)\"");
const std::regex LONG(R"(\d+)");
const std::regex DOUBLE(R"(\d+\.\d+)");
const std::regex COMPARATOR(R"(!=|>=|<=|=|>|<)");
const std::regex BOOLEAN_OP(R"(AND|OR|NOT)");
const std::regex IDENTIFIER(R"(\w+)");
const std::regex WHITESPACE(R"(\s+)");

std::vector<std::string> splitWhitespace(const std::string &str) {
  std::vector<std::string> tokens;
  std::istringstream stream(str);
  std::string token;
  while (stream >> token) {
    std::smatch match;
    if (std::regex_search(token, match, LPAREN)) {
      tokens.push_back(match[0]);
      token.erase(0, 1);
      tokens.push_back(token);
    } else if (std::regex_search(token, match, RPAREN)) {
      tokens.push_back(token.substr(0, token.size() - 1));
      tokens.push_back(match[0]);
    } else {
      tokens.push_back(token);
    }
  }
  return tokens;
}

std::vector<Token> tokenize(const std::string &filterString) {
  auto splitTokens = splitWhitespace(filterString);
  std::vector<Token> tokens;

  for (const auto &word : splitTokens) {
    std::smatch match;
    if (std::regex_match(word, match, LPAREN)) {
      tokens.push_back({word, "LPAREN"});
    } else if (std::regex_match(word, match, RPAREN)) {
      tokens.push_back({word, "RPAREN"});
    } else if (std::regex_match(word, match, COMPARATOR)) {
      tokens.push_back({word, "COMPARATOR"});
    } else if (std::regex_match(word, match, BOOLEAN_OP)) {
      tokens.push_back({word, "BOOLEAN_OP"});
    } else if (std::regex_match(word, match, DOUBLE)) {
      tokens.push_back({word, "DOUBLE"});
    } else if (std::regex_match(word, match, LONG)) {
      tokens.push_back({word, "LONG"});
    } else if (std::regex_match(word, match, STRING)) {
      tokens.push_back({match[1], "STRING"});
    } else if (std::regex_match(word, match, IDENTIFIER)) {
      tokens.push_back({word, "IDENTIFIER"});
    } else {
      throw std::runtime_error("Invalid token in filter string: " + word);
    }
  }

  return tokens;
}

std::shared_ptr<FilterASTNode> parseTerm(int &index, const std::vector<Token> &tokens) {
  if (index >= tokens.size()) {
    return nullptr;
  }

  std::shared_ptr<FilterASTNode> astNode;

  if (tokens[index].type == "LPAREN") {
    index++;
    astNode = parseExpression(index, tokens);
    if (tokens[index].type != "RPAREN") {
      throw std::runtime_error("Expected closing parenthesis at index " + std::to_string(index) +
                               " instead we found: " + tokens[index].value);
    }
    index++;
  } else {
    astNode = parseFactor(index, tokens);
  }
  return astNode;
}

std::shared_ptr<FilterASTNode> parseFactor(int &index, const std::vector<Token> &tokens) {
  if (index >= tokens.size()) {
    return nullptr;
  }

  if (tokens[index].type == "BOOLEAN_OP" && tokens[index].value == "NOT") {
    index++;
    auto child = parseFactor(index, tokens);
    return std::make_shared<FilterASTNode>(NodeType::Not, child);
  }

  if (tokens[index].type == "IDENTIFIER") {
    auto field = tokens[index].value;
    index++;
    if (tokens[index].type != "COMPARATOR") {
      throw std::runtime_error("Expected a comparator after an identifier. After identifier: " + tokens[index - 1].value +
                               " found: " + tokens[index].value);
    }
    auto op = tokens[index].value;
    index++;
    FieldValue convertedValue = convertType(tokens[index].value, tokens[index].type);
    index++;
    return std::make_shared<FilterASTNode>(Filter{field, op, convertedValue});
  }
  throw std::runtime_error("Syntax error in filter string");
}

std::shared_ptr<FilterASTNode> parseExpression(int &index, const std::vector<Token> &tokens) {
  auto astNode = parseTerm(index, tokens);
  while (index < tokens.size() && tokens[index].type == "BOOLEAN_OP" && tokens[index].value != "NOT") {
    auto op = tokens[index].value == "AND" ? BooleanOp::And : BooleanOp::Or;
    index++;
    auto right = parseTerm(index, tokens);
    astNode = std::make_shared<FilterASTNode>(op, astNode, right);
  }
  // index++;
  return astNode;
}

std::shared_ptr<FilterASTNode> parseFilters(const std::string &filterString) {
  auto tokens = tokenize(filterString);

  int index = 0;
  auto astNode = parseExpression(index, tokens);

  if (index < tokens.size()) {
    throw std::runtime_error("Unexpected token at index " + std::to_string(index) + ": " + tokens[index].value);
  }

  return astNode;
}