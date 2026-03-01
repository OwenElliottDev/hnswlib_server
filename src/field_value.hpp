#include <string>
#include <variant>
#include <vector>

using FieldValue = std::variant<long, double, std::string, std::vector<long>, std::vector<double>, std::vector<std::string>>;