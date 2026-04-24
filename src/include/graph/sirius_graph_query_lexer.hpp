#pragma once

#include "sirius_graph_query_token.hpp"

#include <string>
#include <vector>

namespace duckdb {

// converts a raw SQL/PGQ like query string into a flat list of sirius_graph_query_tokens
class sirius_graph_query_lexer {
 public:
  explicit sirius_graph_query_lexer(const std::string& source);
  std::vector<sirius_graph_query_token> tokenize();

 private:
  const std::string& source_;
  std::vector<sirius_graph_query_token> tokens_;
  size_t start_ = 0;
  size_t pos_   = 0;
  int line_     = 1;

  // character navigation
  char advance();
  [[nodiscard]] char peek() const;
  [[nodiscard]] char peekNext() const;
  [[nodiscard]] bool atEnd() const;
  bool match(char expected);

  void addToken(sirius_graph_query_token_type type);
  void scanToken();
  void number();
  void identifier();
  static sirius_graph_query_token_type keywordType(const std::string& text);
  [[noreturn]] void lexError(const std::string& msg) const;
};

}  // namespace duckdb
