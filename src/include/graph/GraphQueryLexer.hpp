#pragma once

#include "GQToken.hpp"

#include <string>
#include <vector>

namespace duckdb {

// ============================================================
//  GraphQueryLexer
//
//  Converts a raw SQL/PGQ graph query string into a flat list
//  of GQTokens.  Mirrors the structure of Lexer.java from
//  BadLang: one pass, character by character.
//
//  Usage:
//      GraphQueryLexer lexer(query_string);
//      auto tokens = lexer.tokenize();
// ============================================================
class GraphQueryLexer {
 public:
  explicit GraphQueryLexer(const std::string& source);
  std::vector<GQToken> tokenize();

 private:
  const std::string& source_;
  std::vector<GQToken> tokens_;
  size_t start_ = 0;
  size_t pos_   = 0;
  int line_     = 1;

  // character navigation
  char advance();
  char peek() const;
  char peekNext() const;
  bool atEnd() const;
  bool match(char expected);

  void addToken(GQTokenType type);
  void scanToken();
  void number();
  void identifier();
  static GQTokenType keywordType(const std::string& text);
  [[noreturn]] void lexError(const std::string& msg) const;
};

}  // namespace duckdb
