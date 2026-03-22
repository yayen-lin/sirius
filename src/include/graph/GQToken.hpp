#pragma once

#include <string>

namespace duckdb {

// All token types the graph query lexer can produce
enum class GQTokenType {
  // Keywords
  MATCH,
  WHERE,
  COLUMNS,
  IN,
  ANY,
  SHORTEST,

  // Symbols
  LEFT_PAREN,     // (
  RIGHT_PAREN,    // )
  LEFT_BRACKET,   // [
  RIGHT_BRACKET,  // ]
  COMMA,          // ,
  COLON,          // :
  DOT,            // .
  SEMICOLON,      // ;

  // Operators
  ARROW_RIGHT,  // ->
  ARROW_LEFT,   // <-
  ARROW_STAR,   // ->*
  ARROW_PLUS,   // ->+
  DASH,         // -
  EQUALS,       // =
  STAR,         // *
  PLUS,         // +

  // Literals
  IDENTIFIER,
  INTEGER,

  END_OF_FILE
};

//  single token produced by lexer
struct GQToken {
  GQTokenType type;
  std::string lexeme;
  int line;

  GQToken(const GQTokenType t, std::string lex, const int ln)
    : type(t), lexeme(std::move(lex)), line(ln)
  {
  }
};

}  // namespace duckdb
