#pragma once

#include <string>

namespace duckdb {

// token types the graph query lexer can produce
enum class sirius_graph_query_token_type {
  // keywords
  MATCH,
  WHERE,
  COLUMNS,
  IN,
  ANY,
  SHORTEST,

  // symbols
  LEFT_PAREN,     // (
  RIGHT_PAREN,    // )
  LEFT_BRACKET,   // [
  RIGHT_BRACKET,  // ]
  COMMA,          // ,
  COLON,          // :
  DOT,            // .
  SEMICOLON,      // ;

  // operators
  ARROW_RIGHT,  // ->
  ARROW_LEFT,   // <-
  ARROW_STAR,   // ->*
  ARROW_PLUS,   // ->+
  DASH,         // -
  EQUALS,       // =
  STAR,         // *
  PLUS,         // +

  // literals
  IDENTIFIER,
  INTEGER,

  END_OF_FILE
};

//  single token produced by lexer
struct sirius_graph_query_token {
  sirius_graph_query_token_type type;
  std::string lexeme;
  int line;

  sirius_graph_query_token(const sirius_graph_query_token_type t, std::string lex, const int ln)
    : type(t), lexeme(std::move(lex)), line(ln)
  {
  }
};

}  // namespace duckdb
