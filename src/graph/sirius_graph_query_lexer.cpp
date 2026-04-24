#include "graph/sirius_graph_query_lexer.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace duckdb {

sirius_graph_query_lexer::sirius_graph_query_lexer(const std::string& source) : source_(source) {}

std::vector<sirius_graph_query_token> sirius_graph_query_lexer::tokenize()
{
  while (!atEnd()) {
    start_ = pos_;
    scanToken();
  }
  tokens_.emplace_back(sirius_graph_query_token_type::END_OF_FILE, "", line_);
  return tokens_;
}

char sirius_graph_query_lexer::advance() { return source_[pos_++]; }

char sirius_graph_query_lexer::peek() const { return atEnd() ? '\0' : source_[pos_]; }

char sirius_graph_query_lexer::peekNext() const
{ return (pos_ + 1 >= source_.size()) ? '\0' : source_[pos_ + 1]; }

bool sirius_graph_query_lexer::atEnd() const { return pos_ >= source_.size(); }

bool sirius_graph_query_lexer::match(char expected)
{
  if (atEnd() || source_[pos_] != expected) return false;
  pos_++;
  return true;
}

void sirius_graph_query_lexer::addToken(sirius_graph_query_token_type type)
{ tokens_.emplace_back(type, source_.substr(start_, pos_ - start_), line_); }

void sirius_graph_query_lexer::scanToken()
{
  char c = advance();
  switch (c) {
    case ' ':
    case '\r':
    case '\t': break;
    case '\n': line_++; break;
    case '(': addToken(sirius_graph_query_token_type::LEFT_PAREN); break;
    case ')': addToken(sirius_graph_query_token_type::RIGHT_PAREN); break;
    case '[': addToken(sirius_graph_query_token_type::LEFT_BRACKET); break;
    case ']': addToken(sirius_graph_query_token_type::RIGHT_BRACKET); break;
    case ',': addToken(sirius_graph_query_token_type::COMMA); break;
    case ':': addToken(sirius_graph_query_token_type::COLON); break;
    case '.': addToken(sirius_graph_query_token_type::DOT); break;
    case ';': addToken(sirius_graph_query_token_type::SEMICOLON); break;
    case '=': addToken(sirius_graph_query_token_type::EQUALS); break;
    case '*': addToken(sirius_graph_query_token_type::STAR); break;
    case '+': addToken(sirius_graph_query_token_type::PLUS); break;

    // '<' is only valid as '<-'
    case '<':
      if (match('-'))
        addToken(sirius_graph_query_token_type::ARROW_LEFT);
      else
        lexError("Unexpected character '<' (did you mean '<-'?)");
      break;

    // '-' could be  ->  ->*  ->+  or plain -
    case '-':
      if (match('>')) {
        if (match('*'))
          addToken(sirius_graph_query_token_type::ARROW_STAR);
        else if (match('+'))
          addToken(sirius_graph_query_token_type::ARROW_PLUS);
        else
          addToken(sirius_graph_query_token_type::ARROW_RIGHT);
      } else {
        addToken(sirius_graph_query_token_type::DASH);
      }
      break;

    // numbers & identifiers
    default:
      if (std::isdigit(static_cast<unsigned char>(c)))
        number();
      else if (std::isalpha(static_cast<unsigned char>(c)) || c == '_')
        identifier();
      else
        lexError(std::string("Unexpected character '") + c + "'");

      break;
  }
}

void sirius_graph_query_lexer::number()
{
  while (!atEnd() && std::isdigit(static_cast<unsigned char>(peek())))
    advance();
  addToken(sirius_graph_query_token_type::INTEGER);
}

void sirius_graph_query_lexer::identifier()
{
  while (!atEnd() && (std::isalnum(static_cast<unsigned char>(peek())) || peek() == '_'))
    advance();

  std::string text = source_.substr(start_, pos_ - start_);
  addToken(keywordType(text));
}

sirius_graph_query_token_type sirius_graph_query_lexer::keywordType(const std::string& text)
{
  std::string up = text;
  std::transform(up.begin(), up.end(), up.begin(), ::toupper);

  if (up == "MATCH") return sirius_graph_query_token_type::MATCH;
  if (up == "WHERE") return sirius_graph_query_token_type::WHERE;
  if (up == "COLUMNS") return sirius_graph_query_token_type::COLUMNS;
  if (up == "IN") return sirius_graph_query_token_type::IN;
  if (up == "ANY") return sirius_graph_query_token_type::ANY;
  if (up == "SHORTEST") return sirius_graph_query_token_type::SHORTEST;

  return sirius_graph_query_token_type::IDENTIFIER;
}

void sirius_graph_query_lexer::lexError(const std::string& msg) const
{
  throw std::runtime_error("GraphQuery lexer error at line " + std::to_string(line_) + ": " + msg);
}

}  // namespace duckdb
