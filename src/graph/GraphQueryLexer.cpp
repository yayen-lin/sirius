#include "graph/GraphQueryLexer.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace duckdb {

GraphQueryLexer::GraphQueryLexer(const std::string& source) : source_(source) {}

std::vector<GQToken> GraphQueryLexer::tokenize()
{
  while (!atEnd()) {
    start_ = pos_;
    scanToken();
  }
  tokens_.emplace_back(GQTokenType::END_OF_FILE, "", line_);
  return tokens_;
}

char GraphQueryLexer::advance() { return source_[pos_++]; }

char GraphQueryLexer::peek() const { return atEnd() ? '\0' : source_[pos_]; }

char GraphQueryLexer::peekNext() const
{ return (pos_ + 1 >= source_.size()) ? '\0' : source_[pos_ + 1]; }

bool GraphQueryLexer::atEnd() const { return pos_ >= source_.size(); }

bool GraphQueryLexer::match(char expected)
{
  if (atEnd() || source_[pos_] != expected) return false;
  pos_++;
  return true;
}

void GraphQueryLexer::addToken(GQTokenType type)
{ tokens_.emplace_back(type, source_.substr(start_, pos_ - start_), line_); }

void GraphQueryLexer::scanToken()
{
  char c = advance();
  switch (c) {
    case ' ':
    case '\r':
    case '\t': break;
    case '\n': line_++; break;
    case '(': addToken(GQTokenType::LEFT_PAREN); break;
    case ')': addToken(GQTokenType::RIGHT_PAREN); break;
    case '[': addToken(GQTokenType::LEFT_BRACKET); break;
    case ']': addToken(GQTokenType::RIGHT_BRACKET); break;
    case ',': addToken(GQTokenType::COMMA); break;
    case ':': addToken(GQTokenType::COLON); break;
    case '.': addToken(GQTokenType::DOT); break;
    case ';': addToken(GQTokenType::SEMICOLON); break;
    case '=': addToken(GQTokenType::EQUALS); break;
    case '*': addToken(GQTokenType::STAR); break;
    case '+': addToken(GQTokenType::PLUS); break;

    // '<' is only valid as '<-'
    case '<':
      if (match('-'))
        addToken(GQTokenType::ARROW_LEFT);
      else
        lexError("Unexpected character '<' (did you mean '<-'?)");
      break;

    // '-' could be  ->  ->*  ->+  or plain -
    case '-':
      if (match('>')) {
        if (match('*'))
          addToken(GQTokenType::ARROW_STAR);
        else if (match('+'))
          addToken(GQTokenType::ARROW_PLUS);
        else
          addToken(GQTokenType::ARROW_RIGHT);
      } else {
        addToken(GQTokenType::DASH);
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

void GraphQueryLexer::number()
{
  while (!atEnd() && std::isdigit(static_cast<unsigned char>(peek())))
    advance();
  addToken(GQTokenType::INTEGER);
}

void GraphQueryLexer::identifier()
{
  while (!atEnd() && (std::isalnum(static_cast<unsigned char>(peek())) || peek() == '_'))
    advance();

  std::string text = source_.substr(start_, pos_ - start_);
  addToken(keywordType(text));
}

GQTokenType GraphQueryLexer::keywordType(const std::string& text)
{
  std::string up = text;
  std::transform(up.begin(), up.end(), up.begin(), ::toupper);

  if (up == "MATCH") return GQTokenType::MATCH;
  if (up == "WHERE") return GQTokenType::WHERE;
  if (up == "COLUMNS") return GQTokenType::COLUMNS;
  if (up == "IN") return GQTokenType::IN;
  if (up == "ANY") return GQTokenType::ANY;
  if (up == "SHORTEST") return GQTokenType::SHORTEST;

  return GQTokenType::IDENTIFIER;
}

void GraphQueryLexer::lexError(const std::string& msg) const
{
  throw std::runtime_error("GraphQuery lexer error at line " + std::to_string(line_) + ": " + msg);
}

}  // namespace duckdb
