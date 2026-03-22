#include "graph/GraphQueryParser.hpp"

#include "graph/GraphQueryLexer.hpp"

#include <stdexcept>

namespace duckdb {

GraphQueryParser::GraphQueryParser(std::vector<GQToken> tokens) : tokens_(std::move(tokens)) {}

ParsedGraphQuery GraphQueryParser::Parse(const std::string& query)
{
  GraphQueryLexer lexer(query);
  auto tokens = lexer.tokenize();
  GraphQueryParser parser(std::move(tokens));
  return parser.parse();
}

ParsedGraphQuery GraphQueryParser::parse()
{
  ParsedGraphQuery result;
  parseMatchClause(result);
  parseColumnsClause(result);
  return result;
}

GQToken& GraphQueryParser::peek() { return tokens_[pos_]; }

GQToken& GraphQueryParser::previous() { return tokens_[pos_ - 1]; }

bool GraphQueryParser::atEnd() const { return tokens_[pos_].type == GQTokenType::END_OF_FILE; }

GQToken GraphQueryParser::advance()
{
  if (!atEnd()) pos_++;
  return tokens_[pos_ - 1];
}

bool GraphQueryParser::check(GQTokenType type) const
{
  if (pos_ >= tokens_.size()) return false;
  return tokens_[pos_].type == type;
}

bool GraphQueryParser::match(GQTokenType type)
{
  if (check(type)) {
    advance();
    return true;
  }
  return false;
}

GQToken GraphQueryParser::consume(GQTokenType type, const std::string& msg)
{
  if (check(type)) return advance();
  parseError(msg);
}

GQToken& GraphQueryParser::peekAhead(size_t offset)
{
  size_t idx = pos_ + offset;
  if (idx >= tokens_.size()) return tokens_.back();  // EOF
  return tokens_[idx];
}

// match_clause → [ANY SHORTEST | SHORTEST] MATCH vertex_pat edge_pat vertex_pat
void GraphQueryParser::parseMatchClause(ParsedGraphQuery& q)
{
  if (match(GQTokenType::ANY)) {
    consume(GQTokenType::SHORTEST, "Expected 'SHORTEST' after 'ANY'");
    q.op = OperationType::UNWEIGHTED_SHORTEST_PATH;
  } else if (match(GQTokenType::SHORTEST)) {
    q.op = OperationType::WEIGHTED_SHORTEST_PATH;
  }

  consume(GQTokenType::MATCH, "Expected 'MATCH'");

  // Source vertex pattern
  consume(GQTokenType::LEFT_PAREN, "Expected '(' for source vertex pattern");
  parseVertexPattern(q, /*isSrc=*/true);
  consume(GQTokenType::RIGHT_PAREN, "Expected ')' to close source vertex pattern");

  // Edge pattern
  parseEdgePattern(q);

  // Destination vertex pattern
  consume(GQTokenType::LEFT_PAREN, "Expected '(' for destination vertex pattern");
  parseVertexPattern(q, /*isSrc=*/false);
  consume(GQTokenType::RIGHT_PAREN, "Expected ')' to close destination vertex pattern");

  // Infer BFS from path pattern (only if not already set by ANY SHORTEST / SHORTEST)
  if (q.op == OperationType::EDGE_TRAVERSAL) {
    if (q.path_pattern == PathPattern::ONE_OR_MORE || q.path_pattern == PathPattern::ZERO_OR_MORE) {
      q.op = OperationType::BFS;
    }
  }
}

// vertex_pat → alias : table [WHERE alias . col filter]
void GraphQueryParser::parseVertexPattern(ParsedGraphQuery& q, const bool isSrc)
{
  const std::string alias = consume(GQTokenType::IDENTIFIER, "Expected vertex alias").lexeme;
  consume(GQTokenType::COLON, "Expected ':' after vertex alias");
  const std::string table = consume(GQTokenType::IDENTIFIER, "Expected vertex table name").lexeme;

  if (isSrc) {
    q.src_alias = alias;
    q.src_table = table;
  } else {
    q.dst_alias = alias;
    q.dst_table = table;
  }

  if (match(GQTokenType::WHERE)) { parseFilterExpr(q, isSrc); }
}

// filter → alias . col IN ( id_list | subquery )
//        | alias . col = integer
void GraphQueryParser::parseFilterExpr(ParsedGraphQuery& q, bool isSrc)
{
  // Consume  alias.col  (e.g. src.id)
  consume(GQTokenType::IDENTIFIER, "Expected alias before '.'");
  consume(GQTokenType::DOT, "Expected '.'");
  consume(GQTokenType::IDENTIFIER, "Expected column name after '.'");

  if (match(GQTokenType::IN)) {
    consume(GQTokenType::LEFT_PAREN, "Expected '(' after IN");

    if (isSubqueryStart()) {
      std::string subq = consumeSubquery();
      if (isSrc) {
        q.source_filter     = subq;
        q.has_source_filter = true;
      } else {
        q.dest_filter     = subq;
        q.has_dest_filter = true;
      }
    } else {
      std::vector<int64_t> ids = parseLiteralIDList();
      if (isSrc) {
        q.source_filter     = ids;
        q.has_source_filter = true;
      } else {
        q.dest_filter     = ids;
        q.has_dest_filter = true;
      }
    }

    consume(GQTokenType::RIGHT_PAREN, "Expected ')' to close IN list");

  } else if (match(GQTokenType::EQUALS)) {
    GQToken num              = consume(GQTokenType::INTEGER, "Expected integer after '='");
    std::vector<int64_t> ids = {std::stoll(num.lexeme)};
    if (isSrc) {
      q.source_filter     = ids;
      q.has_source_filter = true;
    } else {
      q.dest_filter     = ids;
      q.has_dest_filter = true;
    }

  } else {
    parseError("Expected 'IN' or '=' in WHERE filter");
  }
}

void GraphQueryParser::parseEdgeKeyMapping(ParsedGraphQuery& q)
{
  consume(GQTokenType::LEFT_PAREN, "Expected '(' for edge key mapping");

  do {
    std::string key = consume(GQTokenType::IDENTIFIER, "Expected key name").lexeme;
    consume(GQTokenType::EQUALS, "Expected '=' after key name");
    std::string val = consume(GQTokenType::IDENTIFIER, "Expected column name").lexeme;

    if (key == "src_col")
      q.edge_src_col = val;
    else if (key == "dst_col")
      q.edge_dst_col = val;
    else
      parseError("Unknown edge key mapping '" + key + "' (expected src_col or dst_col)");

  } while (match(GQTokenType::COMMA));

  consume(GQTokenType::RIGHT_PAREN, "Expected ')' to close edge key mapping");
}

// edge_pat → [-| <-] [ [alias:] edge_table ] (-> | ->* | ->+)
void GraphQueryParser::parseEdgePattern(ParsedGraphQuery& q)
{
  bool leftArrow = false;

  if (match(GQTokenType::ARROW_LEFT)) {
    leftArrow = true;
  } else {
    consume(GQTokenType::DASH, "Expected '-' or '<-' to start edge pattern");
  }

  if (match(GQTokenType::LEFT_BRACKET)) {
    if (check(GQTokenType::IDENTIFIER) && peekAhead(1).type == GQTokenType::COLON) {
      q.edge_alias = advance().lexeme;
      advance();  // consume ':'
    } else {
      match(GQTokenType::COLON);
    }
    q.edge_table = consume(GQTokenType::IDENTIFIER, "Expected edge table name").lexeme;

    // optional (src_col=..., dst_col=...)
    if (check(GQTokenType::LEFT_PAREN)) { parseEdgeKeyMapping(q); }

    consume(GQTokenType::RIGHT_BRACKET, "Expected ']' after edge pattern");
  }

  if (match(GQTokenType::ARROW_STAR)) {
    q.path_pattern = PathPattern::ZERO_OR_MORE;
    q.direction    = leftArrow ? EdgeDirection::BOTH : EdgeDirection::RIGHT;
  } else if (match(GQTokenType::ARROW_PLUS)) {
    q.path_pattern = PathPattern::ONE_OR_MORE;
    q.direction    = leftArrow ? EdgeDirection::BOTH : EdgeDirection::RIGHT;
  } else if (match(GQTokenType::ARROW_RIGHT)) {
    q.path_pattern = PathPattern::DIRECT;
    q.direction    = leftArrow ? EdgeDirection::BOTH : EdgeDirection::RIGHT;
  } else if (leftArrow && match(GQTokenType::DASH)) {
    q.path_pattern = PathPattern::DIRECT;
    q.direction    = EdgeDirection::LEFT;
  } else {
    parseError("Expected edge direction arrow (->, ->*, ->+)");
  }
}

// columns_clause → COLUMNS ( col [, col]* )
void GraphQueryParser::parseColumnsClause(ParsedGraphQuery& q)
{
  consume(GQTokenType::COLUMNS, "Expected 'COLUMNS'");
  consume(GQTokenType::LEFT_PAREN, "Expected '(' after COLUMNS");

  do {
    std::string col = consume(GQTokenType::IDENTIFIER, "Expected column name").lexeme;
    if (match(GQTokenType::DOT)) {
      col += "." + consume(GQTokenType::IDENTIFIER, "Expected field name after '.'").lexeme;
    }

    // set flags for special column names
    if (col == "distance") q.return_distance = true;
    if (col == "predecessor") q.return_predecessor = true;
    if (col == "path") q.reconstruct_path = true;

    q.output_columns.push_back(col);
  } while (match(GQTokenType::COMMA));

  consume(GQTokenType::RIGHT_PAREN, "Expected ')' to close COLUMNS");
}

// Returns true when the current token looks like the start of a subquery
// (i.e. not an integer and not an immediate closing paren)
bool GraphQueryParser::isSubqueryStart()
{
  if (atEnd()) return false;
  GQTokenType t = peek().type;
  if (t == GQTokenType::RIGHT_PAREN) return false;
  if (t == GQTokenType::INTEGER) return false;
  return true;
}

// Consume everything up to (but not including) the matching closing ')'
// The opening '(' has already been consumed.
std::string GraphQueryParser::consumeSubquery()
{
  std::string subq;
  int depth = 1;
  while (!atEnd() && depth > 0) {
    GQToken tok = advance();
    if (tok.type == GQTokenType::LEFT_PAREN) depth++;
    if (tok.type == GQTokenType::RIGHT_PAREN) {
      depth--;
      if (depth == 0) {
        pos_--;
        break;
      }  // leave ')' for consume() above
    }
    if (!subq.empty()) subq += ' ';
    subq += tok.lexeme;
  }
  return subq;
}

// Parse comma-separated list of integer literals
// The opening '(' has already been consumed; closing ')' is left for the caller.
std::vector<int64_t> GraphQueryParser::parseLiteralIDList()
{
  std::vector<int64_t> ids;
  do {
    GQToken num = consume(GQTokenType::INTEGER, "Expected integer in ID list");
    ids.push_back(std::stoll(num.lexeme));
  } while (match(GQTokenType::COMMA));
  return ids;
}

void GraphQueryParser::parseError(const std::string& msg)
{
  throw std::runtime_error("GraphQuery parser error at line " + std::to_string(peek().line) + ": " +
                           msg + " (got '" + peek().lexeme + "')");
}

}  // namespace duckdb
