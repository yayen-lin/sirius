#include "graph/sirius_graph_query_parser.hpp"

#include "graph/sirius_graph_query_lexer.hpp"

#include <stdexcept>

namespace duckdb {

sirius_graph_query_parser::sirius_graph_query_parser(std::vector<sirius_graph_query_token> tokens) : tokens_(std::move(tokens)) {}

sirius_parsed_graph_query sirius_graph_query_parser::Parse(const std::string& query)
{
  sirius_graph_query_lexer lexer(query);
  auto tokens = lexer.tokenize();
  sirius_graph_query_parser parser(std::move(tokens));
  return parser.parse();
}

sirius_parsed_graph_query sirius_graph_query_parser::parse()
{
  sirius_parsed_graph_query result;
  parseMatchClause(result);
  parseColumnsClause(result);
  return result;
}

sirius_graph_query_token& sirius_graph_query_parser::peek() { return tokens_[pos_]; }

sirius_graph_query_token& sirius_graph_query_parser::previous() { return tokens_[pos_ - 1]; }

bool sirius_graph_query_parser::atEnd() const { return tokens_[pos_].type == sirius_graph_query_token_type::END_OF_FILE; }

sirius_graph_query_token sirius_graph_query_parser::advance()
{
  if (!atEnd()) pos_++;
  return tokens_[pos_ - 1];
}

bool sirius_graph_query_parser::check(sirius_graph_query_token_type type) const
{
  if (pos_ >= tokens_.size()) return false;
  return tokens_[pos_].type == type;
}

bool sirius_graph_query_parser::match(sirius_graph_query_token_type type)
{
  if (check(type)) {
    advance();
    return true;
  }
  return false;
}

sirius_graph_query_token sirius_graph_query_parser::consume(sirius_graph_query_token_type type, const std::string& msg)
{
  if (check(type)) return advance();
  parseError(msg);
}

sirius_graph_query_token& sirius_graph_query_parser::peekAhead(size_t offset)
{
  size_t idx = pos_ + offset;
  if (idx >= tokens_.size()) return tokens_.back();  // EOF
  return tokens_[idx];
}

// match_clause → [ANY SHORTEST | SHORTEST] MATCH vertex_pat edge_pat vertex_pat
void sirius_graph_query_parser::parseMatchClause(sirius_parsed_graph_query& q)
{
  if (match(sirius_graph_query_token_type::ANY)) {
    consume(sirius_graph_query_token_type::SHORTEST, "Expected 'SHORTEST' after 'ANY'");
    q.op = sirius_operation_type::UNWEIGHTED_SHORTEST_PATH;
  } else if (match(sirius_graph_query_token_type::SHORTEST)) {
    q.op = sirius_operation_type::WEIGHTED_SHORTEST_PATH;
  }

  consume(sirius_graph_query_token_type::MATCH, "Expected 'MATCH'");

  // source vertex pattern
  consume(sirius_graph_query_token_type::LEFT_PAREN, "Expected '(' for source vertex pattern");
  parseVertexPattern(q, true);
  consume(sirius_graph_query_token_type::RIGHT_PAREN, "Expected ')' to close source vertex pattern");

  // edge pattern
  parseEdgePattern(q);

  // destination vertex pattern
  consume(sirius_graph_query_token_type::LEFT_PAREN, "Expected '(' for destination vertex pattern");
  parseVertexPattern(q, false);
  consume(sirius_graph_query_token_type::RIGHT_PAREN, "Expected ')' to close destination vertex pattern");

  // Infer BFS from path pattern (only if not already set by ANY SHORTEST / SHORTEST)
  if (q.op == sirius_operation_type::EDGE_TRAVERSAL) {
    if (q.path_pattern == sirius_path_pattern::ONE_OR_MORE || q.path_pattern == sirius_path_pattern::ZERO_OR_MORE) {
      q.op = sirius_operation_type::BFS;
    }
  }
}

// vertex_pat → alias : table [WHERE alias . col filter]
void sirius_graph_query_parser::parseVertexPattern(sirius_parsed_graph_query& q, const bool isSrc)
{
  const std::string alias = consume(sirius_graph_query_token_type::IDENTIFIER, "Expected vertex alias").lexeme;
  consume(sirius_graph_query_token_type::COLON, "Expected ':' after vertex alias");
  const std::string table = consume(sirius_graph_query_token_type::IDENTIFIER, "Expected vertex table name").lexeme;

  if (isSrc) {
    q.src_alias = alias;
    q.src_table = table;
  } else {
    q.dst_alias = alias;
    q.dst_table = table;
  }

  if (match(sirius_graph_query_token_type::WHERE)) { parseFilterExpr(q, isSrc); }
}

// filter → alias . col IN ( id_list | subquery )
//        | alias . col = integer
void sirius_graph_query_parser::parseFilterExpr(sirius_parsed_graph_query& q, bool isSrc)
{
  // Consume  alias.col  (e.g. src.id)
  consume(sirius_graph_query_token_type::IDENTIFIER, "Expected alias before '.'");
  consume(sirius_graph_query_token_type::DOT, "Expected '.'");
  consume(sirius_graph_query_token_type::IDENTIFIER, "Expected column name after '.'");

  if (match(sirius_graph_query_token_type::IN)) {
    consume(sirius_graph_query_token_type::LEFT_PAREN, "Expected '(' after IN");

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

    consume(sirius_graph_query_token_type::RIGHT_PAREN, "Expected ')' to close IN list");

  } else if (match(sirius_graph_query_token_type::EQUALS)) {
    sirius_graph_query_token num              = consume(sirius_graph_query_token_type::INTEGER, "Expected integer after '='");
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

void sirius_graph_query_parser::parseEdgeKeyMapping(sirius_parsed_graph_query& q)
{
  consume(sirius_graph_query_token_type::LEFT_PAREN, "Expected '(' for edge key mapping");

  do {
    std::string key = consume(sirius_graph_query_token_type::IDENTIFIER, "Expected key name").lexeme;
    consume(sirius_graph_query_token_type::EQUALS, "Expected '=' after key name");
    std::string val = consume(sirius_graph_query_token_type::IDENTIFIER, "Expected column name").lexeme;

    if (key == "src_col")
      q.edge_src_col = val;
    else if (key == "dst_col")
      q.edge_dst_col = val;
    else
      parseError("Unknown edge key mapping '" + key + "' (expected src_col or dst_col)");

  } while (match(sirius_graph_query_token_type::COMMA));

  consume(sirius_graph_query_token_type::RIGHT_PAREN, "Expected ')' to close edge key mapping");
}

// edge_pat → [-| <-] [ [alias:] edge_table ] (-> | ->* | ->+)
void sirius_graph_query_parser::parseEdgePattern(sirius_parsed_graph_query& q)
{
  bool leftArrow = false;

  if (match(sirius_graph_query_token_type::ARROW_LEFT)) {
    leftArrow = true;
  } else {
    consume(sirius_graph_query_token_type::DASH, "Expected '-' or '<-' to start edge pattern");
  }

  if (match(sirius_graph_query_token_type::LEFT_BRACKET)) {
    if (check(sirius_graph_query_token_type::IDENTIFIER) && peekAhead(1).type == sirius_graph_query_token_type::COLON) {
      q.edge_alias = advance().lexeme;
      advance();  // consume ':'
    } else {
      match(sirius_graph_query_token_type::COLON);
    }
    q.edge_table = consume(sirius_graph_query_token_type::IDENTIFIER, "Expected edge table name").lexeme;

    // optional (src_col=..., dst_col=...)
    if (check(sirius_graph_query_token_type::LEFT_PAREN)) { parseEdgeKeyMapping(q); }

    consume(sirius_graph_query_token_type::RIGHT_BRACKET, "Expected ']' after edge pattern");
  }

  if (match(sirius_graph_query_token_type::ARROW_STAR)) {
    q.path_pattern = sirius_path_pattern::ZERO_OR_MORE;
    q.direction    = leftArrow ? sirius_edge_direction::BOTH : sirius_edge_direction::RIGHT;
  } else if (match(sirius_graph_query_token_type::ARROW_PLUS)) {
    q.path_pattern = sirius_path_pattern::ONE_OR_MORE;
    q.direction    = leftArrow ? sirius_edge_direction::BOTH : sirius_edge_direction::RIGHT;
  } else if (match(sirius_graph_query_token_type::ARROW_RIGHT)) {
    q.path_pattern = sirius_path_pattern::DIRECT;
    q.direction    = leftArrow ? sirius_edge_direction::BOTH : sirius_edge_direction::RIGHT;
  } else if (leftArrow && match(sirius_graph_query_token_type::DASH)) {
    q.path_pattern = sirius_path_pattern::DIRECT;
    q.direction    = sirius_edge_direction::LEFT;
  } else {
    parseError("Expected edge direction arrow (->, ->*, ->+)");
  }
}

// columns_clause → COLUMNS ( col [, col]* )
void sirius_graph_query_parser::parseColumnsClause(sirius_parsed_graph_query& q)
{
  consume(sirius_graph_query_token_type::COLUMNS, "Expected 'COLUMNS'");
  consume(sirius_graph_query_token_type::LEFT_PAREN, "Expected '(' after COLUMNS");

  do {
    std::string col = consume(sirius_graph_query_token_type::IDENTIFIER, "Expected column name").lexeme;
    if (match(sirius_graph_query_token_type::DOT)) {
      col += "." + consume(sirius_graph_query_token_type::IDENTIFIER, "Expected field name after '.'").lexeme;
    }

    // flags for special column names
    if (col == "distance") q.return_distance = true;
    if (col == "predecessor") q.return_predecessor = true;
    if (col == "path") q.reconstruct_path = true;

    q.output_columns.push_back(col);
  } while (match(sirius_graph_query_token_type::COMMA));

  consume(sirius_graph_query_token_type::RIGHT_PAREN, "Expected ')' to close COLUMNS");
}

// Returns true when the current token looks like the start of a subquery
// (i.e. not an integer and not an immediate closing paren)
bool sirius_graph_query_parser::isSubqueryStart()
{
  if (atEnd()) return false;
  sirius_graph_query_token_type t = peek().type;
  if (t == sirius_graph_query_token_type::RIGHT_PAREN) return false;
  if (t == sirius_graph_query_token_type::INTEGER) return false;
  return true;
}

// Consume everything up to (but not including) the matching closing ')'
// The opening '(' has already been consumed.
std::string sirius_graph_query_parser::consumeSubquery()
{
  std::string subq;
  int depth = 1;
  while (!atEnd() && depth > 0) {
    sirius_graph_query_token tok = advance();
    if (tok.type == sirius_graph_query_token_type::LEFT_PAREN) depth++;
    if (tok.type == sirius_graph_query_token_type::RIGHT_PAREN) {
      depth--;
      if (depth == 0) {
        pos_--;
        break;
      }
    }
    if (!subq.empty()) subq += ' ';
    subq += tok.lexeme;
  }
  return subq;
}

// Parse comma-separated list of integer literals
// The opening '(' has already been consumed; closing ')' is left for the caller.
std::vector<int64_t> sirius_graph_query_parser::parseLiteralIDList()
{
  std::vector<int64_t> ids;
  do {
    sirius_graph_query_token num = consume(sirius_graph_query_token_type::INTEGER, "Expected integer in ID list");
    ids.push_back(std::stoll(num.lexeme));
  } while (match(sirius_graph_query_token_type::COMMA));
  return ids;
}

void sirius_graph_query_parser::parseError(const std::string& msg)
{
  throw std::runtime_error("GraphQuery parser error at line " + std::to_string(peek().line) + ": " +
                           msg + " (got '" + peek().lexeme + "')");
}

}  // namespace duckdb
