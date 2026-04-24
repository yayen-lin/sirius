#pragma once

#include "sirius_graph_query_token.hpp"
#include "sirius_parsed_graph_query.hpp"

#include <string>
#include <vector>

namespace duckdb {

// sirius_graph_query_parser (recursive descent parser)
//
// Grammar:
//   graph_query    → match_clause columns_clause
//
//   match_clause   → [ANY SHORTEST | SHORTEST]
//                    MATCH vertex_pat edge_pat vertex_pat
//   vertex_pat     → ( alias : table [WHERE alias.col filter] )
//   filter         → IN ( id_list | subquery ) | = integer
//   edge_pat       → [-| <-] [ [alias:] edge_table [(src_col=col, dst_col=col)] ] (-> | ->* | ->+)
//
//   columns_clause → COLUMNS ( col [, col]* )
//                    special col names: distance, predecessor, path
class sirius_graph_query_parser {
 public:
  explicit sirius_graph_query_parser(std::vector<sirius_graph_query_token> tokens);
  static sirius_parsed_graph_query Parse(const std::string& query);
  sirius_parsed_graph_query parse();

 private:
  std::vector<sirius_graph_query_token> tokens_;
  size_t pos_ = 0;

  // token navigation
  sirius_graph_query_token& peek();
  sirius_graph_query_token& previous();
  [[nodiscard]] bool atEnd() const;
  sirius_graph_query_token advance();
  [[nodiscard]] bool check(sirius_graph_query_token_type type) const;
  bool match(sirius_graph_query_token_type type);
  sirius_graph_query_token consume(sirius_graph_query_token_type type, const std::string& msg);
  sirius_graph_query_token& peekAhead(size_t offset);

  // grammar
  void parseMatchClause(sirius_parsed_graph_query& q);
  void parseVertexPattern(sirius_parsed_graph_query& q, bool isSrc);
  void parseFilterExpr(sirius_parsed_graph_query& q, bool isSrc);
  void parseEdgePattern(sirius_parsed_graph_query& q);
  void parseEdgeKeyMapping(sirius_parsed_graph_query& q);
  void parseColumnsClause(sirius_parsed_graph_query& q);

  // helpers
  bool isSubqueryStart();
  std::string consumeSubquery();
  std::vector<int64_t> parseLiteralIDList();
  [[noreturn]] void parseError(const std::string& msg);
};

}  // namespace duckdb