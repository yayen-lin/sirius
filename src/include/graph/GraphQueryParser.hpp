#pragma once

#include "GQToken.hpp"
#include "ParsedGraphQuery.hpp"

#include <string>
#include <vector>

namespace duckdb {

// ============================================================
//  GraphQueryParser (recursive descent parser)
//
//  Grammar:
//    graph_query    → match_clause columns_clause
//
//    match_clause   → [ANY SHORTEST | SHORTEST]
//                     MATCH vertex_pat edge_pat vertex_pat
//    vertex_pat     → ( alias : table [WHERE alias.col filter] )
//    filter         → IN ( id_list | subquery ) | = integer
//    edge_pat       → [-| <-] [ [alias:] edge_table [(src_col=col, dst_col=col)] ] (-> | ->* | ->+)
//
//    columns_clause → COLUMNS ( col [, col]* )
//                     special col names: distance, predecessor, path
// ============================================================
class GraphQueryParser {
 public:
  explicit GraphQueryParser(std::vector<GQToken> tokens);
  static ParsedGraphQuery Parse(const std::string& query);
  ParsedGraphQuery parse();

 private:
  std::vector<GQToken> tokens_;
  size_t pos_ = 0;

  // token navigation
  GQToken& peek();
  GQToken& previous();
  bool atEnd() const;
  GQToken advance();
  bool check(GQTokenType type) const;
  bool match(GQTokenType type);
  GQToken consume(GQTokenType type, const std::string& msg);
  GQToken& peekAhead(size_t offset);

  // grammar
  void parseMatchClause(ParsedGraphQuery& q);
  void parseVertexPattern(ParsedGraphQuery& q, bool isSrc);
  void parseFilterExpr(ParsedGraphQuery& q, bool isSrc);
  void parseEdgePattern(ParsedGraphQuery& q);
  void parseEdgeKeyMapping(ParsedGraphQuery& q);
  void parseColumnsClause(ParsedGraphQuery& q);

  // helpers
  bool isSubqueryStart();
  std::string consumeSubquery();
  std::vector<int64_t> parseLiteralIDList();
  [[noreturn]] void parseError(const std::string& msg);
};

}  // namespace duckdb