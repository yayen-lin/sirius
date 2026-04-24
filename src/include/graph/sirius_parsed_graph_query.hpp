#pragma once

#include <string>
#include <variant>
#include <vector>

namespace duckdb {

// source filter right now can either be a list of IDs or a subquery
using sirius_source_filter = std::variant<std::vector<int64_t>, std::string>;

enum class sirius_edge_direction { RIGHT, LEFT, BOTH };
enum class sirius_path_pattern {
  DIRECT,       // -[]->
  ONE_OR_MORE,  // ->+
  ZERO_OR_MORE  // ->*
};
enum class sirius_operation_type { EDGE_TRAVERSAL, BFS, UNWEIGHTED_SHORTEST_PATH, WEIGHTED_SHORTEST_PATH };

struct sirius_parsed_graph_query {
  // vertex tables
  std::string src_table;
  std::string src_alias;
  std::string dst_table;
  std::string dst_alias;

  // edge
  std::string edge_table;
  std::string edge_alias;
  std::string edge_src_col;
  std::string edge_dst_col;

  // source filter
  sirius_source_filter source_filter;
  bool has_source_filter = false;

  // destination filter
  sirius_source_filter dest_filter;
  bool has_dest_filter = false;

  // graph operation
  sirius_operation_type op         = sirius_operation_type::EDGE_TRAVERSAL;
  sirius_edge_direction direction  = sirius_edge_direction::RIGHT;
  sirius_path_pattern path_pattern = sirius_path_pattern::DIRECT;

  // result
  std::vector<std::string> output_columns;
  bool return_distance    = false;
  bool return_predecessor = false;
  bool reconstruct_path   = false;

  // helpers
  [[nodiscard]] bool sourceIsLiteralList() const
  { return has_source_filter && std::holds_alternative<std::vector<int64_t>>(source_filter); }

  [[nodiscard]] const std::vector<int64_t>& sourceLiteralIDs() const
  { return std::get<std::vector<int64_t>>(source_filter); }

  [[nodiscard]] const std::string& sourceSubquery() const
  { return std::get<std::string>(source_filter); }
};

}  // namespace duckdb
