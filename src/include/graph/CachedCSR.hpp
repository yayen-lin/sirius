#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <algorithm>

namespace duckdb {

struct CachedCSR {
  // CSR arrays allocated in GPU cache
  int64_t* offsets  = nullptr; // size: num_vertices + 1
  int64_t* indices  = nullptr; // size: num_edges
  float*   weights  = nullptr; // size: num_edges, nullptr if unweighted

  int64_t  num_vertices = 0;
  int64_t  num_edges    = 0;
  bool     is_weighted  = false;

  // Cache key: "EDGE_TABLE:src_col:dst_col"
  static std::string MakeKey(const std::string& edge_table,
                              const std::string& src_col,
                              const std::string& dst_col)
  {
    std::string upper_table = edge_table;
    std::transform(upper_table.begin(), upper_table.end(), upper_table.begin(), ::toupper);
    return upper_table + ":" + src_col + ":" + dst_col;
  }
};

}  // namespace duckdb