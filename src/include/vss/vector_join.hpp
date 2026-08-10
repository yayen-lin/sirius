/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sirius::vss {

/// Which pairs the join keeps.
enum class knn_join_mode : std::uint8_t {
  global_top_k,   ///< the k closest (probe, corpus) pairs overall
  per_row_top_k,  ///< the k closest corpus rows for each probe row
  threshold,      ///< every pair with distance <= distance_threshold
};

struct knn_join_side {
  std::string catalog;                      ///< resolved catalog of the pinned table
  std::string schema;                       ///< resolved schema
  std::string table;                        ///< GPU-pinned base table
  std::string column;                       ///< FLOAT[dim] vector column
  std::vector<std::string> output_columns;  ///< base-table columns to emit, in order
};

/// Parsed form of a sirius_knn_join call. Filled by the TVF bind, consumed by
/// the plan generator when it swaps the LogicalGet for the GPU join operator.
struct knn_join_request {
  knn_join_side probe;
  knn_join_side corpus;
  knn_join_mode mode{knn_join_mode::per_row_top_k};
  std::string metric{"l2"};        ///< 'l2' or 'cosine'
  bool use_index{false};           ///< search_mode 'approx' => true, 'exact' => false
  std::int64_t k{0};               ///< top-k; unused in threshold mode
  double distance_threshold{0.0};  ///< keep pairs with distance <= this
  bool has_threshold{false};       ///< threshold given (a prune filter in the top-k modes)
  std::int64_t n_probes{0};        ///< IVF lists to probe (approx only); 0 = index default
  std::int64_t dim{0};             ///< shared vector dimensionality
  bool self_join{false};           ///< both sides resolve to the same table
  bool self_exclude{false};        ///< drop the pair a row forms with itself (search k+1, drop i==j)
};

}  // namespace sirius::vss
