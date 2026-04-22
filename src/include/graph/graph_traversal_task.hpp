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

#include "graph/CachedCSR.hpp"
#include "pipeline/gpu_pipeline_task.hpp"

#include <memory>

namespace sirius::graph {

/**
 * @brief Custom task for graph traversal operators that estimates memory based on CSR size.
 *
 * Graph traversal operators receive a trigger batch (0 rows) as input since they read from
 * a shared CSR pointer rather than consuming input batches. The default reservation logic
 * would estimate 0 bytes for such tasks. This custom task overrides memory estimation to
 * account for the actual CSR size and traversal output.
 */
class graph_traversal_task : public pipeline::gpu_pipeline_task {
 public:
  graph_traversal_task(uint64_t task_id,
                       std::vector<cucascade::shared_data_repository*> data_repos,
                       std::unique_ptr<pipeline::sirius_pipeline_task_local_state> local_state,
                       std::shared_ptr<pipeline::sirius_pipeline_task_global_state> global_state,
                       duckdb::shared_ptr<duckdb::CachedCSR> csr);

  /**
   * @brief Estimate memory reservation based on CSR size and expected output.
   *
   * The estimate accounts for:
   * - CSR arrays (offsets + indices) if not yet built
   * - Traversal output arrays (node_ids, distances, predecessors)
   * - Temporary buffers used during traversal
   *
   * @return Estimated peak memory in bytes
   */
  [[nodiscard]] std::size_t get_estimated_reservation_size() const override;

  std::unique_ptr<gpu_pipeline_task> create_rescheduled_task(
    uint64_t task_id,
    std::unique_ptr<pipeline::sirius_pipeline_task_local_state> local_state) override;

 private:
  duckdb::shared_ptr<duckdb::CachedCSR> _csr;
};

}  // namespace sirius::graph
