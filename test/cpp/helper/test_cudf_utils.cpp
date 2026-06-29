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

// Tests for the cuDF type-mapping helpers in cudf/cudf_utils.hpp:
//   - sirius::get_cudf_type(const sirius::logical_type&)
//   - duckdb::GetCudfType(const duckdb::LogicalType&)
//
// These focus on the ARRAY (fixed-size list) mapping added alongside the
// ARRAY data type, both helpers must lower ARRAY to a cuDF LIST column.

#include "catch.hpp"
#include "cudf/cudf_utils.hpp"
#include "helper/logical_type.hpp"

#include <cudf/types.hpp>

#include <duckdb/common/optional_idx.hpp>
#include <duckdb/common/types.hpp>

using sirius::logical_type;
using sirius::type_id;

// ============================================================================
// sirius::get_cudf_type — ARRAY mapping
// ============================================================================

TEST_CASE("get_cudf_type - ARRAY maps to cuDF LIST", "[cudf_utils]")
{
  auto arr = logical_type::make_array(logical_type::make(type_id::INTEGER), 3);
  REQUIRE(sirius::get_cudf_type(arr).id() == cudf::type_id::LIST);
}

TEST_CASE("get_cudf_type - ARRAY mapping ignores child type and size", "[cudf_utils]")
{
  // The cuDF type is always LIST regardless of element type or fixed size;
  // the element type is carried separately on the child column
  REQUIRE(
    sirius::get_cudf_type(logical_type::make_array(logical_type::make(type_id::DOUBLE), 1)).id() ==
    cudf::type_id::LIST);
  REQUIRE(
    sirius::get_cudf_type(logical_type::make_array(logical_type::make(type_id::BIGINT), 0)).id() ==
    cudf::type_id::LIST);
  // Nested ARRAY of ARRAYs still maps to a single top-level LIST
  auto nested =
    logical_type::make_array(logical_type::make_array(logical_type::make(type_id::INTEGER), 2), 4);
  REQUIRE(sirius::get_cudf_type(nested).id() == cudf::type_id::LIST);
}

// ============================================================================
// duckdb::GetCudfType — ARRAY mapping
// ============================================================================

TEST_CASE("GetCudfType - duckdb ARRAY maps to cuDF LIST", "[cudf_utils]")
{
  using duckdb::LogicalType;

  auto arr = LogicalType::ARRAY(LogicalType::INTEGER, duckdb::optional_idx(3));
  REQUIRE(duckdb::GetCudfType(arr).id() == cudf::type_id::LIST);
}

TEST_CASE("GetCudfType - duckdb ARRAY mapping is independent of child and size", "[cudf_utils]")
{
  using duckdb::LogicalType;

  REQUIRE(
    duckdb::GetCudfType(LogicalType::ARRAY(LogicalType::DOUBLE, duckdb::optional_idx(1))).id() ==
    cudf::type_id::LIST);
  REQUIRE(
    duckdb::GetCudfType(LogicalType::ARRAY(LogicalType::BIGINT, duckdb::optional_idx(16))).id() ==
    cudf::type_id::LIST);
}
