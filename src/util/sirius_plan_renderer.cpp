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

#include "util/sirius_plan_renderer.hpp"

#include <sstream>

namespace sirius::util {

static void render_node(std::ostringstream& out,
                        const op::sirius_physical_operator& node,
                        const std::string& prefix,
                        bool is_last)
{
  out << prefix << (is_last ? "\u2514\u2500\u2500 " : "\u251C\u2500\u2500 ");
  out << node.to_string();
  out << "  (est. " << node.estimated_cardinality << ")\n";

  auto children   = node.get_children();
  auto new_prefix = prefix + (is_last ? "    " : "\u2502   ");
  for (size_t i = 0; i < children.size(); i++) {
    render_node(out, children[i].get(), new_prefix, i == children.size() - 1);
  }
}

std::string render_operator_tree(const op::sirius_physical_operator& root)
{
  std::ostringstream out;

  // Print root node without connector
  out << root.to_string();
  out << "  (est. " << root.estimated_cardinality << ")\n";

  auto children = root.get_children();
  for (size_t i = 0; i < children.size(); i++) {
    render_node(out, children[i].get(), "", i == children.size() - 1);
  }

  return out.str();
}

}  // namespace sirius::util
