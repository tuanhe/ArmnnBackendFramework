//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <functional>
#include <map>

#include <Layer.hpp>
#include <SubgraphView.hpp>


namespace armnn
{

class CustomSubgraphViewOptimizer
{
    public:
        SubgraphView OptimizeSubgraph(Graph&);
        Graph CloneGraph(const SubgraphView& originalSubgraphView);
        
    private:
        CustomSubgraphViewOptimizer()  = default;
        ~CustomSubgraphViewOptimizer() = default;
        SubgraphView ReinterpretGraphToSubgraph(Graph& graph);
        
    private:
};

} // namespace armnn
