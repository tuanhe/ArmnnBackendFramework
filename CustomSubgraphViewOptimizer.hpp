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
        CustomSubgraphViewOptimizer()  = default;
        ~CustomSubgraphViewOptimizer() = default;
        
        SubgraphView OptimizeSubgraph(Graph&);
        Graph CloneGraph(const SubgraphView& originalSubgraphView);
        
    private:
        SubgraphView ReinterpretGraphToSubgraph(Graph& graph);
        
    private:
};

} // namespace armnn
