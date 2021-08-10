//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <functional>
#include <map>

#include <Layer.hpp>
#include <SubgraphView.hpp>
#include <armnn/BackendOptions.hpp>
#include "ISubgraphViewConverter.hpp"
#include "CustomLayerBridge.hpp"

namespace armnn
{

class CustomSubgraphViewConverter : public ISubgraphViewConverter
{
    public:
        CustomSubgraphViewConverter(const SubgraphView& subgraph, ModelOptions modelOptions);
        ~CustomSubgraphViewConverter() = default;

        std::vector<CompiledBlobPtr> CompileNetwork() override;

        static void ResetNextInstanceId();

    private:
        void AddInput(uint32_t inputSlotIdx);
        void AddOutput(uint32_t outputSlotIdx);
        bool AddCustomLayer(Layer& layer);
        bool RecusedLayer(Layer& layer);
        bool ConstructCustomNetwork();
        
    private:
        // ID number for next constructed instance
        static uint32_t ms_NextInstanceId;

        const uint32_t m_InstanceId;

        // Original Arm NN sub-graph
        const SubgraphView& m_Subgraph;
        // converted layers
        std::vector<LayerGuid> m_ConvertedLayer;
        //custom backend variable info
        void* m_CustomDefined;
};

} // namespace armnn
