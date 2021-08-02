//
// Copyright © tuanhe. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CustomSubgraphViewConverter.hpp"

namespace armnn
{

uint32_t CustomSubgraphViewConverter::ms_NextInstanceId = 0;

CustomSubgraphViewConverter::CustomSubgraphViewConverter(const SubgraphView& subgraph, 
                                                        ModelOptions modelOptions)
    : m_InstanceId(ms_NextInstanceId++)
    , m_Subgraph(subgraph)
{
    armnn::IgnoreUnused(modelOptions);
}

void CustomSubgraphViewConverter::ResetNextInstanceId()
{
    ms_NextInstanceId = 0;
}

void CustomSubgraphViewConverter::AddInput(uint32_t inputSlotIdx)
{
    const InputSlot& inputSlot      = *m_Subgraph.GetInputSlot(inputSlotIdx);
    const OutputSlot* connectedSlot = inputSlot.GetConnectedOutputSlot();
    ARMNN_ASSERT(connectedSlot != nullptr);
    armnn::IgnoreUnused(connectedSlot);

    // Add input to the Ethos-N network
    //ethosn_lib::TensorInfo CustomTensorInfo = BuildCustomTensorInfo(connectedSlot->GetTensorInfo(), DataLayout::NHWC);
    //CustomAddOperationResult inputOperandAndId = ethosn_lib::AddInput(m_Network, CustomTensorInfo);

}

void CustomSubgraphViewConverter::AddOutput(uint32_t outputSlotIdx)
{
    const OutputSlot& outputSlot = *m_Subgraph.GetOutputSlot(outputSlotIdx);
    armnn::IgnoreUnused(outputSlot);

    // Get the Ethos-N operand that should connect to this output
    //auto input = AddOrRetrieveCustomOperand(&outputSlot);

    // Add output operand to Ethos-N network
    //ethosn_lib::TensorAndId<ethosn_lib::Output> output = ethosn_lib::AddOutput(m_Network, *input.tensor);

}

std::vector<CompiledBlobPtr> CustomSubgraphViewConverter::CompileNetwork()
{
    std::vector<CompiledBlobPtr> compiledBlobs;
    
    // Add inputs
    for (uint32_t inputSlotIdx = 0; inputSlotIdx < m_Subgraph.GetNumInputSlots(); ++inputSlotIdx)
    {
        AddInput(inputSlotIdx);
    }

    // Add outputs.
    // This will recurse through the graph converting layers until we end up connecting to the input operations
    // added to the Ethos-N graph above.
    for (uint32_t outputSlotIdx = 0; outputSlotIdx < m_Subgraph.GetNumOutputSlots(); ++outputSlotIdx)
    {
        AddOutput(outputSlotIdx);
    }
    return compiledBlobs;
}

} // namespace armnn
