//
// Copyright © tuanhe. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CustomSubgraphViewConverter.hpp"
#include "CustomPreCompiledObject.hpp"

namespace armnn
{

uint32_t CustomSubgraphViewConverter::ms_NextInstanceId = 0;

CustomSubgraphViewConverter::CustomSubgraphViewConverter(const SubgraphView& subgraph, 
                                                        ModelOptions modelOptions)
    : m_InstanceId(ms_NextInstanceId++)
    , m_Subgraph(subgraph)
{
    armnn::IgnoreUnused(modelOptions);
    ConstructCustomNetwork();
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

    Layer& inputLayer = connectedSlot->GetOwningLayer();
    ARMNN_LOG(info) << "Graph input layer : " << inputLayer.GetNameStr();
    
    if(!AddCustomLayer(inputLayer))
    {
        throw armnn::BackendUnavailableException("Convert input layer failed");
    }


    // Add input to the Ethos-N network
    //ethosn_lib::TensorInfo CustomTensorInfo = BuildCustomTensorInfo(connectedSlot->GetTensorInfo(), DataLayout::NHWC);
    //CustomAddOperationResult inputOperandAndId = ethosn_lib::AddInput(m_Network, CustomTensorInfo);

}

void CustomSubgraphViewConverter::AddOutput(uint32_t outputSlotIdx)
{
    const OutputSlot& outputSlot = *m_Subgraph.GetOutputSlot(outputSlotIdx);
    //Layer* outputLayer = &outputSlot.GetOwningLayer();
    Layer& outputLayer = outputSlot.GetOwningLayer();
    ARMNN_LOG(info) << "Graph output layer : " << outputLayer.GetNameStr();

    if(!RecusedLayer(outputLayer))
    {
        throw armnn::BackendUnavailableException("Convert layer failed");
    }

    // Get the Ethos-N operand that should connect to this output
    //auto input = AddOrRetrieveCustomOperand(&outputSlot);

    // Add output operand to Ethos-N network
    //ethosn_lib::TensorAndId<ethosn_lib::Output> output = ethosn_lib::AddOutput(m_Network, *input.tensor);

}

bool CustomSubgraphViewConverter::RecusedLayer(Layer& layer)
{
    for(uint32_t i = 0; i < layer.GetNumInputSlots(); ++i)
    {
        auto& prevLayer = layer.GetInputSlot(i).GetConnectedOutputSlot()->GetOwningLayer();
        if(std::find(m_ConvertedLayer.begin(), m_ConvertedLayer.end(), prevLayer.GetGuid()) == m_ConvertedLayer.end())
            if(!RecusedLayer(prevLayer))
                return false;
    }

    //previous layer had convetered
    //check the current layer
    if(std::find(m_ConvertedLayer.begin(), m_ConvertedLayer.end(), layer.GetGuid()) == m_ConvertedLayer.end())
        if(!AddCustomLayer(layer))
            return false;

    return true; 
}

bool CustomSubgraphViewConverter::AddCustomLayer(Layer& layer)
{
    if(std::find(m_ConvertedLayer.begin(), m_ConvertedLayer.end(), layer.GetGuid()) != m_ConvertedLayer.end())
        return true;
    
    LayerFunctionPtr layerFunction = CustomLayerBridge::GetBridge().GetLayerFunction(layer.GetType());
    
    if( nullptr == layerFunction)
    {
        std::stringstream errorMessage;
        errorMessage << "Layer[" << GetLayerTypeAsCString(layer.GetType()) 
                     << "] had not been registed into the backend";
        throw armnn::UnimplementedException(errorMessage.str());
    }

    if(!layerFunction(layer, m_CustomDefined))
    {
        ARMNN_LOG(fatal) << "Layer " << layer.GetNameStr() << " convert failed";
        return false;
    }

    m_ConvertedLayer.push_back(layer.GetGuid());
    return true; 
}

bool CustomSubgraphViewConverter::ConstructCustomNetwork()
{
    // Add inputs
    for (uint32_t inputSlotIdx = 0; inputSlotIdx < m_Subgraph.GetNumInputSlots(); ++inputSlotIdx)
    {
        AddInput(inputSlotIdx);
    }

    // Add outputs.
    for (uint32_t outputSlotIdx = 0; outputSlotIdx < m_Subgraph.GetNumOutputSlots(); ++outputSlotIdx)
    {
        AddOutput(outputSlotIdx);
    }

    return true;
}

namespace
{
template <typename T>
void DeleteAsType(const void* const blob)
{
    delete static_cast<const T*>(blob);
}
}    // namespace

std::vector<CompiledBlobPtr> CustomSubgraphViewConverter::CompileNetwork()
{

    auto preCompiledObject = std::make_unique<CustomPreCompiledObject>();

    preCompiledObject->SetDebugName("Debug Name");
    preCompiledObject->PreInferenceStage(m_CustomDefined);

    std::vector<CompiledBlobPtr> compiledBlobs;
    // Convert the EthosNPreCompiledObject into a "blob" (void) object and attach the custom blob deleter
    compiledBlobs.emplace_back(preCompiledObject.release(), DeleteAsType<CustomPreCompiledObject>);

    return compiledBlobs;
}

} // namespace armnn
