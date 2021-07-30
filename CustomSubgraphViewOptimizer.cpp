//
// Copyright © tuanhe. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CustomSubgraphViewOptimizer.hpp"

namespace armnn
{

Graph CustomSubgraphViewOptimizer::CloneGraph(const SubgraphView& originalSubgraph)
{
    Graph newGraph = Graph();

    std::unordered_map<const Layer*, Layer*> originalToClonedLayerMap;
    std::list<armnn::Layer*> originalSubgraphLayers = originalSubgraph.GetLayers();

    for (auto&& originalLayer : originalSubgraphLayers)
    {
        Layer* const layer = originalLayer->Clone(newGraph);
        originalToClonedLayerMap.emplace(originalLayer, layer);
    }

    LayerBindingId slotCount = 0;

    // SubstituteSubgraph() currently cannot be called on a Graph that contains only one layer.
    // CloneGraph() and ReinterpretGraphToSubgraph() are used to work around this.

    // creating new layers for the input slots, adding them to the new graph and connecting them
    for (auto originalSubgraphInputSlot : originalSubgraph.GetInputSlots())
    {
        Layer& originalSubgraphLayer = originalSubgraphInputSlot->GetOwningLayer();
        Layer* const clonedLayer     = originalToClonedLayerMap[&originalSubgraphLayer];

        const std::string& originalLayerName =
            originalSubgraphInputSlot->GetConnectedOutputSlot()->GetOwningLayer().GetNameStr();

        // add it as an input layer into the new graph
        InputLayer* const newInputLayer = newGraph.AddLayer<InputLayer>(slotCount, originalLayerName.c_str());
        InputSlot& clonedLayerIS        = clonedLayer->GetInputSlot(originalSubgraphInputSlot->GetSlotIndex());
        newInputLayer->GetOutputSlot(0).Connect(clonedLayerIS);
        newInputLayer->GetOutputSlot(0).SetTensorInfo(
            originalSubgraphInputSlot->GetConnectedOutputSlot()->GetTensorInfo());

        ++slotCount;
    }

    std::list<Layer*>::iterator it;
    for (it = originalSubgraphLayers.begin(); it != originalSubgraphLayers.end(); ++it)
    {
        Layer* originalSubgraphLayer = *it;
        Layer* const clonedLayer     = originalToClonedLayerMap[originalSubgraphLayer];

        //connect all cloned layers as per original subgraph
        auto outputSlot = clonedLayer->BeginOutputSlots();
        for (auto&& originalOutputSlot : originalSubgraphLayer->GetOutputSlots())
        {
            for (auto&& connection : originalOutputSlot.GetConnections())
            {
                const Layer& otherTgtLayer = connection->GetOwningLayer();
                // in the case that the connection is a layer outside the subgraph, it will not have a corresponding connection
                if (originalToClonedLayerMap.find(&otherTgtLayer) != originalToClonedLayerMap.end())
                {
                    Layer* const newGrTgtLayer = originalToClonedLayerMap[&otherTgtLayer];

                    InputSlot& inputSlot = newGrTgtLayer->GetInputSlot(connection->GetSlotIndex());
                    outputSlot->Connect(inputSlot);
                }
            }
            outputSlot->SetTensorInfo(originalOutputSlot.GetTensorInfo());
            ++outputSlot;
        }
    }

    // creating new layers for the output slots, adding them to the new graph and connecting them
    for (auto os : originalSubgraph.GetOutputSlots())
    {
        Layer& originalSubgraphLayer = os->GetOwningLayer();
        Layer* const clonedLayer     = originalToClonedLayerMap[&originalSubgraphLayer];

        uint32_t i = 0;
        for (; i < originalSubgraphLayer.GetNumOutputSlots(); ++i)
        {
            if (os == &originalSubgraphLayer.GetOutputSlot(i))
            {
                break;
            }
        }

        ARMNN_ASSERT(i < originalSubgraphLayer.GetNumOutputSlots());

        const std::string& originalLayerName = os->GetConnection(0)->GetOwningLayer().GetNameStr();

        OutputSlot* outputSlotOfLayer     = &clonedLayer->GetOutputSlot(i);
        OutputLayer* const newOutputLayer = newGraph.AddLayer<OutputLayer>(slotCount, originalLayerName.c_str());
        ++slotCount;

        outputSlotOfLayer->Connect(newOutputLayer->GetInputSlot(0));
        outputSlotOfLayer->SetTensorInfo(originalSubgraphLayer.GetOutputSlot(i).GetTensorInfo());
    }

    return newGraph;
}

SubgraphView CustomSubgraphViewOptimizer::ReinterpretGraphToSubgraph(Graph& newGraph)
{
    std::list<Layer*> graphLayers(newGraph.begin(), newGraph.end());
    std::list<Layer*> subgrLayers;

    std::vector<InputLayer*> inputLayersNewGr;
    std::vector<OutputLayer*> outputLayersNewGr;

    for (auto layer : graphLayers)
    {
        switch (layer->GetType())
        {
            case LayerType::Input:
                inputLayersNewGr.push_back(PolymorphicDowncast<InputLayer*>(layer));
                break;
            case LayerType::Output:
                outputLayersNewGr.push_back(PolymorphicDowncast<OutputLayer*>(layer));
                break;
            default:
                subgrLayers.push_back(layer);
                break;
        }
    }

    std::vector<InputSlot*> inSlotsPointers;
    for (InputLayer* is : inputLayersNewGr)
    {
        inSlotsPointers.push_back(is->GetOutputSlot(0).GetConnection(0));
    }

    std::vector<OutputSlot*> outSlotsPointers;
    for (OutputLayer* os : outputLayersNewGr)
    {
        outSlotsPointers.push_back(os->GetInputSlot(0).GetConnectedOutputSlot());
    }

    return SubgraphView(std::move(inSlotsPointers), std::move(outSlotsPointers), std::move(subgrLayers));
}

SubgraphView CustomSubgraphViewOptimizer::OptimizeSubgraph(Graph& graph)
{
   
    SubgraphView subgraphToCompile = ReinterpretGraphToSubgraph(graph);
    //check for sure
    //PrintGraph(subgraphToCompile);
    return subgraphToCompile; 
}

} // namespace armnn
