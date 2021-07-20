//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CustomBackend.hpp"
#include "CustomWorkloadFactory.hpp"
#include "CustomLayerSupport.hpp"
#include "CustomPreCompiledObject.hpp"

#include <backendsCommon/IBackendContext.hpp>
#include <backendsCommon/IMemoryManager.hpp>
#include <armnn/BackendRegistry.hpp>
#include <armnn/utility/IgnoreUnused.hpp>

#include <Optimizer.hpp>

#include <boost/assert.hpp>
#include <boost/cast.hpp>

namespace armnn
{

const BackendId& CustomBackend::GetIdStatic()
{
    static const BackendId s_Id{"Custom"};
    return s_Id;
}

IBackendInternal::IWorkloadFactoryPtr CustomBackend::CreateWorkloadFactory(
        const IBackendInternal::IMemoryManagerSharedPtr& memoryManager) const
{
        IgnoreUnused(memoryManager);
	return std::make_unique<CustomWorkloadFactory>();
}

IBackendInternal::IBackendContextPtr CustomBackend::CreateBackendContext(const IRuntime::CreationOptions&) const
{
    return IBackendContextPtr{};
}

IBackendInternal::IMemoryManagerUniquePtr CustomBackend::CreateMemoryManager() const
{
    return IMemoryManagerUniquePtr{};
}

IBackendInternal::ILayerSupportSharedPtr CustomBackend::GetLayerSupport() const
{
    static ILayerSupportSharedPtr layerSupport{new CustomLayerSupport};
    return layerSupport;
}

OptimizationViews CustomBackend::OptimizeSubgraphView(const SubgraphView& subgraph) const
{
    // Mocking a substitution of the whole given sub-graph with a single pre-compiled layer

    // Initialize the optimization views to return
    OptimizationViews optimizationViews;

    // Pretending that the only thing this custom backend can optimize is a single addition layer
    const SubgraphView::Layers& subgraphLayers = subgraph.GetLayers();
    if (subgraphLayers.size() != 1)
    {
        // This custom backend cannot optimize this sub-graph (as it has too many layers),
        // but it can run on the backend as it is so we add it to the untouched subgraphs.
        optimizationViews.AddUntouchedSubgraph(SubgraphView(subgraph));

        return optimizationViews;
    }

    // Get the sub-graph's only layer
    Layer* const subgraphLayer = *(subgraphLayers.begin());
    if (subgraphLayer->GetType() != LayerType::Addition)
    {
        // This custom backend cannot optimize this sub-graph (has it is not an addition layer),
        // but it can run on the backend as it is so we add it to the untouched subgraphs.
        optimizationViews.AddUntouchedSubgraph(SubgraphView(subgraph));

        return optimizationViews;
    }

    // Optimize the addition layer by replacing it with a pre-compiled layer that implements the same function,
    // (see CustomPreCompiledObject.cpp), but here for the sake of the example we pretend that this custom backend
    // actually performs it better!

    // Create the pre-compiled layer
    PreCompiledLayer* preCompiledLayer =
            optimizationViews.GetGraph().AddLayer<PreCompiledLayer>(PreCompiledDescriptor(subgraph.GetNumInputSlots(),
                                                                                          subgraph.GetNumOutputSlots()),
                                                                    "pre-compiled");

    // Defining a simple deleter for the mock pre-compiled object
    PreCompiledObjectDeleter customPrecompiledObjectDeleter = [](const void* data)
    {
        if (!data)
        {
            return;
        }

        const CustomPreCompiledObject* p = static_cast<const CustomPreCompiledObject*>(data);
        delete p;
    };

    // Mocking a pre-compiled object (the result of the optimization process done by the backend)
    // This example of a pre-compiled object simply performs an element-wise addition (see CustomPreCompiledObject.cpp)
    PreCompiledObjectPtr customPrecompiledObject(new CustomPreCompiledObject(), customPrecompiledObjectDeleter);

    // Check if we were able to create a pre-compiled layer
    if (preCompiledLayer)
    {
        // Optimization applied, copy the output tensor infos from the sub-graph
        for (unsigned int i = 0; i < subgraph.GetNumOutputSlots(); ++i)
        {
            preCompiledLayer->GetOutputSlot(i).SetTensorInfo(subgraph.GetOutputSlot(i)->GetTensorInfo());
        }

        // Set the backend id to the pre-compiled layer, so that it will be executed on this backend
        preCompiledLayer->SetBackendId(GetIdStatic());

        // Assign the mock pre-compiled object to the layer
        preCompiledLayer->SetPreCompiledObject(std::move(customPrecompiledObject));

        // Add the pair sub-graph <-> pre-compiled layer to the list of substitutions
        optimizationViews.AddSubstitution({ SubgraphView(subgraph), SubgraphView(preCompiledLayer) });
    }
    else
    {
        // No optimization applied, report the optimization of the given sub-graph as failed
        optimizationViews.AddFailedSubgraph(SubgraphView(subgraph));
    }

    return optimizationViews;
}

} // namespace armnn
