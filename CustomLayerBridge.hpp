//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <functional>
#include <map>

#include <Layer.hpp>

namespace armnn
{

//replace it with user defined object ptr or reference
using CustomDefinedPtr = void*; 
using LayerFunctionPtr = std::function<bool(Layer&, CustomDefinedPtr)>;

class CustomLayerBridge
{
    public:
        LayerFunctionPtr GetLayerFunction(LayerType);
        void RegisterLayer(LayerType, LayerFunctionPtr);
        static CustomLayerBridge& GetBridge();

    private:
        //CustomLayerBridge()  = default;
        //~CustomLayerBridge() = default;
        CustomLayerBridge();
        ~CustomLayerBridge();

    private:
        std::map<LayerType,LayerFunctionPtr> m_LayerMap;
};

class CustomLayerRegistry
{
    public:
        CustomLayerRegistry(LayerType type, LayerFunctionPtr fn)
        {
             std::cout << "Register " << GetLayerTypeAsCString(type);
            CustomLayerBridge::GetBridge().RegisterLayer(type, fn);
        }
};

#define CUSTOM_LAYER_REGISTRY(LAYERTYPE) \
    static CustomLayerRegistry g_Register##LAYERTYPE(LayerType::LAYERTYPE, Add##LAYERTYPE##Layer);

} // namespace armnn
