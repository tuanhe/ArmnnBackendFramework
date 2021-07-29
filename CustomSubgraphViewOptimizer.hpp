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

using LayerFunctionPtr = std::function<bool(Layer*)>;

class CustomLayerBridge
{
    public:
        LayerFunctionPtr GetLayerFunction(LayerType);
        void RegisterLayer(LayerType, LayerFunctionPtr);
        static CustomLayerBridge& GetBridge();

    private:
        CustomLayerBridge()  = default;
        ~CustomLayerBridge() = default;

    private:
        std::map<LayerType,LayerFunctionPtr> m_LayerMap;
};

} // namespace armnn
