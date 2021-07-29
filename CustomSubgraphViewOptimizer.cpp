//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "CustomLayerBridge.hpp"


namespace armnn
{

CustomLayerBridge& CustomLayerBridge::GetBridge()
{
    static CustomLayerBridge bridge;
    return bridge;
}

LayerFunctionPtr CustomLayerBridge::GetLayerFunction(LayerType type)
{
    std::map<LayerType, LayerFunctionPtr>::iterator iter = m_LayerMap.find(type);
    if(iter == m_LayerMap.end())
        return nullptr;
    return iter->second;
}

void CustomLayerBridge::RegisterLayer(LayerType type, LayerFunctionPtr fn)
{
    m_LayerMap.insert(std::make_pair<LayerType, LayerFunctionPtr>(type, fn));
}



} // namespace armnn
