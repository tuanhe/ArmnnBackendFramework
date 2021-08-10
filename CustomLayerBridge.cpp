//
// Copyright © tuanhe. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CustomLayerBridge.hpp"

namespace armnn
{

CustomLayerBridge::CustomLayerBridge()
{
}

CustomLayerBridge::~CustomLayerBridge()
{
}

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
    std::cout << "Register " << GetLayerTypeAsCString(type);
    //printf("Register %s" , (GetLayerTypeAsCString(type)).c_str());
    m_LayerMap.insert(std::pair<LayerType, LayerFunctionPtr>(type, fn));
}

} // namespace armnn
