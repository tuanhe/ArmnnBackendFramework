//
// Copyright © 2021 tuanhe. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../CustomLayerBridge.hpp"
#include <armnn/Logging.hpp>

namespace armnn
{

bool Pooling2d(Layer& layer, CustomDefinedPtr obj)
{
    ARMNN_LOG(info) << __FUNCTION__ << " : " << __LINE__ << " : " << layer.GetNameStr();
    armnn::IgnoreUnused(obj);
    return true;
}

CUSTOM_LAYER_REGISTRY(Pooling2d);

} // namespace armnn
