//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CustomPreCompiledObject.hpp"

#include <armnn/Exceptions.hpp>

namespace armnn
{

void CustomPreCompiledObject::DoElementwiseAddition(const float* inputData0,
                                                  const float* inputData1,
                                                  float* outputData,
                                                  unsigned int numElements) const
{
    if (!inputData0 ||
        !inputData1 ||
        !outputData)
    {
        throw RuntimeException("Invalid input/output data");
    }

    // Do an element-wise addition of the given inputs
    for (unsigned int i = 0; i < numElements; ++i)
    {
        outputData[i] = inputData0[i] + inputData1[i];
    }
}

} // namespace armnn
