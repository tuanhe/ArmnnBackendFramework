//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CustomPreCompiledObject.hpp"

#include <armnn/Exceptions.hpp>

namespace armnn
{

bool CustomPreCompiledObject::PreInferenceStage(void*)
{
    return true;
}

bool CustomPreCompiledObject::Inference(std::vector<Buffer>& input, std::vector<Buffer>& output) const
{
    DoElementwiseAddition( reinterpret_cast<const float*> (input[0].data),
                           reinterpret_cast<const float*> (input[1].data),
                           reinterpret_cast<float*> (output[0].data),
                           input[0].size);
    return true;
}

bool CustomPreCompiledObject::PostInferenceStage()
{
    return true;
}


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

void CustomPreCompiledObject::SetDebugName(const std::string name)
{
    m_Name = name;
}

std::string CustomPreCompiledObject::GetDebugName() const
{
    return m_Name;
}


} // namespace armnn
