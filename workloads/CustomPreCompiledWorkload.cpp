//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CustomPreCompiledWorkload.hpp"

#include <custom/CustomBackendUtils.hpp>

#include <armnn/Tensor.hpp>
#include <armnn/Types.hpp>

#include <Profiling.hpp>

#include <backendsCommon/TensorHandle.hpp>

#include <boost/polymorphic_cast.hpp>

namespace armnn
{

CustomPreCompiledWorkload::CustomPreCompiledWorkload(const PreCompiledQueueDescriptor& descriptor,
                                                     const WorkloadInfo& info)
    : BaseWorkload<PreCompiledQueueDescriptor>(descriptor, info)
    , m_PreCompiledObject(static_cast<const CustomPreCompiledObject*>(descriptor.m_PreCompiledObject))
{
    // Check that the workload is holdind a pointer to a valid pre-compiled object
    if (m_PreCompiledObject == nullptr)
    {
        throw InvalidArgumentException("CustomPreCompiledWorkload requires a valid pre-compiled object");
    }

    std::string debugName = m_PreCompiledObject->GetDebugName();
    ARMNN_LOG(info) << "CustomPreCompiledObject name : " << debugName;
}

void CustomPreCompiledWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT("Custom", "CustomPreCompiledWorkload_Execute");

    // This example pre-compiled workload does an element-wise addition on the given inputs using the
    // method stored in the pre-compiled object (to mock a computation done on the custom backend)
    // Get the number of elements
    const TensorInfo& info = GetTensorInfo(m_Data.m_Inputs[0]);
    unsigned int numElements = info.GetNumElements();
    
    std::vector<Buffer> input;
    std::vector<Buffer> output;
    for(uint32_t i = 0; i < 2; ++i)
    {
        Buffer b;
        const float* inputData = GetInputTensorData<PreCompiledQueueDescriptor, float>(i, m_Data);
        b.data = const_cast<float*>(inputData);
        b.size = numElements;
        input.push_back(b);
    }

    for(uint32_t i = 0; i < 1; ++i)
    {
        Buffer b;
        float* outputData = GetOutputTensorData<PreCompiledQueueDescriptor, float>(i, m_Data);
        b.data = reinterpret_cast<float*>(outputData);
        b.size = numElements;
        output.push_back(b);
    }
    // Do the work
    //m_PreCompiledObject->DoElementwiseAddition(inputData0, inputData1, outputData, numElements);
    m_PreCompiledObject->Inference(input, output);
}

} // namespace armnn
