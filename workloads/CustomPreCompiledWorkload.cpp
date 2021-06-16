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
}

void CustomPreCompiledWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT("Custom", "CustomPreCompiledWorkload_Execute");

    // This example pre-compiled workload does an element-wise addition on the given inputs using the
    // method stored in the pre-compiled object (to mock a computation done on the custom backend)

    // Get the input/output buffers
    const float* inputData0 = GetInputTensorData<PreCompiledQueueDescriptor, float>(0, m_Data);
    const float* inputData1 = GetInputTensorData<PreCompiledQueueDescriptor, float>(1, m_Data);
    float* outputData       = GetOutputTensorData<PreCompiledQueueDescriptor, float>(0, m_Data);

    // Get the number of elements
    const TensorInfo& info = GetTensorInfo(m_Data.m_Inputs[0]);
    unsigned int numElements = info.GetNumElements();

    // Do the work
    m_PreCompiledObject->DoElementwiseAddition(inputData0, inputData1, outputData, numElements);
}

} // namespace armnn
