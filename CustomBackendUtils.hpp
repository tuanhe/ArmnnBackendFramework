//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Tensor.hpp>
#include <armnn/Types.hpp>

#include <backendsCommon/TensorHandle.hpp>

#include <boost/polymorphic_cast.hpp>

const armnn::TensorInfo& GetTensorInfo(const armnn::ITensorHandle* tensorHandle);

template <typename DataType>
const DataType* GetConstCpuData(const armnn::ITensorHandle* tensorHandle)
{
    const armnn::ConstTensorHandle* cpuTensorHandle =
        boost::polymorphic_downcast<const armnn::ConstTensorHandle*>(tensorHandle);

    return cpuTensorHandle->GetConstTensor<DataType>();
}

template <typename DataType>
DataType* GetCpuData(const armnn::ITensorHandle* tensorHandle)
{
    // We know that reference workloads use CpuTensorHandles only, so this cast is legitimate.
    const armnn::TensorHandle* cpuTensorHandle =
            boost::polymorphic_downcast<const armnn::TensorHandle*>(tensorHandle);

    return cpuTensorHandle->GetTensor<DataType>();
}

template <typename QueueDescriptor, typename DataType>
const DataType* GetInputTensorData(unsigned int idx, const QueueDescriptor& data)
{
    const armnn::ITensorHandle* tensorHandle = data.m_Inputs[idx];

    return GetConstCpuData<DataType>(tensorHandle);
}

template <typename QueueDescriptor, typename DataType>
DataType* GetOutputTensorData(unsigned int idx, const QueueDescriptor& data)
{
    const armnn::ITensorHandle* tensorHandle = data.m_Outputs[idx];

    return GetCpuData<DataType>(tensorHandle);
}
