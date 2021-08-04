//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CustomLayerSupport.hpp"

#include <InternalTypes.hpp>
#include <LayerSupportCommon.hpp>
#include <armnn/Types.hpp>

#include <boost/core/ignore_unused.hpp>

using namespace boost;

namespace armnn
{

bool IsDataTypeSupported(DataType dataType, Optional<std::string&> reasonIfUnsupported)
{
    if (dataType == DataType::QAsymmU8)
    {
        return true;
    }
    else
    {
        reasonIfUnsupported.value() = "Data type not supported.";
        return false;
    }
}

bool CustomLayerSupport::IsInputSupported(const TensorInfo& input,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    return IsDataTypeSupported(input.GetDataType(), reasonIfUnsupported);
}

bool CustomLayerSupport::IsOutputSupported(const TensorInfo& output,
                                           Optional<std::string&> reasonIfUnsupported) const
{
    return IsDataTypeSupported(output.GetDataType(), reasonIfUnsupported);
}

bool CustomLayerSupport::IsAdditionSupported(const TensorInfo& input0,
                                             const TensorInfo& input1,
                                             const TensorInfo& output,
                                             Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(input1);
    ignore_unused(output);
    return IsDataTypeSupported(input0.GetDataType(), reasonIfUnsupported);
}

bool CustomLayerSupport::IsMemCopySupported(const armnn::TensorInfo &input,
                                            const armnn::TensorInfo &output,
                                            armnn::Optional<std::string &> reasonIfUnsupported) const
{
    ignore_unused(output);
    return IsDataTypeSupported(input.GetDataType(), reasonIfUnsupported);
}

bool CustomLayerSupport::IsPreCompiledSupported(const armnn::TensorInfo &input,
                                                const armnn::PreCompiledDescriptor &descriptor,
                                                armnn::Optional<std::string &> reasonIfUnsupported) const
{
    ignore_unused(descriptor);
    return IsDataTypeSupported(input.GetDataType(), reasonIfUnsupported);
}

bool CustomLayerSupport::IsConvolution2dSupported(const TensorInfo& input,
                                  const TensorInfo& output,
                                  const Convolution2dDescriptor& descriptor,
                                  const TensorInfo& weights,
                                  const Optional<TensorInfo>& biases,
                                  Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(weights);
    ignore_unused(biases);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsDepthwiseConvolutionSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         const DepthwiseConvolution2dDescriptor& descriptor,
                                         const TensorInfo& weights,
                                         const Optional<TensorInfo>& biases,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(weights);
    ignore_unused(biases);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsTransposeConvolution2dSupported(const TensorInfo& input,
                                           const TensorInfo& output,
                                           const TransposeConvolution2dDescriptor& descriptor,
                                           const TensorInfo& weights,
                                           const Optional<TensorInfo>& biases,
                                           Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(weights);
    ignore_unused(biases);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsFullyConnectedSupported(const TensorInfo& input,
                                   const TensorInfo& output,
                                   const TensorInfo& weights,
                                   const TensorInfo& biases,
                                   const FullyConnectedDescriptor& descriptor,
                                   Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(weights);
    ignore_unused(biases);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsPooling2dSupported(const TensorInfo& input,
                              const TensorInfo& output,
                              const Pooling2dDescriptor& descriptor,
                              Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsReshapeSupported(const TensorInfo& input,
                            const TensorInfo& output,
                            const ReshapeDescriptor& descriptor,
                            Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsSoftmaxSupported(const TensorInfo& input,
                            const TensorInfo& output,
                            const SoftmaxDescriptor& descriptor,
                            Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}


} // namespace armnn
