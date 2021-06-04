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
    if (dataType == DataType::Float32)
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

} // namespace armnn
