//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <string>
#include <vector>

namespace armnn
{
typedef struct _buffer{
    float* data;
    uint32_t size;
}Buffer;
// Mock class to simulate a pre-compiled object
// The pre-compiled object is used by the pre-compiled workload to simulate an optimized addition operation
class CustomPreCompiledObject
{
public:
    CustomPreCompiledObject() = default;
    ~CustomPreCompiledObject() = default;

    bool PreInferenceStage(void*);
    bool Inference(std::vector<Buffer>&, std::vector<Buffer>&) const;
    bool PostInferenceStage();

private:
    void DoElementwiseAddition(const float* inputData0,
                             const float* inputData1,
                             float* outputData,
                             unsigned int numElements) const;
public:
    void SetDebugName(const std::string name);
    std::string GetDebugName() const;
private:
    std::string m_Name;
};

} // namespace armnn
