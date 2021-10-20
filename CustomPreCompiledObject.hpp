//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

namespace armnn
{

// Mock class to simulate a pre-compiled object
// The pre-compiled object is used by the pre-compiled workload to simulate an optimized addition operation
class CustomPreCompiledObject
{
public:
    CustomPreCompiledObject() = default;
    ~CustomPreCompiledObject() = default;

    bool CompileGraph(void*);
    bool Inference();
    bool ReleaseMemory();
    // Simple example method to perform an element-wise addition to the given data (pretending that
    // the backend performs it better than the default implementation)
    void DoElementwiseAddition(const float* inputData0,
                             const float* inputData1,
                             float* outputData,
                             unsigned int numElements) const;
};

} // namespace armnn
