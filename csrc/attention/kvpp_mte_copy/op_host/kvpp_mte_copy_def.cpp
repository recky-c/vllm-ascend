/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "register/op_def_registry.h"

namespace ops {
class KvppMteCopy : public OpDef {
public:
    explicit KvppMteCopy(const char* name) : OpDef(name)
    {
        this->Input("anchor")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8, ge::DT_INT8, ge::DT_FLOAT16,
                       ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32,
                       ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("localOffsets")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("stagingOffsets")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("lengths")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Attr("stagingBase").Int();
        this->Attr("sourceRank").Int();
        this->Attr("destinationRank").Int();
        this->Attr("shmId").Int();

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(KvppMteCopy);
}  // namespace ops
