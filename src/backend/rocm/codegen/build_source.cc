#include <tvm/ffi/reflection/registry.h>

#include "../../../codegen/build_common.h"
#include "rocm_fallback_module.h"

namespace tvm::codegen {

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def(
      "target.build.rocm_from_source",
      [](IRModule module, ffi::String source, ffi::Optional<ffi::Bytes> binary) {
        auto functions = ExtractFuncInfo(module);
        ffi::Map<ffi::String, ffi::String> sources{{"hip", source}};
        if (!binary) {
          return target::ROCmFallbackModuleCreate(ffi::Bytes(), "hsaco", std::move(functions),
                                                  std::move(sources));
        }
        TVM_FFI_CHECK(binary.value().size() != 0, ValueError)
            << "ROCm compilation returned an empty binary";
        return target::ROCmModuleCreateWithFallback(*binary, "hsaco", std::move(functions),
                                                    std::move(sources));
      });
}

}
