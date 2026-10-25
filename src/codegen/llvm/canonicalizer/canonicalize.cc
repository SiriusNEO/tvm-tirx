/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include "canonicalize.h"

#include <string>

namespace tvm {
namespace target {
namespace canonicalizer {
namespace llvm {

ffi::Optional<ffi::String> DetectSystemTriple() {
  auto pf = tvm::ffi::Function::GetGlobal("codegen.llvm_get_system_triple");
  if (pf) return (*pf)().cast<ffi::String>();
  return {};
}

ffi::Map<ffi::String, ffi::Any> Canonicalize(ffi::Map<ffi::String, ffi::Any> target) {
  ffi::String kind = target.Get("kind").value().as_or_throw<ffi::String>();
  ffi::Optional<ffi::String> mtriple =
      target.Get("mtriple").value_or(nullptr).as_or_throw<ffi::Optional<ffi::String>>();
  ffi::Optional<ffi::String> mcpu =
      target.Get("mcpu").value_or(nullptr).as_or_throw<ffi::Optional<ffi::String>>();

  // Try to fill in the blanks by detecting target information from the system
  if (kind == "llvm" && !mtriple.has_value() && !mcpu.has_value()) {
    ffi::String system_triple = DetectSystemTriple().value_or("");
    target.Set("mtriple", system_triple);
  }

  std::string triple = target.Get("mtriple").value_or(ffi::String("")).as_or_throw<ffi::String>();
  std::string arch = triple.substr(0, triple.find('-'));
  auto canonicalize = ffi::Function::GetGlobal("codegen.llvm.canonicalize_target." + arch);
  if (!canonicalize) {
    canonicalize = ffi::Function::GetGlobal("codegen.llvm.canonicalize_target.default");
  }
  if (canonicalize) {
    return (*canonicalize)(target).cast<ffi::Map<ffi::String, ffi::Any>>();
  }

  return target;
}

}  // namespace llvm
}  // namespace canonicalizer
}  // namespace target
}  // namespace tvm
