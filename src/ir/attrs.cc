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

/*!
 * \file attrs.cc
 */
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>

namespace tvm {

namespace {

ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> VisitMetadata(ffi::StructuralVisitorObj* visitor,
                                                                ffi::AnyView value) noexcept {
  if (auto map = value.as<ffi::MapObj>()) {
    for (const auto& [key, element] : *map) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(VisitMetadata(visitor, key));
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(VisitMetadata(visitor, element));
    }
    return std::nullopt;
  }
  if (auto array = value.as<ffi::ArrayObj>()) {
    for (const auto& element : *array) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(VisitMetadata(visitor, element));
    }
    return std::nullopt;
  }
  return visitor->VisitExpected(value);
}

ffi::Expected<ffi::UnchangedOr<ffi::Any>> MutateMetadata(ffi::StructuralMutatorObj* mutator,
                                                         ffi::AnyView value) noexcept {
  if (auto map = value.as<ffi::MapObj>()) {
    ffi::Map<ffi::Any, ffi::Any> rewritten;
    bool changed = false;
    for (const auto& [key, element] : *map) {
      TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Any>, mapped_key,
                                        MutateMetadata(mutator, key));
      TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Any>, mapped_value,
                                        MutateMetadata(mutator, element));
      changed |= !mapped_key.UnchangedOrSameAs(key) || !mapped_value.UnchangedOrSameAs(element);
      rewritten.Set(std::move(mapped_key).ValueOrUnchanged(key),
                    std::move(mapped_value).ValueOrUnchanged(element));
    }
    if (!changed) return ffi::Unchanged();
    return ffi::Any(std::move(rewritten));
  }
  if (auto array = value.as<ffi::ArrayObj>()) {
    ffi::Array<ffi::Any> rewritten;
    bool changed = false;
    for (const auto& element : *array) {
      TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Any>, mapped,
                                        MutateMetadata(mutator, element));
      changed |= !mapped.UnchangedOrSameAs(element);
      rewritten.push_back(std::move(mapped).ValueOrUnchanged(element));
    }
    if (!changed) return ffi::Unchanged();
    return ffi::Any(std::move(rewritten));
  }
  return mutator->MutateExpected(value, ffi::InplaceMode::kDisallow);
}

ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> DictAttrsVisit(ffi::StructuralVisitorObj* visitor,
                                                                 ffi::AnyView value) noexcept {
  const auto* attrs = value.as<DictAttrsNode>();
  return VisitMetadata(visitor, attrs->dict);
}

ffi::Expected<ffi::UnchangedOr<ffi::Any>> DictAttrsMutate(ffi::StructuralMutatorObj* mutator,
                                                          ffi::AnyView value) noexcept {
  const auto* attrs = value.as<DictAttrsNode>();
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Any>, mapped,
                                    MutateMetadata(mutator, attrs->dict));
  if (mapped.IsUnchanged()) return ffi::Unchanged();
  return ffi::Any(
      DictAttrs(std::move(mapped).ValueUnchecked().as_or_throw<ffi::Map<ffi::String, ffi::Any>>()));
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  DictAttrsNode::RegisterReflection();
  ffi::reflection::TypeAttrDef<DictAttrsNode>()
      .native_fn_attr<&DictAttrsVisit>(ffi::type_attr::kStructuralVisit)
      .native_fn_attr<&DictAttrsMutate>(ffi::type_attr::kStructuralMutate)
      .native_fn_attr<&DictAttrsMutate>(ffi::type_attr::kStructuralMaybeInplaceMutate);
}

TVM_FFI_STATIC_INIT_BLOCK() { tvm::ffi::reflection::ObjectDef<AttrsNode>(); }

}  // namespace tvm
