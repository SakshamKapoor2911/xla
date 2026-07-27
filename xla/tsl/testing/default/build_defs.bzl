# Copyright 2026 The OpenXLA Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Default (OSS) implementation"""

load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")
load("//xla/util:build_defs.bzl", "text_to_binary_proto")

visibility(DEFAULT_LOAD_VISIBILITY)

def tsl_text_proto_test(name, src, message, deps = None, **_kwargs):
    text_to_binary_proto(
        name = name,
        src = src,
        proto_name = message,
        proto_deps = deps,
        testonly = True,
    )

def tsl_text_proto_test_suite(name, srcs, message, deps = None, **_kwargs):
    for i, src in enumerate(srcs):
        text_to_binary_proto(
            name = "%s_%d_binproto" % (name, i),
            src = src,
            proto_name = message,
            proto_deps = deps,
            testonly = True,
        )
