# Bundled Vulkan headers

This directory contains the complete `include` tree and `LICENSE.md` from
KhronosGroup/Vulkan-Headers tag `vulkan-sdk-1.4.357.0`.

- Upstream: `https://github.com/KhronosGroup/Vulkan-Headers`
- Source archive:
  `https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/vulkan-sdk-1.4.357.0.tar.gz`
- Source archive SHA-256:
  `e87dce08116151f6b6d7de6b6faf41498e87e6cf848ff16fa3bd5402190ad4a3`
- Imported subset: `include/` and `LICENSE.md`
- License: Apache-2.0 OR MIT as marked in the individual upstream files;
  see `LICENSE.md`.

The Android Vulkan native-assets variant uses these headers only while
cross-compiling. The Flutter-selected Android NDK remains authoritative for
the target Vulkan loader, SPIR-V headers, and host `glslc`. Updating this pin
requires replacing the complete imported subset, checking its notices, and
re-running the Android native build and device qualification paths.
