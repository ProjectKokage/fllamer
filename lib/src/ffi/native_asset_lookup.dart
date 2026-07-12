import 'dart:ffi' as ffi;

import 'generated_native_asset_bindings.dart' as native_asset;

/// Looks up a bridge symbol in the bundled native asset.
///
/// Keep this switch in sync with `ffigen.native_assets.yaml`. Its cases are
/// deliberately lazy: only the requested symbol is resolved by the runtime.
ffi.Pointer<T> lookupLlamaDartNativeAssetSymbol<T extends ffi.NativeType>(
  String symbolName,
) {
  return switch (symbolName) {
    'llama_dart_abi_version' =>
      native_asset.addresses.llama_dart_abi_version.cast<T>(),
    'llama_dart_upstream_commit' =>
      native_asset.addresses.llama_dart_upstream_commit.cast<T>(),
    'llama_dart_build_flags' =>
      native_asset.addresses.llama_dart_build_flags.cast<T>(),
    'llama_dart_model_file_type_name' =>
      native_asset.addresses.llama_dart_model_file_type_name.cast<T>(),
    'llama_dart_multimodal_marker' =>
      native_asset.addresses.llama_dart_multimodal_marker.cast<T>(),
    'llama_dart_log_set_level' =>
      native_asset.addresses.llama_dart_log_set_level.cast<T>(),
    'llama_dart_log_next' =>
      native_asset.addresses.llama_dart_log_next.cast<T>(),
    'llama_dart_backend_init' =>
      native_asset.addresses.llama_dart_backend_init.cast<T>(),
    'llama_dart_backend_free' =>
      native_asset.addresses.llama_dart_backend_free.cast<T>(),
    'llama_dart_get_capabilities' =>
      native_asset.addresses.llama_dart_get_capabilities.cast<T>(),
    'llama_dart_model_load' =>
      native_asset.addresses.llama_dart_model_load.cast<T>(),
    'llama_dart_model_free' =>
      native_asset.addresses.llama_dart_model_free.cast<T>(),
    'llama_dart_model_get_info' =>
      native_asset.addresses.llama_dart_model_get_info.cast<T>(),
    'llama_dart_model_get_description' =>
      native_asset.addresses.llama_dart_model_get_description.cast<T>(),
    'llama_dart_model_metadata_count' =>
      native_asset.addresses.llama_dart_model_metadata_count.cast<T>(),
    'llama_dart_model_metadata_get' =>
      native_asset.addresses.llama_dart_model_metadata_get.cast<T>(),
    'llama_dart_model_get_chat_template' =>
      native_asset.addresses.llama_dart_model_get_chat_template.cast<T>(),
    'llama_dart_model_tokenize' =>
      native_asset.addresses.llama_dart_model_tokenize.cast<T>(),
    'llama_dart_model_detokenize' =>
      native_asset.addresses.llama_dart_model_detokenize.cast<T>(),
    'llama_dart_model_apply_chat_template' =>
      native_asset.addresses.llama_dart_model_apply_chat_template.cast<T>(),
    'llama_dart_model_get_chat_template_capabilities' =>
      native_asset.addresses.llama_dart_model_get_chat_template_capabilities
          .cast<T>(),
    'llama_dart_model_create_chat_plan' =>
      native_asset.addresses.llama_dart_model_create_chat_plan.cast<T>(),
    'llama_dart_chat_parse_output' =>
      native_asset.addresses.llama_dart_chat_parse_output.cast<T>(),
    'llama_dart_json_schema_to_grammar' =>
      native_asset.addresses.llama_dart_json_schema_to_grammar.cast<T>(),
    'llama_dart_context_create' =>
      native_asset.addresses.llama_dart_context_create.cast<T>(),
    'llama_dart_context_free' =>
      native_asset.addresses.llama_dart_context_free.cast<T>(),
    'llama_dart_context_reset' =>
      native_asset.addresses.llama_dart_context_reset.cast<T>(),
    'llama_dart_context_warm_up' =>
      native_asset.addresses.llama_dart_context_warm_up.cast<T>(),
    'llama_dart_context_cancel' =>
      native_asset.addresses.llama_dart_context_cancel.cast<T>(),
    'llama_dart_context_get_info' =>
      native_asset.addresses.llama_dart_context_get_info.cast<T>(),
    'llama_dart_context_state_get' =>
      native_asset.addresses.llama_dart_context_state_get.cast<T>(),
    'llama_dart_context_state_set' =>
      native_asset.addresses.llama_dart_context_state_set.cast<T>(),
    'llama_dart_context_shift' =>
      native_asset.addresses.llama_dart_context_shift.cast<T>(),
    'llama_dart_context_complete' =>
      native_asset.addresses.llama_dart_context_complete.cast<T>(),
    'llama_dart_generation_start' =>
      native_asset.addresses.llama_dart_generation_start.cast<T>(),
    'llama_dart_generation_next' =>
      native_asset.addresses.llama_dart_generation_next.cast<T>(),
    'llama_dart_generation_free' =>
      native_asset.addresses.llama_dart_generation_free.cast<T>(),
    'llama_dart_context_embed' =>
      native_asset.addresses.llama_dart_context_embed.cast<T>(),
    'llama_dart_context_rerank' =>
      native_asset.addresses.llama_dart_context_rerank.cast<T>(),
    'llama_dart_lora_load' =>
      native_asset.addresses.llama_dart_lora_load.cast<T>(),
    'llama_dart_lora_free' =>
      native_asset.addresses.llama_dart_lora_free.cast<T>(),
    'llama_dart_context_set_lora_adapters' =>
      native_asset.addresses.llama_dart_context_set_lora_adapters.cast<T>(),
    'llama_dart_buffer_free' =>
      native_asset.addresses.llama_dart_buffer_free.cast<T>(),
    'llama_dart_float_buffer_free' =>
      native_asset.addresses.llama_dart_float_buffer_free.cast<T>(),
    'llama_dart_last_error_message' =>
      native_asset.addresses.llama_dart_last_error_message.cast<T>(),
    'llama_dart_clear_last_error' =>
      native_asset.addresses.llama_dart_clear_last_error.cast<T>(),
    _ => throw ArgumentError.value(
      symbolName,
      'symbolName',
      'is not exported by the llama.dart bridge',
    ),
  };
}
