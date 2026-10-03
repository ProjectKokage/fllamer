import 'dart:convert';
import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'config.dart';
import 'errors.dart';
import 'ffi/generated_bindings.dart';
import 'model_info.dart';
import 'prompt_source_limits.dart';

import 'native_bridge.dart';
import 'native_model_ops.dart';

extension NativeChatOps on NativeLlamaBridge {
  String loadAndFormatChat(
    LlamaModelConfig config,
    List<ChatMessage> messages, {
    required bool addAssistantPrompt,
    required LlamaToolCallingConfig toolCalling,
    bool? enableThinking,
    int? reasoningBudgetTokens,
    int? maximumPromptBytes,
  }) {
    return withLoadedModel(
      config,
      (model) => formatChatWithModel(
        model,
        messages,
        addAssistantPrompt: addAssistantPrompt,
        toolCalling: toolCalling,
        enableThinking: enableThinking,
        reasoningBudgetTokens: reasoningBudgetTokens,
        maximumPromptBytes: maximumPromptBytes,
      ),
    );
  }

  String formatChatWithModel(
    ffi.Pointer<llama_dart_model> model,
    List<ChatMessage> messages, {
    required bool addAssistantPrompt,
    required LlamaToolCallingConfig toolCalling,
    bool? enableThinking,
    int? reasoningBudgetTokens,
    int? maximumPromptBytes,
  }) {
    final prepared = prepareMultimodalChat(
      messages,
      multimodalMarker,
      maximumPromptBytes: maximumPromptBytes,
    );
    return renderPreparedChat(
      model,
      prepared.messages,
      addAssistantPrompt: addAssistantPrompt,
      toolCalling: toolCalling,
      enableThinking: enableThinking,
      reasoningBudgetTokens: reasoningBudgetTokens,
      maximumPromptBytes: maximumPromptBytes,
    ).prompt;
  }

  NativeRenderedChat renderPreparedChat(
    ffi.Pointer<llama_dart_model> model,
    List<ChatMessage> messages, {
    required bool addAssistantPrompt,
    required LlamaToolCallingConfig toolCalling,
    String? grammar,
    Map<String, Object?>? jsonSchema,
    bool? enableThinking,
    int? maximumPromptBytes,
    int? reasoningBudgetTokens,
  }) {
    final parseOutput = _requiresChatPlan(messages, toolCalling);
    if (parseOutput ||
        enableThinking != null ||
        reasoningBudgetTokens != null) {
      final plan = _createChatPlan(
        model,
        messages,
        toolCalling,
        addAssistantPrompt: addAssistantPrompt,
        grammar: grammar,
        jsonSchema: jsonSchema,
        enableThinking: enableThinking,
        maximumPromptBytes: maximumPromptBytes,
        reasoningBudgetTokens: reasoningBudgetTokens,
        parseOutput: parseOutput,
      );
      return NativeRenderedChat(prompt: plan.prompt, plan: plan);
    }

    try {
      return NativeRenderedChat(
        prompt: _applyChatTemplate(
          model,
          messages,
          addAssistantPrompt: addAssistantPrompt,
          maximumPromptBytes: maximumPromptBytes,
        ),
      );
    } on UnsupportedFeatureException {
      // Keep the plan for its prompt, grammar metadata, and stop strings. Plain
      // bounded output stays raw because the upstream terminal parser is
      // strict and can reject an otherwise valid truncated response.
      final plan = _createChatPlan(
        model,
        messages,
        toolCalling,
        addAssistantPrompt: addAssistantPrompt,
        grammar: grammar,
        jsonSchema: jsonSchema,
        enableThinking: enableThinking,
        maximumPromptBytes: maximumPromptBytes,
        reasoningBudgetTokens: reasoningBudgetTokens,
        parseOutput: false,
      );
      return NativeRenderedChat(prompt: plan.prompt, plan: plan);
    }
  }

  LlamaChatTemplateCapabilities loadAndReadChatTemplateCapabilities(
    LlamaModelConfig config,
  ) {
    return withLoadedModel(config, chatTemplateCapabilitiesOfModel);
  }

  LlamaChatTemplateCapabilities chatTemplateCapabilitiesOfModel(
    ffi.Pointer<llama_dart_model> model,
  ) {
    final out = calloc<llama_dart_chat_template_capabilities>();
    try {
      out.ref.struct_size = ffi.sizeOf<llama_dart_chat_template_capabilities>();
      check(
        bindings.llama_dart_model_get_chat_template_capabilities(model, out),
      );
      return LlamaChatTemplateCapabilities(
        supportsTools: out.ref.supports_tools != 0,
        supportsToolCalls: out.ref.supports_tool_calls != 0,
        supportsParallelToolCalls: out.ref.supports_parallel_tool_calls != 0,
      );
    } finally {
      calloc.free(out);
    }
  }

  int loadAndCountChatTokens(
    LlamaModelConfig config,
    List<ChatMessage> messages, {
    required bool addAssistantPrompt,
    required LlamaToolCallingConfig toolCalling,
    bool? enableThinking,
    int? reasoningBudgetTokens,
    int? maximumPromptBytes,
  }) {
    return withLoadedModel(
      config,
      (model) => countChatTokensWithModel(
        model,
        messages,
        addAssistantPrompt: addAssistantPrompt,
        toolCalling: toolCalling,
        enableThinking: enableThinking,
        reasoningBudgetTokens: reasoningBudgetTokens,
        maximumPromptBytes: maximumPromptBytes,
      ),
    );
  }

  int countChatTokensWithModel(
    ffi.Pointer<llama_dart_model> model,
    List<ChatMessage> messages, {
    required bool addAssistantPrompt,
    required LlamaToolCallingConfig toolCalling,
    bool? enableThinking,
    int? reasoningBudgetTokens,
    int? maximumPromptBytes,
  }) {
    final prompt = formatChatWithModel(
      model,
      messages,
      addAssistantPrompt: addAssistantPrompt,
      toolCalling: toolCalling,
      enableThinking: enableThinking,
      reasoningBudgetTokens: reasoningBudgetTokens,
      maximumPromptBytes: maximumPromptBytes,
    );
    if (maximumPromptBytes == null) {
      return tokenizeWithModel(
        model,
        prompt,
        addSpecial: true,
        parseSpecial: true,
      ).length;
    }
    // The native size-query already returns the exact count. A preflight needs
    // no owned token array (and may be measuring an over-context candidate).
    final bytes = utf8.encode(prompt);
    final pointer = calloc<ffi.Uint8>(bytes.length);
    final count = calloc<ffi.Size>();
    try {
      pointer.asTypedList(bytes.length).setAll(0, bytes);
      final result = bindings.llama_dart_model_tokenize(
        model,
        pointer,
        bytes.length,
        ffi.nullptr,
        0,
        count,
        1,
        1,
      );
      if (result == llama_dart_result.LLAMA_DART_ERROR_BUFFER_TOO_SMALL) {
        bindings.llama_dart_clear_last_error();
      } else {
        check(result);
      }
      return count.value;
    } finally {
      calloc.free(count);
      calloc.free(pointer);
    }
  }

  NativeChatPlan _createChatPlan(
    ffi.Pointer<llama_dart_model> model,
    List<ChatMessage> messages,
    LlamaToolCallingConfig toolCalling, {
    required bool addAssistantPrompt,
    String? grammar,
    Map<String, Object?>? jsonSchema,
    bool? enableThinking,
    int? maximumPromptBytes,
    int? reasoningBudgetTokens,
    required bool parseOutput,
  }) {
    var tools = toolCalling.toJson();
    final String toolChoice;
    switch (toolCalling.toolChoice) {
      case LlamaAutoToolChoice():
        toolChoice = 'auto';
      case LlamaNoToolChoice():
        toolChoice = 'none';
      case LlamaRequiredToolChoice():
        toolChoice = 'required';
      case LlamaNamedToolChoice(:final name):
        tools = List<Map<String, Object?>>.unmodifiable(
          tools.where((tool) {
            final function = tool['function']! as Map<String, Object?>;
            return function['name'] == name;
          }),
        );
        toolChoice = 'required';
    }
    final request = <String, Object?>{
      'messages': <Map<String, Object?>>[
        for (final message in messages) _chatMessageToOpenAiJson(message),
      ],
      'tools': tools,
      'tool_choice': toolChoice,
      'parallel_tool_calls': toolCalling.allowParallelToolCalls,
      'add_generation_prompt': addAssistantPrompt,
      'grammar': ?grammar,
      'json_schema': ?jsonSchema,
      'enable_thinking': ?enableThinking,
      'reasoning_budget_tokens': ?reasoningBudgetTokens,
    };
    final requestBytes = encodePromptJson(request, maximumPromptBytes);
    final requestPointer = calloc<ffi.Uint8>(requestBytes.length);
    final out = calloc<llama_dart_buffer>();
    try {
      requestPointer.asTypedList(requestBytes.length).setAll(0, requestBytes);
      check(
        bindings.llama_dart_model_create_chat_plan(
          model,
          requestPointer,
          requestBytes.length,
          out,
        ),
      );
      final data = out.ref.data;
      final size = out.ref.size;
      if (data == ffi.nullptr || size == 0) {
        throw const NativeBridgeException(
          'Native bridge returned an empty chat plan.',
        );
      }
      // This buffer was materialized by upstream/native rendering. Bound its
      // downstream Dart decode/JSON/prompt copies; do not claim renderer bounds.
      checkPromptBufferSize(size, maximumPromptBytes);
      final planJson = utf8.decode(data.asTypedList(size));
      final decoded = jsonDecode(planJson);
      if (decoded is! Map<Object?, Object?> || decoded['prompt'] is! String) {
        throw const NativeBridgeException(
          'Native bridge returned an invalid chat plan.',
        );
      }
      if (maximumPromptBytes != null) {
        promptUtf8Bytes(decoded['prompt'] as String, maximumPromptBytes);
      }
      final thinkingEndTags = decoded['thinking_end_tags'];
      if (reasoningBudgetTokens != null &&
          (decoded['reasoning_budget_tokens'] != reasoningBudgetTokens ||
              decoded['thinking_start_tag'] is! String ||
              (decoded['thinking_start_tag']! as String).isEmpty ||
              thinkingEndTags is! List<Object?> ||
              thinkingEndTags.isEmpty ||
              thinkingEndTags.any((tag) => tag is! String || tag.isEmpty))) {
        throw const UnsupportedFeatureException(
          'The native bridge does not support bounded reasoning.',
        );
      }
      return NativeChatPlan(
        json: planJson,
        prompt: decoded['prompt'] as String,
        parseOutput: parseOutput,
        usedToolCallIds: <String>{
          for (final message in messages) ...<String>{
            for (final call in message.toolCalls)
              if (call.id != null) call.id!,
            if (message.toolCallId != null) message.toolCallId!,
          },
        },
      );
    } on FormatException catch (error) {
      throw NativeBridgeException(
        'Native bridge returned invalid chat plan JSON.',
        cause: error,
      );
    } finally {
      bindings.llama_dart_buffer_free(out.ref.data);
      calloc.free(out);
      calloc.free(requestPointer);
    }
  }

  ChatMessage _parseChatOutput(
    NativeChatPlan plan,
    String output,
    LlamaToolCallingConfig toolCalling,
  ) {
    final planBytes = utf8.encode(plan.json);
    final outputBytes = utf8.encode(output);
    final planPointer = calloc<ffi.Uint8>(planBytes.length);
    final outputPointer = outputBytes.isEmpty
        ? ffi.nullptr
        : calloc<ffi.Uint8>(outputBytes.length);
    final out = calloc<llama_dart_buffer>();
    try {
      planPointer.asTypedList(planBytes.length).setAll(0, planBytes);
      if (outputBytes.isNotEmpty) {
        outputPointer.asTypedList(outputBytes.length).setAll(0, outputBytes);
      }
      check(
        bindings.llama_dart_chat_parse_output(
          planPointer,
          planBytes.length,
          outputPointer,
          outputBytes.length,
          out,
        ),
      );
      final data = out.ref.data;
      final size = out.ref.size;
      if (data == ffi.nullptr || size == 0) {
        throw const GenerationException(
          'Native chat parser returned an empty message.',
        );
      }
      final Object? decoded;
      try {
        decoded = jsonDecode(utf8.decode(data.asTypedList(size)));
      } on FormatException catch (error) {
        throw GenerationException(
          'Native chat parser returned malformed UTF-8 or JSON.',
          cause: error,
        );
      }
      if (decoded is! Map<Object?, Object?> || decoded['role'] != 'assistant') {
        throw const GenerationException(
          'Native chat parser returned an invalid assistant message.',
        );
      }
      final contentValue = decoded['content'];
      final content = switch (contentValue) {
        null => '',
        String value => value,
        _ => throw const GenerationException(
          'Native chat parser returned invalid assistant content.',
        ),
      };
      final callsValue = decoded['tool_calls'];
      if (callsValue == null) {
        return ChatMessage.assistant(content);
      }
      final calls = LlamaToolCalls.fromJson(
        callsValue,
        allowParallelToolCalls: toolCalling.allowParallelToolCalls,
      );
      if (toolCalling.toolChoice is LlamaNoToolChoice) {
        throw const GenerationException(
          'Model returned tool calls when tool choice was none.',
        );
      }
      final allowed = <String, LlamaToolDefinition>{
        for (final tool in toolCalling.tools) tool.name: tool,
      };
      final usedIds = <String>{...plan.usedToolCallIds};
      for (final call in calls) {
        final id = call.id;
        if (id != null && !usedIds.add(id)) {
          throw GenerationException(
            'Model reused tool call id $id from chat history.',
          );
        }
      }
      final normalizedCalls = <LlamaToolCall>[];
      for (final call in calls) {
        final tool = allowed[call.name];
        if (tool == null) {
          throw GenerationException(
            'Model returned an unknown tool call: ${call.name}.',
          );
        }
        if (toolCalling.toolChoice case LlamaNamedToolChoice(:final name)) {
          if (call.name != name) {
            throw GenerationException(
              'Model returned ${call.name} when $name was required.',
            );
          }
        }
        try {
          tool.validateArguments(call.arguments);
        } on ArgumentError catch (error) {
          throw GenerationException(
            'Model returned invalid arguments for ${call.name}: '
            '${error.message ?? error}.',
            cause: error,
          );
        }
        var id = call.id;
        if (id == null) {
          do {
            id = 'call_${nextToolCallId++}';
          } while (usedIds.contains(id));
          usedIds.add(id);
        }
        normalizedCalls.add(
          LlamaToolCall(id: id, name: call.name, arguments: call.arguments),
        );
      }
      return ChatMessage.assistantToolCalls(
        text: content,
        toolCalls: normalizedCalls,
      );
    } on FormatException catch (error) {
      throw GenerationException(
        'Native chat parser returned invalid JSON.',
        cause: error,
      );
    } on ArgumentError catch (error) {
      throw GenerationException(
        'Native chat parser returned invalid tool calls.',
        cause: error,
      );
    } finally {
      bindings.llama_dart_buffer_free(out.ref.data);
      calloc.free(out);
      if (outputPointer != ffi.nullptr) {
        calloc.free(outputPointer);
      }
      calloc.free(planPointer);
    }
  }

  String _applyChatTemplate(
    ffi.Pointer<llama_dart_model> model,
    List<ChatMessage> messages, {
    required bool addAssistantPrompt,
    int? maximumPromptBytes,
  }) {
    if (maximumPromptBytes != null) {
      var sourceBytes = ffi.sizeOf<llama_dart_chat_message>() * messages.length;
      checkPromptBufferSize(sourceBytes, maximumPromptBytes);
      for (final message in messages) {
        sourceBytes +=
            promptUtf8Bytes(message.text, maximumPromptBytes) +
            _chatRoleName(message.role).length;
        checkPromptBufferSize(sourceBytes, maximumPromptBytes);
      }
    }
    final nativeMessages = calloc<llama_dart_chat_message>(messages.length);
    final allocated = <ffi.Pointer<ffi.Uint8>>[];
    final out = calloc<llama_dart_buffer>();
    try {
      for (var i = 0; i < messages.length; i += 1) {
        final message = messages[i];
        final role = utf8.encode(_chatRoleName(message.role));
        final content = utf8.encode(message.text);
        final rolePointer = calloc<ffi.Uint8>(role.length);
        rolePointer.asTypedList(role.length).setAll(0, role);
        allocated.add(rolePointer);

        ffi.Pointer<ffi.Uint8> contentPointer = ffi.nullptr;
        if (content.isNotEmpty) {
          contentPointer = calloc<ffi.Uint8>(content.length);
          contentPointer.asTypedList(content.length).setAll(0, content);
          allocated.add(contentPointer);
        }

        nativeMessages[i]
          ..struct_size = ffi.sizeOf<llama_dart_chat_message>()
          ..role_data = rolePointer
          ..role_size = role.length
          ..content_data = contentPointer
          ..content_size = content.length;
      }

      check(
        bindings.llama_dart_model_apply_chat_template(
          model,
          nativeMessages,
          messages.length,
          addAssistantPrompt ? 1 : 0,
          out,
        ),
      );

      final data = out.ref.data;
      final size = out.ref.size;
      if (data == ffi.nullptr || size == 0) {
        return '';
      }
      // Upstream rendering and its native return buffer already exist. Guard
      // before the Dart decoded copy and subsequent tokenization/FFI copies.
      checkPromptBufferSize(size, maximumPromptBytes);
      try {
        return utf8.decode(data.asTypedList(size));
      } on FormatException catch (error) {
        throw NativeBridgeException(
          'Model chat template produced invalid UTF-8.',
          cause: error,
        );
      }
    } finally {
      bindings.llama_dart_buffer_free(out.ref.data);
      calloc.free(out);
      for (final pointer in allocated) {
        calloc.free(pointer);
      }
      calloc.free(nativeMessages);
    }
  }
}

({List<ChatMessage> messages, List<NativeMediaInput> media})
prepareMultimodalChat(
  List<ChatMessage> messages,
  String marker, {
  int? maximumPromptBytes,
}) {
  if (maximumPromptBytes != null) {
    var bytes = 0;
    for (final message in messages) {
      bytes += promptUtf8Bytes(message.text, maximumPromptBytes);
      for (final part in message.parts) {
        if (part is! TextPart) {
          bytes += promptUtf8Bytes(marker, maximumPromptBytes);
        }
      }
      checkPromptBufferSize(bytes, maximumPromptBytes);
    }
  }
  final formatted = <ChatMessage>[];
  final media = <NativeMediaInput>[];
  for (final message in messages) {
    if (message.parts.isEmpty) {
      formatted.add(message);
      continue;
    }
    final content = StringBuffer();
    for (final part in message.parts) {
      switch (part) {
        case TextPart(:final text):
          content.write(text);
        case ImagePart(:final path, :final bytes):
          content.write(marker);
          media.add(
            NativeMediaInput(
              type: llama_dart_media_type.LLAMA_DART_MEDIA_IMAGE.value,
              path: path,
              bytes: bytes,
            ),
          );
        case AudioPart(:final path, :final bytes):
          content.write(marker);
          media.add(
            NativeMediaInput(
              type: llama_dart_media_type.LLAMA_DART_MEDIA_AUDIO.value,
              path: path,
              bytes: bytes,
            ),
          );
        case VideoPart():
          throw const UnsupportedFeatureException(
            'Video chat input is not available in mobile builds.',
          );
      }
    }
    formatted.add(ChatMessage(role: message.role, text: content.toString()));
  }
  return (
    messages: List<ChatMessage>.unmodifiable(formatted),
    media: List<NativeMediaInput>.unmodifiable(media),
  );
}

bool _requiresChatPlan(
  List<ChatMessage> messages,
  LlamaToolCallingConfig toolCalling,
) {
  return toolCalling.tools.isNotEmpty ||
      messages.any((message) => message.hasToolData);
}

Map<String, Object?> _chatMessageToOpenAiJson(ChatMessage message) {
  final json = <String, Object?>{
    'role': _chatRoleName(message.role),
    'content': message.toolCalls.isNotEmpty && message.text.isEmpty
        ? null
        : message.text,
  };
  if (message.toolCalls.isNotEmpty) {
    json['tool_calls'] = <Map<String, Object?>>[
      for (final call in message.toolCalls) call.toOpenAiJson(),
    ];
  }
  final toolName = message.toolName;
  final toolCallId = message.toolCallId;
  if (toolName != null) {
    json['name'] = toolName;
  }
  if (toolCallId != null) {
    json['tool_call_id'] = toolCallId;
  }
  return json;
}

final class NativeChatPlan {
  NativeChatPlan({
    required this.json,
    required this.prompt,
    required this.parseOutput,
    required Set<String> usedToolCallIds,
  }) : usedToolCallIds = Set<String>.unmodifiable(usedToolCallIds);

  final String json;
  final String prompt;
  final bool parseOutput;
  final Set<String> usedToolCallIds;
}

final class NativeRenderedChat {
  const NativeRenderedChat({required this.prompt, this.plan});

  final String prompt;
  final NativeChatPlan? plan;
}

final class NativeMediaInput {
  const NativeMediaInput({
    required this.type,
    required this.path,
    required this.bytes,
  });

  final int type;
  final String? path;
  final Uint8List? bytes;
}

ChatMessage parseTerminalChatOutput(
  NativeLlamaBridge bridge,
  NativeChatPlan plan,
  String output,
  LlamaToolCallingConfig toolCalling,
  GenerationStopReason stopReason,
) {
  try {
    return bridge._parseChatOutput(plan, output, toolCalling);
  } on GenerationException catch (error) {
    if (stopReason == GenerationStopReason.maxTokens &&
        error.message.startsWith('failed to parse chat output:')) {
      throw GenerationException(
        'Maximum token limit was reached before the chat or tool call was complete.',
        cause: error,
      );
    }
    rethrow;
  }
}

String _chatRoleName(ChatRole role) {
  return switch (role) {
    ChatRole.system => 'system',
    ChatRole.user => 'user',
    ChatRole.assistant => 'assistant',
    ChatRole.tool => 'tool',
  };
}
