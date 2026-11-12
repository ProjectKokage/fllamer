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
    bool separateReasoning = false,
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
        separateReasoning: separateReasoning,
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
    bool separateReasoning = false,
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
      separateReasoning: separateReasoning,
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
    bool separateReasoning = false,
    bool withOutputParser = false,
  }) {
    final parseOutput = _requiresChatPlan(messages, toolCalling);
    final needsPlan =
        parseOutput || enableThinking != null || reasoningBudgetTokens != null;
    if (needsPlan || separateReasoning) {
      try {
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
          separateReasoning: separateReasoning,
          parseOutput: parseOutput,
          withOutputParser: withOutputParser,
        );
        return NativeRenderedChat(prompt: plan.prompt, plan: plan);
      } on UnsupportedFeatureException {
        if (needsPlan) {
          rethrow;
        }
        // Separation alone was asked of a template the planner cannot
        // render. The legacy formatter renders it, and its output streams as
        // generated.
      }
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
        separateReasoning: separateReasoning,
        parseOutput: false,
        withOutputParser: withOutputParser,
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
        marksReasoning: out.ref.marks_reasoning != 0,
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
    bool separateReasoning = false,
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
        separateReasoning: separateReasoning,
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
    bool separateReasoning = false,
    int? maximumPromptBytes,
  }) {
    final prepared = prepareMultimodalChat(
      messages,
      multimodalMarker,
      maximumPromptBytes: maximumPromptBytes,
    );
    final rendered = renderPreparedChat(
      model,
      prepared.messages,
      addAssistantPrompt: addAssistantPrompt,
      toolCalling: toolCalling,
      enableThinking: enableThinking,
      reasoningBudgetTokens: reasoningBudgetTokens,
      separateReasoning: separateReasoning,
      maximumPromptBytes: maximumPromptBytes,
    );
    final plan = rendered.plan;
    if (plan != null) {
      return _countChatPlanTokens(model, plan);
    }
    // The legacy formatter returns one string with no record of where its
    // messages are, and generation parses special tokens over all of it.
    final prompt = rendered.prompt;
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

  /// The number of tokens a completion of [plan] evaluates as its prompt.
  ///
  /// The bridge tokenizes the plan's prompt as generation does: the
  /// template's text with special-token parsing, and message text that spells
  /// a control token as text. The size query returns the exact count, so no
  /// token array is allocated.
  int _countChatPlanTokens(
    ffi.Pointer<llama_dart_model> model,
    NativeChatPlan plan,
  ) {
    final planBytes = utf8.encode(plan.json);
    final planPointer = calloc<ffi.Uint8>(planBytes.length);
    final count = calloc<ffi.Size>();
    try {
      planPointer.asTypedList(planBytes.length).setAll(0, planBytes);
      final result = bindings.llama_dart_model_tokenize_chat_plan(
        model,
        planPointer,
        planBytes.length,
        ffi.nullptr,
        0,
        count,
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
      calloc.free(planPointer);
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
    bool separateReasoning = false,
    required bool parseOutput,
    required bool withOutputParser,
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
    // The one rule for parsed streams: a plan whose stream is parsed is asked
    // to separate reasoning, so its parser never reads reasoning as response
    // text.
    final parsesStream =
        parseOutput ||
        enableThinking != null ||
        reasoningBudgetTokens != null ||
        separateReasoning;
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
      if (parsesStream) 'separate_reasoning': true,
    };
    final requestBytes = encodePromptJson(request, maximumPromptBytes);
    final requestPointer = calloc<ffi.Uint8>(requestBytes.length);
    final out = calloc<llama_dart_buffer>();
    // The pinned parser cannot be serialized, so a plan whose output will be
    // parsed receives its parser from the call that makes the plan.
    final parserOut = withOutputParser && parsesStream
        ? calloc<ffi.Pointer<llama_dart_chat_stream_parser>>()
        : ffi.nullptr;
    NativeChatStreamParser? outputParser;
    var planOwnsParser = false;
    try {
      requestPointer.asTypedList(requestBytes.length).setAll(0, requestBytes);
      check(
        bindings.llama_dart_model_create_chat_plan(
          model,
          requestPointer,
          requestBytes.length,
          out,
          parserOut,
        ),
      );
      if (parserOut != ffi.nullptr) {
        if (parserOut.value == ffi.nullptr) {
          throw const NativeBridgeException(
            'Native bridge returned a null chat stream parser.',
          );
        }
        outputParser = NativeChatStreamParser._(this, parserOut.value);
      }
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
      final plan = NativeChatPlan(
        json: planJson,
        prompt: decoded['prompt'] as String,
        parseOutput: parseOutput,
        parsesStream: parsesStream,
        outputParser: outputParser,
        usedToolCallIds: <String>{
          for (final message in messages) ...<String>{
            for (final call in message.toolCalls)
              if (call.id != null) call.id!,
            if (message.toolCallId != null) message.toolCallId!,
          },
        },
      );
      planOwnsParser = true;
      return plan;
    } on FormatException catch (error) {
      throw NativeBridgeException(
        'Native bridge returned invalid chat plan JSON.',
        cause: error,
      );
    } finally {
      if (!planOwnsParser) {
        outputParser?.close();
      }
      bindings.llama_dart_buffer_free(out.ref.data);
      calloc.free(out);
      if (parserOut != ffi.nullptr) {
        calloc.free(parserOut);
      }
      calloc.free(requestPointer);
    }
  }

  ({String text, String reasoningText, int toolCallBytes}) _parseChatStream(
    ffi.Pointer<llama_dart_chat_stream_parser> parser,
    String output, {
    required bool partial,
  }) {
    final outputBytes = utf8.encode(output);
    final outputPointer = outputBytes.isEmpty
        ? ffi.nullptr
        : calloc<ffi.Uint8>(outputBytes.length);
    final out = calloc<llama_dart_buffer>();
    try {
      if (outputBytes.isNotEmpty) {
        outputPointer.asTypedList(outputBytes.length).setAll(0, outputBytes);
      }
      check(
        bindings.llama_dart_chat_stream_parser_parse(
          parser,
          outputPointer,
          outputBytes.length,
          partial ? 1 : 0,
          out,
        ),
      );
      final data = out.ref.data;
      final size = out.ref.size;
      final decoded = data == ffi.nullptr || size == 0
          ? null
          : jsonDecode(utf8.decode(data.asTypedList(size)));
      if (decoded is! Map<String, Object?> ||
          decoded['content'] is! String ||
          decoded['reasoning_content'] is! String ||
          decoded['tool_call_bytes'] is! int) {
        throw const GenerationException(
          'Native bridge returned invalid parsed chat output.',
        );
      }
      return (
        text: decoded['content']! as String,
        reasoningText: decoded['reasoning_content']! as String,
        toolCallBytes: decoded['tool_call_bytes']! as int,
      );
    } on FormatException catch (error) {
      throw GenerationException(
        'Native bridge returned invalid parsed chat output.',
        cause: error,
      );
    } finally {
      bindings.llama_dart_buffer_free(out.ref.data);
      calloc.free(out);
      if (outputPointer != ffi.nullptr) {
        calloc.free(outputPointer);
      }
    }
  }

  ChatMessage _parseChatOutput(
    NativeChatPlan plan,
    String output,
    LlamaToolCallingConfig toolCalling,
  ) {
    final parser = plan.outputParser;
    if (parser == null) {
      throw StateError('The chat plan was made without an output parser.');
    }
    final outputBytes = utf8.encode(output);
    final outputPointer = outputBytes.isEmpty
        ? ffi.nullptr
        : calloc<ffi.Uint8>(outputBytes.length);
    final out = calloc<llama_dart_buffer>();
    try {
      if (outputBytes.isNotEmpty) {
        outputPointer.asTypedList(outputBytes.length).setAll(0, outputBytes);
      }
      check(
        bindings.llama_dart_chat_stream_parser_parse_message(
          parser._handle,
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
    required this.parsesStream,
    required this.outputParser,
    required Set<String> usedToolCallIds,
  }) : usedToolCallIds = Set<String>.unmodifiable(usedToolCallIds);

  /// What a completion of the plan is given.
  final String json;
  final String prompt;
  final bool parseOutput;

  /// Whether streamed output is parsed into response text and reasoning.
  ///
  /// True for a plan that asked the template for tools, a thinking mode, a
  /// reasoning budget or the separation itself. Such a plan was asked to
  /// separate reasoning. A plan made only because the legacy formatter could
  /// not render the template streams its output as generated.
  final bool parsesStream;

  /// The parser of this plan's output, for a plan that [parsesStream] and was
  /// made to be generated from. The plan owns it until [close].
  final NativeChatStreamParser? outputParser;
  final Set<String> usedToolCallIds;

  void close() => outputParser?.close();
}

/// Separates a planned chat response's text from its reasoning while the
/// response is generated, using the pinned upstream chat parser.
final class NativeChatStreamParser {
  NativeChatStreamParser._(this._bridge, this._parser);

  final NativeLlamaBridge _bridge;
  ffi.Pointer<llama_dart_chat_stream_parser> _parser;

  ffi.Pointer<llama_dart_chat_stream_parser> get _handle {
    if (_parser == ffi.nullptr) {
      throw const ResourceDisposedException('Chat stream parser is closed.');
    }
    return _parser;
  }

  String _text = '';
  String _reasoningText = '';
  int _toolCallBytes = 0;

  /// Parses [output], everything generated so far, and returns what it adds
  /// to the response text and to the reasoning since the previous call, and
  /// whether a tool call grew.
  ///
  /// Before [isDone] an end of [output] that could still become one of the
  /// template's markers is left unread. The last call delivers it as response
  /// text or as reasoning, whichever the model was writing, unless the output
  /// ends inside a tool call or the parser rejects the output.
  ({String text, String reasoningText, bool writesToolCall}) advance(
    String output, {
    required bool isDone,
  }) {
    final parsed = isDone
        ? _parseComplete(output)
        : _bridge._parseChatStream(_handle, output, partial: true);
    final text = _added(_text, parsed.text)!;
    final reasoningText = _added(_reasoningText, parsed.reasoningText)!;
    final writesToolCall = parsed.toolCallBytes > _toolCallBytes;
    _text += text;
    _reasoningText += reasoningText;
    if (writesToolCall) {
      _toolCallBytes = parsed.toolCallBytes;
    }
    return (
      text: text,
      reasoningText: reasoningText,
      writesToolCall: writesToolCall,
    );
  }

  /// Reads the finished response as ended output when the parser accepts it
  /// and that reading continues what was streamed.
  ///
  /// The parser rejects output that does not follow the template's format,
  /// and a reading of ended output may divide the output differently from the
  /// readings made while it streamed. The response is then read the way it
  /// was read while it streamed, so the last step never contradicts text
  /// that was already delivered. An end of the output that could still have
  /// become a marker is not delivered in that case.
  ({String text, String reasoningText, int toolCallBytes}) _parseComplete(
    String output,
  ) {
    try {
      final complete = _bridge._parseChatStream(
        _handle,
        output,
        partial: false,
      );
      if (_added(_text, complete.text, orNull: true) != null &&
          _added(_reasoningText, complete.reasoningText, orNull: true) !=
              null) {
        return complete;
      }
    } on GenerationException {
      // The parser rejects the output.
    }
    return _bridge._parseChatStream(_handle, output, partial: true);
  }

  void close() {
    if (_parser == ffi.nullptr) {
      return;
    }
    _bridge.bindings.llama_dart_chat_stream_parser_free(_parser);
    _parser = ffi.nullptr;
  }
}

/// The part of [current] after [delivered], which was already streamed.
///
/// The parser may briefly report less than it did before; nothing is taken
/// back then. Text that no longer begins with what was streamed is an error,
/// or null with [orNull].
String? _added(String delivered, String current, {bool orNull = false}) {
  if (current.startsWith(delivered)) {
    return current.substring(delivered.length);
  }
  if (delivered.startsWith(current)) {
    return '';
  }
  if (orNull) {
    return null;
  }
  throw const GenerationException(
    'The chat parser changed output that was already streamed.',
  );
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
