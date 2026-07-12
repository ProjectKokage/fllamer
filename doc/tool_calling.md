# Tool calling

Tool calling uses the model's pinned upstream Jinja chat template and parser.
Check the model template before enabling tools:

```dart
final model = LlamaModelConfig(modelPath: modelPath);
final capabilities = await LlamaChatTemplate.capabilities(model);
if (!capabilities.supportsTools || !capabilities.supportsToolCalls) {
  throw const UnsupportedFeatureException(
    'This model chat template does not support tools.',
  );
}
```

Define tools in `GenerationConfig`, then read the parsed assistant message from
the terminal stream chunk:

```dart
const lookup = LlamaToolDefinition(
  name: 'lookup',
  description: 'Look up a value in the app-local index.',
  parametersSchema: <String, Object?>{
    'type': 'object',
    'properties': <String, Object?>{
      'query': <String, Object?>{'type': 'string'},
    },
    'required': <Object?>['query'],
    'additionalProperties': false,
  },
);

final history = <ChatMessage>[ChatMessage.user('Find alpha.')];
ChatMessage? assistant;
await for (final chunk in engine.chat(
  messages: history,
  config: const GenerationConfig(
    toolCalling: LlamaToolCallingConfig(
      tools: <LlamaToolDefinition>[lookup],
    ),
  ),
)) {
  if (chunk.isDone) assistant = chunk.assistantMessage;
}

final parsed = assistant!;
history.add(parsed);
for (final call in parsed.toolCalls) {
  final query = call.arguments['query'] as String;
  final result = localIndex[query] ?? 'not found';
  history.add(
    ChatMessage.toolResult(
      toolCallId: call.id!,
      name: call.name,
      text: result,
    ),
  );
}
```

Call `engine.chat` again with the updated history to let the model consume the
tool results. Tool execution is always app-owned; `fllamer` does not call a
network service or dispatch functions.

The bridge applies template-provided lazy grammar triggers, parser state,
generation prefixes, and stop strings. Templates without tool support fail
with `UnsupportedFeatureException` instead of silently omitting definitions.
Parallel calls are enabled only when both the request and model template allow
them. Missing model-generated call IDs receive unique per-engine IDs.

Gemma 4 uses a model-specific non-JSON argument syntax. fllamer's pinned
planner derives that grammar from each declared `parametersSchema`, including
strict property names, required fields, nested types, arrays, compositions,
and literal values. Regardless of model family, every parsed terminal call is
validated again against its matching schema before `assistantMessage` is
returned. Invalid calls fail with `GenerationException`; they are never handed
to application tool dispatch.

Active tools cannot be combined with a custom grammar or JSON Schema response
constraint. Named tool choice is implemented by exposing only the selected
tool to the upstream template and requiring a call. Raw completion has no chat
template and therefore rejects tool configuration.

`GenerationChunk.text` remains the raw streamed model output. Use the terminal
chunk's `assistantMessage` for normalized content and typed `toolCalls`.
`GenerationChunk.stopReason` is non-null on terminal chunks. If `maxTokens` is
reached before strict chat parsing can finish, the stream fails with an
explicit maximum-token `GenerationException` rather than a generic parser
message.

An opt-in checksum-pinned official Apache-2.0 Qwen2.5 0.5B fixture verifies
weighted generation through the full path: live template capability detection,
named-call constraints, streamed XML output, native parsing, generated call
ids, typed argument maps, assistant-call history, correlated tool results, and
the final non-tool assistant response. Exact fixture metadata and the command
are recorded in [native_builds.md](native_builds.md). Model weights remain
app-owned and are never included in the package.
