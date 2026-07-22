import 'dart:ffi' as ffi;
import 'dart:io';
import 'dart:typed_data';

import 'package:fllamer/fllamer.dart';
import 'package:test/test.dart';

void main() {
  group('ordinary chat template fallback', () {
    _FakeBridgeFixture? fixture;
    _FakeBridgeCounters? counters;

    setUpAll(() async {
      fixture = await _buildFakeChatBridge();
      final active = fixture;
      if (active != null) {
        counters = _FakeBridgeCounters(
          ffi.DynamicLibrary.open(active.libraryPath),
        );
      }
    });

    tearDownAll(() async {
      final active = fixture;
      if (active != null && await active.directory.exists()) {
        await active.directory.delete(recursive: true);
      }
    });

    test(
      'preserves a successful legacy formatter without creating a plan',
      () async {
        final active = fixture;
        final activeCounters = counters;
        if (active == null || activeCounters == null) {
          markTestSkipped('A C compiler is not available for the fake bridge.');
          return;
        }
        activeCounters.reset();

        final prompt = await LlamaChatTemplate.format(
          LlamaModelConfig(
            modelPath: 'legacy.gguf',
            nativeLibraryPath: active.libraryPath,
          ),
          <ChatMessage>[ChatMessage.user('hello')],
        );

        expect(prompt, 'legacy:assistant');
        expect(activeCounters.applyCalls(), 1);
        expect(activeCounters.planCalls(), 0);
      },
    );

    test('explicit thinking control uses the Jinja plan', () async {
      final active = fixture;
      final activeCounters = counters;
      if (active == null || activeCounters == null) {
        markTestSkipped('A C compiler is not available for the fake bridge.');
        return;
      }
      activeCounters.reset();
      final engine = await LlamaEngine.load(
        LlamaModelConfig(
          modelPath: 'thinking.gguf',
          nativeLibraryPath: active.libraryPath,
        ),
      );
      try {
        final chunks = await engine
            .chat(
              messages: <ChatMessage>[ChatMessage.user('hello')],
              config: const GenerationConfig(
                maxTokens: 1,
                enableThinking: false,
              ),
            )
            .toList();

        expect(chunks.single.text, 'terminal');
        expect(chunks.single.assistantMessage, isNull);
        expect(activeCounters.applyCalls(), 0);
        expect(activeCounters.planCalls(), 1);
      } finally {
        await engine.close();
      }
    });

    test('uses the Jinja plan after unsupported legacy formatting', () async {
      final active = fixture;
      final activeCounters = counters;
      if (active == null || activeCounters == null) {
        markTestSkipped('A C compiler is not available for the fake bridge.');
        return;
      }
      activeCounters.reset();
      final config = LlamaModelConfig(
        modelPath: 'fallback.gguf',
        nativeLibraryPath: active.libraryPath,
      );
      final messages = <ChatMessage>[ChatMessage.user('hello')];

      expect(
        await LlamaChatTemplate.format(config, messages),
        'jinja:assistant',
      );
      expect(
        await LlamaChatTemplate.format(
          config,
          messages,
          addAssistantPrompt: false,
        ),
        'jinja:no-assistant',
      );
      expect(activeCounters.applyCalls(), 2);
      expect(activeCounters.planCalls(), 2);
    });

    test(
      'streams fallback chat with raw grammar and its custom root',
      () async {
        final active = fixture;
        final activeCounters = counters;
        if (active == null || activeCounters == null) {
          markTestSkipped('A C compiler is not available for the fake bridge.');
          return;
        }
        activeCounters.reset();
        final engine = await LlamaEngine.load(
          LlamaModelConfig(
            modelPath: 'fallback-grammar.gguf',
            nativeLibraryPath: active.libraryPath,
          ),
        );
        try {
          final chunks = await engine
              .chat(
                messages: <ChatMessage>[ChatMessage.user('hello')],
                config: const GenerationConfig(
                  maxTokens: 1,
                  grammar: 'answer ::= "ok"',
                  grammarRoot: 'answer',
                ),
              )
              .toList();

          expect(chunks, hasLength(1));
          expect(chunks.single.text, 'terminal');
          expect(chunks.single.isDone, isTrue);
          expect(chunks.single.assistantMessage, isNull);
          expect(activeCounters.applyCalls(), 1);
          expect(activeCounters.planCalls(), 1);
          expect(activeCounters.parseCalls(), 0);
        } finally {
          await engine.close();
        }
      },
    );

    test(
      'streams fallback chat with JSON-schema bytes and a chat plan',
      () async {
        final active = fixture;
        final activeCounters = counters;
        if (active == null || activeCounters == null) {
          markTestSkipped('A C compiler is not available for the fake bridge.');
          return;
        }
        activeCounters.reset();
        final engine = await LlamaEngine.load(
          LlamaModelConfig(
            modelPath: 'fallback-schema.gguf',
            nativeLibraryPath: active.libraryPath,
          ),
        );
        try {
          final chunks = await engine
              .chat(
                messages: <ChatMessage>[ChatMessage.user('hello')],
                config: GenerationConfig.jsonSchema(
                  maxTokens: 1,
                  schema: <String, Object?>{
                    'type': 'object',
                    'properties': <String, Object?>{
                      'answer': <String, Object?>{'type': 'string'},
                    },
                    'required': <String>['answer'],
                  },
                ),
              )
              .toList();

          expect(chunks, hasLength(1));
          expect(chunks.single.text, 'terminal');
          expect(chunks.single.isDone, isTrue);
          expect(chunks.single.assistantMessage, isNull);
          expect(activeCounters.applyCalls(), 1);
          expect(activeCounters.planCalls(), 1);
          expect(activeCounters.parseCalls(), 0);
        } finally {
          await engine.close();
        }
      },
    );

    test(
      'keeps media markers and payloads in message order during fallback',
      () async {
        final active = fixture;
        final activeCounters = counters;
        if (active == null || activeCounters == null) {
          markTestSkipped('A C compiler is not available for the fake bridge.');
          return;
        }
        activeCounters.reset();
        final engine = await LlamaEngine.load(
          LlamaModelConfig(
            modelPath: 'fallback-media.gguf',
            mmprojPath: 'mmproj.gguf',
            nativeLibraryPath: active.libraryPath,
          ),
        );
        try {
          final chunks = await engine
              .chat(
                messages: <ChatMessage>[
                  ChatMessage.content(
                    role: ChatRole.user,
                    parts: <ChatContentPart>[
                      const TextPart('left'),
                      ImagePart.fromBytes(Uint8List.fromList(<int>[1, 2, 3])),
                      const TextPart('middle'),
                      const AudioPart.fromFile('clip.wav'),
                      const TextPart('right'),
                    ],
                  ),
                ],
                config: const GenerationConfig(maxTokens: 1),
              )
              .toList();

          expect(chunks.single.text, 'terminal');
          expect(chunks.single.isDone, isTrue);
          expect(chunks.single.assistantMessage, isNull);
          expect(activeCounters.applyCalls(), 1);
          expect(activeCounters.planCalls(), 1);
          expect(activeCounters.parseCalls(), 0);
        } finally {
          await engine.close();
        }
      },
    );

    test(
      'propagates non-unsupported legacy errors without creating a plan',
      () async {
        final active = fixture;
        final activeCounters = counters;
        if (active == null || activeCounters == null) {
          markTestSkipped('A C compiler is not available for the fake bridge.');
          return;
        }
        activeCounters.reset();

        await expectLater(
          LlamaChatTemplate.format(
            LlamaModelConfig(
              modelPath: 'legacy-error.gguf',
              nativeLibraryPath: active.libraryPath,
            ),
            <ChatMessage>[ChatMessage.user('hello')],
          ),
          throwsA(
            isA<NativeBridgeException>().having(
              (error) => error.message,
              'message',
              contains('legacy formatter exploded'),
            ),
          ),
        );
        expect(activeCounters.applyCalls(), 1);
        expect(activeCounters.planCalls(), 0);
      },
    );
  });
}

final class _FakeBridgeCounters {
  _FakeBridgeCounters(ffi.DynamicLibrary library)
    : reset = library.lookupFunction<ffi.Void Function(), void Function()>(
        'fake_reset_counters',
      ),
      applyCalls = library
          .lookupFunction<ffi.Uint32 Function(), int Function()>(
            'fake_apply_calls',
          ),
      planCalls = library.lookupFunction<ffi.Uint32 Function(), int Function()>(
        'fake_plan_calls',
      ),
      parseCalls = library
          .lookupFunction<ffi.Uint32 Function(), int Function()>(
            'fake_parse_calls',
          );

  final void Function() reset;
  final int Function() applyCalls;
  final int Function() planCalls;
  final int Function() parseCalls;
}

final class _FakeBridgeFixture {
  const _FakeBridgeFixture({
    required this.directory,
    required this.libraryPath,
  });

  final Directory directory;
  final String libraryPath;
}

Future<_FakeBridgeFixture?> _buildFakeChatBridge() async {
  if (Platform.isWindows) {
    return null;
  }
  final directory = await Directory.systemTemp.createTemp(
    'fllamer_chat_fallback_',
  );
  final source = File(
    '${directory.path}${Platform.pathSeparator}chat_fallback.c',
  );
  final extension = Platform.isMacOS || Platform.isIOS ? '.dylib' : '.so';
  final output =
      '${directory.path}${Platform.pathSeparator}libchat_fallback$extension';
  try {
    await source.writeAsString(_fakeBridgeSource);
    final include = Directory('native/llama_dart_bridge/include').absolute.path;
    final arguments = Platform.isMacOS || Platform.isIOS
        ? <String>['-dynamiclib', source.path, '-I', include, '-o', output]
        : <String>[
            '-shared',
            '-fPIC',
            source.path,
            '-I',
            include,
            '-o',
            output,
          ];
    final result = await Process.run('cc', arguments);
    if (result.exitCode != 0) {
      throw StateError(
        'Fake bridge compilation failed (${result.exitCode}).\n'
        'stdout:\n${result.stdout}\n'
        'stderr:\n${result.stderr}',
      );
    }
    return _FakeBridgeFixture(directory: directory, libraryPath: output);
  } on ProcessException {
    await directory.delete(recursive: true);
    return null;
  } catch (_) {
    await directory.delete(recursive: true);
    rethrow;
  }
}

const _fakeBridgeSource = r'''
#include "llama_dart.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum fake_mode {
  MODE_FALLBACK = 0,
  MODE_LEGACY = 1,
  MODE_ERROR = 2,
  MODE_GRAMMAR = 3,
  MODE_SCHEMA = 4,
  MODE_MEDIA = 5,
  MODE_THINKING = 6,
};

static const char *last_error = "";
static uintptr_t fake_model_storage;
static uintptr_t fake_context_storage;
static uintptr_t fake_generation_storage;
static enum fake_mode current_mode = MODE_FALLBACK;
static uint32_t apply_calls;
static uint32_t plan_calls;
static uint32_t parse_calls;
static int generation_active;

static llama_dart_result fail(llama_dart_result result, const char *message) {
  last_error = message;
  return result;
}

static int bytes_equal(const uint8_t *data, size_t size, const char *expected) {
  const size_t expected_size = strlen(expected);
  return size == expected_size && data != NULL &&
      memcmp(data, expected, expected_size) == 0;
}

static int bytes_contain(const uint8_t *data, size_t size,
                         const char *needle) {
  const size_t needle_size = strlen(needle);
  if (data == NULL || needle_size == 0 || needle_size > size) {
    return 0;
  }
  for (size_t i = 0; i + needle_size <= size; ++i) {
    if (memcmp(data + i, needle, needle_size) == 0) {
      return 1;
    }
  }
  return 0;
}

static llama_dart_result copy_buffer(const char *text,
                                     llama_dart_buffer *out) {
  const size_t size = strlen(text);
  out->data = (uint8_t *)malloc(size);
  if (out->data == NULL) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
  }
  memcpy(out->data, text, size);
  out->size = size;
  last_error = "";
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT void fake_reset_counters(void) {
  apply_calls = 0;
  plan_calls = 0;
  parse_calls = 0;
}

LLAMA_DART_EXPORT uint32_t fake_apply_calls(void) { return apply_calls; }
LLAMA_DART_EXPORT uint32_t fake_plan_calls(void) { return plan_calls; }
LLAMA_DART_EXPORT uint32_t fake_parse_calls(void) { return parse_calls; }

LLAMA_DART_EXPORT uint32_t llama_dart_abi_version(void) {
  return LLAMA_DART_ABI_VERSION;
}

LLAMA_DART_EXPORT const char *llama_dart_multimodal_marker(void) {
  return "<__media__>";
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_model_load(
    const llama_dart_model_load_config *config, llama_dart_model **out_model) {
  if (config == NULL || out_model == NULL ||
      config->model_path_data == NULL || config->model_path_size == 0) {
    return fail(LLAMA_DART_ERROR_MODEL_LOAD, "invalid model config");
  }
  if (bytes_contain(config->model_path_data, config->model_path_size,
                    "legacy-error")) {
    current_mode = MODE_ERROR;
  } else if (bytes_contain(config->model_path_data, config->model_path_size,
                           "thinking")) {
    current_mode = MODE_THINKING;
  } else if (bytes_contain(config->model_path_data, config->model_path_size,
                           "legacy")) {
    current_mode = MODE_LEGACY;
  } else if (bytes_contain(config->model_path_data, config->model_path_size,
                           "grammar")) {
    current_mode = MODE_GRAMMAR;
  } else if (bytes_contain(config->model_path_data, config->model_path_size,
                           "schema")) {
    current_mode = MODE_SCHEMA;
  } else if (bytes_contain(config->model_path_data, config->model_path_size,
                           "media")) {
    current_mode = MODE_MEDIA;
  } else {
    current_mode = MODE_FALLBACK;
  }
  *out_model = (llama_dart_model *)&fake_model_storage;
  last_error = "";
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT void llama_dart_model_free(llama_dart_model *model) {
  (void)model;
  last_error = "";
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_model_apply_chat_template(
    const llama_dart_model *model, const llama_dart_chat_message *messages,
    size_t message_count, uint8_t add_assistant_prompt,
    llama_dart_buffer *out_prompt) {
  (void)model;
  ++apply_calls;
  if (messages == NULL || message_count != 1 || out_prompt == NULL) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "invalid messages");
  }
  if (current_mode == MODE_LEGACY || current_mode == MODE_THINKING) {
    return copy_buffer(add_assistant_prompt != 0
        ? "legacy:assistant"
        : "legacy:no-assistant", out_prompt);
  }
  if (current_mode == MODE_ERROR) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "legacy formatter exploded");
  }
  return fail(LLAMA_DART_ERROR_UNSUPPORTED,
              "legacy formatter cannot render Jinja");
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_model_create_chat_plan(
    const llama_dart_model *model, const uint8_t *request_data,
    size_t request_size, llama_dart_buffer *out_plan) {
  (void)model;
  ++plan_calls;
  if (current_mode == MODE_LEGACY || current_mode == MODE_ERROR) {
    return fail(LLAMA_DART_ERROR_INTERNAL,
                "chat plan must not be created in this mode");
  }
  if (request_data == NULL || request_size == 0 || out_plan == NULL) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "invalid chat request");
  }
  char *request = (char *)malloc(request_size + 1);
  if (request == NULL) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
  }
  memcpy(request, request_data, request_size);
  request[request_size] = '\0';
  const int add_assistant =
      strstr(request, "\"add_generation_prompt\":true") != NULL;
  const int no_tools = strstr(request, "\"tools\":[]") != NULL;
  const int thinking_disabled =
      strstr(request, "\"enable_thinking\":false") != NULL;
  const int media_request =
      strstr(request, "left<__media__>middle<__media__>right") != NULL;
  free(request);
  if (!no_tools) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "ordinary fallback injected tool metadata");
  }
  if (current_mode == MODE_THINKING && !thinking_disabled) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "thinking control was not forwarded");
  }
  if (current_mode == MODE_MEDIA && !media_request) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "media markers were reordered before planning");
  }

  const char *prompt = current_mode == MODE_MEDIA
      ? "jinja:left<__media__>middle<__media__>right:assistant"
      : add_assistant ? "jinja:assistant" : "jinja:no-assistant";
  char plan[1024];
  const int size = snprintf(
      plan, sizeof(plan),
      "{\"version\":1,\"prompt\":\"%s\",\"grammar\":\"\","
      "\"grammar_lazy\":false,\"grammar_triggers\":[],"
      "\"preserved_tokens\":[],\"additional_stops\":[],"
      "\"generation_prompt\":\"\"}", prompt);
  if (size <= 0 || (size_t)size >= sizeof(plan)) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "chat plan buffer overflow");
  }
  return copy_buffer(plan, out_plan);
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_chat_parse_output(
    const uint8_t *plan_data, size_t plan_size, const uint8_t *output_data,
    size_t output_size, llama_dart_buffer *out_message_json) {
  (void)plan_data;
  (void)plan_size;
  (void)output_data;
  (void)output_size;
  (void)out_message_json;
  ++parse_calls;
  return fail(LLAMA_DART_ERROR_INTERNAL,
              "plain fallback output must not be parsed");
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_context_create(
    llama_dart_model *model, const llama_dart_context_config *config,
    llama_dart_context **out_context) {
  (void)model;
  if (config == NULL || out_context == NULL) {
    return fail(LLAMA_DART_ERROR_CONTEXT_CREATE, "invalid context config");
  }
  *out_context = (llama_dart_context *)&fake_context_storage;
  last_error = "";
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT void llama_dart_context_free(llama_dart_context *context) {
  (void)context;
  last_error = "";
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_context_reset(
    llama_dart_context *context) {
  (void)context;
  generation_active = 0;
  last_error = "";
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_context_cancel(
    llama_dart_context *context) {
  (void)context;
  last_error = "";
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_generation_start(
    llama_dart_context *context, const llama_dart_completion_config *config,
    llama_dart_generation **out_generation) {
  (void)context;
  if (config == NULL || out_generation == NULL || generation_active ||
      config->chat_plan_data == NULL || config->chat_plan_size == 0 ||
      config->parse_special != 1 ||
      config->add_special != LLAMA_DART_ADD_SPECIAL_IF_CONTEXT_EMPTY) {
    return fail(LLAMA_DART_ERROR_GENERATION,
                "fallback generation did not receive its chat plan");
  }
  if (current_mode == MODE_GRAMMAR) {
    if (!bytes_equal(config->prompt_data, config->prompt_size,
                     "jinja:assistant") ||
        !bytes_equal(config->grammar_data, config->grammar_size,
                     "answer ::= \"ok\"") ||
        !bytes_equal(config->grammar_root_data, config->grammar_root_size,
                     "answer") ||
        config->json_schema_size != 0) {
      return fail(LLAMA_DART_ERROR_GENERATION,
                  "raw grammar or grammar root was not forwarded");
    }
  } else if (current_mode == MODE_SCHEMA) {
    static const char schema[] =
        "{\"type\":\"object\",\"properties\":{\"answer\":{"
        "\"type\":\"string\"}},\"required\":[\"answer\"]}";
    if (!bytes_equal(config->prompt_data, config->prompt_size,
                     "jinja:assistant") ||
        config->grammar_size != 0 || config->grammar_root_size != 0 ||
        !bytes_equal(config->json_schema_data, config->json_schema_size,
                     schema)) {
      return fail(LLAMA_DART_ERROR_GENERATION,
                  "JSON schema was not forwarded with the chat plan");
    }
  } else if (current_mode == MODE_MEDIA) {
    static const uint8_t image[] = {1, 2, 3};
    if (!bytes_equal(config->prompt_data, config->prompt_size,
                     "jinja:left<__media__>middle<__media__>right:assistant") ||
        config->media_input_count != 2 || config->media_inputs == NULL) {
      return fail(LLAMA_DART_ERROR_GENERATION,
                  "media prompt or payload count was not preserved");
    }
    const llama_dart_media_input *media = config->media_inputs;
    if (media[0].type != LLAMA_DART_MEDIA_IMAGE ||
        media[0].content_size != sizeof(image) ||
        memcmp(media[0].content_data, image, sizeof(image)) != 0 ||
        media[0].path_data != NULL ||
        media[1].type != LLAMA_DART_MEDIA_AUDIO ||
        !bytes_equal(media[1].path_data, media[1].path_size, "clip.wav") ||
        media[1].content_data != NULL) {
      return fail(LLAMA_DART_ERROR_GENERATION,
                  "media payload order was not preserved");
    }
  }
  generation_active = 1;
  *out_generation = (llama_dart_generation *)&fake_generation_storage;
  last_error = "";
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_generation_next(
    llama_dart_generation *generation, llama_dart_buffer *out_text,
    llama_dart_completion_stats *out_stats, uint8_t *out_done) {
  (void)generation;
  if (!generation_active || out_text == NULL || out_stats == NULL ||
      out_done == NULL) {
    return fail(LLAMA_DART_ERROR_GENERATION, "invalid generation step");
  }
  const llama_dart_result copied = copy_buffer("terminal", out_text);
  if (copied != LLAMA_DART_SUCCESS) {
    return copied;
  }
  memset(out_stats, 0, sizeof(*out_stats));
  out_stats->struct_size = sizeof(*out_stats);
  out_stats->prompt_tokens = 3;
  out_stats->generated_tokens = 1;
  *out_done = 1;
  last_error = "";
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT void llama_dart_generation_free(
    llama_dart_generation *generation) {
  (void)generation;
  generation_active = 0;
  last_error = "";
}

LLAMA_DART_EXPORT void llama_dart_buffer_free(uint8_t *data) {
  free(data);
  last_error = "";
}

LLAMA_DART_EXPORT const char *llama_dart_last_error_message(void) {
  return last_error;
}

LLAMA_DART_EXPORT void llama_dart_clear_last_error(void) {
  last_error = "";
}
''';
