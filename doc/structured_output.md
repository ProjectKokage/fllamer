# Structured output

`fllamer` supports three structured-output paths:

```dart
const GenerationConfig(
  grammar: 'root ::= "yes" | "no"',
  grammarRoot: 'root',
)

const GenerationConfig.jsonMode()

GenerationConfig.jsonSchema(
  schema: <String, Object?>{
    'type': 'object',
    'properties': <String, Object?>{
      'answer': <String, Object?>{'type': 'string'},
    },
    'required': <Object?>['answer'],
    'additionalProperties': false,
  },
)
```

Raw `grammar` is passed to the pinned upstream grammar sampler. JSON mode uses
the package's pinned JSON GBNF. JSON Schema mode snapshots the input map, sends
ordered JSON to the worker isolate, and calls pinned upstream
`json_schema_to_grammar` before sampler creation. Schema conversion therefore
does not block the Flutter UI isolate.

The same request-owned grammar or schema remains authoritative when ordinary
chat has to fall back from the legacy formatter to the model's Jinja chat
plan. The plan still contributes its generation prefix and stop strings.

The pinned converter supports common object, array, string, numeric, union,
`const`/`enum`, local `$defs`/`$ref`, and composition constraints. Unsupported
schemas return `UnsupportedFeatureException`; malformed JSON or conflicting raw
grammar/schema input returns a bridge validation error. External
references are not resolved and must be bundled into a local schema first.

Active tools and response-format grammar are mutually exclusive because the
model template owns the tool-call grammar. Use `toolChoice: LlamaToolChoice.none()`
when a request must apply structured output without allowing a tool call.

`llamaJsonSchemaGrammar()` remains available as a synchronous compatibility
helper. It opens the bundled native bridge and delegates to the same pinned
upstream converter; `GenerationConfig.jsonSchema` remains the non-blocking
worker-isolate path for application generation.

Grammar enforcement constrains syntax, not model quality or semantic truth.
Apps should still parse and validate generated JSON before using it.
