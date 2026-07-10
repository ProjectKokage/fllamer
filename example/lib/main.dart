import 'dart:async';
import 'dart:io';

import 'package:file_selector/file_selector.dart';
import 'package:fllamer/fllamer.dart';
import 'package:flutter/material.dart';

const _ggufFiles = XTypeGroup(
  label: 'GGUF models',
  extensions: <String>['gguf'],
  mimeTypes: <String>['application/octet-stream'],
  uniformTypeIdentifiers: <String>['public.data'],
);

const _imageFiles = XTypeGroup(
  label: 'Images',
  extensions: <String>['jpg', 'jpeg', 'png', 'webp', 'bmp'],
  mimeTypes: <String>['image/*'],
  uniformTypeIdentifiers: <String>['public.image'],
);

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const FllamerExampleApp());
}

class FllamerExampleApp extends StatelessWidget {
  const FllamerExampleApp({super.key});

  @override
  Widget build(BuildContext context) {
    final colorScheme = ColorScheme.fromSeed(seedColor: const Color(0xFF006A65))
        .copyWith(
          secondary: const Color(0xFF8A5700),
          surface: const Color(0xFFFCFCF9),
        );
    return MaterialApp(
      title: 'fllamer',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: colorScheme,
        useMaterial3: true,
        scaffoldBackgroundColor: const Color(0xFFF5F7F4),
        cardTheme: const CardThemeData(
          elevation: 0,
          margin: EdgeInsets.zero,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.all(Radius.circular(6)),
          ),
        ),
        inputDecorationTheme: const InputDecorationTheme(
          border: OutlineInputBorder(
            borderRadius: BorderRadius.all(Radius.circular(6)),
          ),
        ),
      ),
      home: const InferenceWorkspace(),
    );
  }
}

class InferenceWorkspace extends StatefulWidget {
  const InferenceWorkspace({super.key});

  @override
  State<InferenceWorkspace> createState() => _InferenceWorkspaceState();
}

class _InferenceWorkspaceState extends State<InferenceWorkspace> {
  final _promptController = TextEditingController();
  final _scrollController = ScrollController();
  final _transcript = <_TranscriptEntry>[];

  late final LlamaRuntimeCapabilities _runtimeCapabilities;
  _SelectedFile? _modelFile;
  _SelectedFile? _projectorFile;
  _SelectedFile? _pendingImage;
  LlamaEngine? _engine;
  LlamaContextInfo? _contextInfo;
  StreamSubscription<GenerationChunk>? _generationSubscription;
  GenerationTelemetry? _lastTelemetry;
  String? _error;
  bool _isLoading = false;
  bool _isGenerating = false;
  int _generationId = 0;
  int? _streamingEntryIndex;

  @override
  void initState() {
    super.initState();
    _runtimeCapabilities = LlamaRuntime.currentCapabilities();
    _promptController.addListener(_onPromptChanged);
  }

  @override
  void dispose() {
    _generationId++;
    final subscription = _generationSubscription;
    if (subscription != null) {
      unawaited(subscription.cancel());
    }
    final engine = _engine;
    if (engine != null) {
      unawaited(engine.close());
    }
    _promptController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  bool get _canSend {
    return _engine != null &&
        !_isLoading &&
        !_isGenerating &&
        (_promptController.text.trim().isNotEmpty || _pendingImage != null);
  }

  void _onPromptChanged() {
    if (mounted) {
      setState(() {});
    }
  }

  Future<void> _selectGguf({required bool projector}) async {
    try {
      final file = await openFile(
        acceptedTypeGroups: const <XTypeGroup>[_ggufFiles],
      );
      if (!mounted || file == null) {
        return;
      }
      if (file.path.isEmpty) {
        throw const FileSystemException('The selected file has no local path.');
      }
      setState(() {
        final selected = _SelectedFile(path: file.path, name: file.name);
        if (projector) {
          _projectorFile = selected;
        } else {
          _modelFile = selected;
        }
        _error = null;
      });
    } catch (error) {
      _showError(error);
    }
  }

  Future<void> _selectImage() async {
    try {
      final file = await openFile(
        acceptedTypeGroups: const <XTypeGroup>[_imageFiles],
      );
      if (!mounted || file == null) {
        return;
      }
      if (file.path.isEmpty) {
        throw const FileSystemException(
          'The selected image has no local path.',
        );
      }
      setState(() {
        _pendingImage = _SelectedFile(path: file.path, name: file.name);
        _error = null;
      });
    } catch (error) {
      _showError(error);
    }
  }

  Future<void> _loadModel() async {
    final modelFile = _modelFile;
    if (modelFile == null || _isLoading) {
      return;
    }
    setState(() {
      _isLoading = true;
      _error = null;
    });

    LlamaEngine? engine;
    try {
      await LlamaModel.validateFile(modelFile.path);
      final projectorFile = _projectorFile;
      if (projectorFile != null) {
        await LlamaModel.validateFile(projectorFile.path);
      }
      engine = await LlamaEngine.load(
        LlamaModelConfig(
          modelPath: modelFile.path,
          mmprojPath: projectorFile?.path,
          contextSize: 4096,
          batchSize: 512,
          gpu: const GpuConfig.auto(),
        ),
      );
      final contextInfo = await engine.contextInfo();
      if (!mounted) {
        await engine.close();
        return;
      }
      setState(() {
        _engine = engine;
        _contextInfo = contextInfo;
        _transcript.clear();
        _pendingImage = null;
        _lastTelemetry = null;
        _isLoading = false;
      });
    } catch (error) {
      if (engine != null) {
        await engine.close();
      }
      if (mounted) {
        setState(() {
          _isLoading = false;
          _error = _errorText(error);
        });
      }
    }
  }

  Future<void> _unloadModel() async {
    final engine = _engine;
    if (engine == null || _isLoading) {
      return;
    }
    setState(() {
      _isLoading = true;
      _error = null;
    });
    try {
      await _cancelGeneration(markCancelled: false);
      await engine.close();
      if (mounted) {
        setState(() {
          _engine = null;
          _contextInfo = null;
          _transcript.clear();
          _pendingImage = null;
          _lastTelemetry = null;
          _isLoading = false;
        });
      }
    } catch (error) {
      if (mounted) {
        setState(() {
          _isLoading = false;
          _error = _errorText(error);
        });
      }
    }
  }

  Future<void> _sendPrompt() async {
    final engine = _engine;
    final contextInfo = _contextInfo;
    if (engine == null || contextInfo == null || !_canSend) {
      return;
    }
    final image = _pendingImage;
    if (image != null && !contextInfo.supportsVision) {
      _showError(
        const UnsupportedFeatureException(
          'The loaded projector does not support image input.',
        ),
      );
      return;
    }

    var prompt = _promptController.text.trim();
    if (prompt.isEmpty && image != null) {
      prompt = 'Describe this image.';
    }
    final userEntry = _TranscriptEntry(
      role: ChatRole.user,
      text: prompt,
      attachmentPath: image?.path,
      attachmentName: image?.name,
    );
    final requestEntries = <_TranscriptEntry>[..._transcript, userEntry];
    final messages = requestEntries.map(_toChatMessage).toList();

    setState(() {
      _isGenerating = true;
      _error = null;
      _lastTelemetry = null;
    });
    try {
      await engine.reset();
      if (!mounted) {
        return;
      }
      _promptController.clear();
      final assistantEntry = _TranscriptEntry(
        role: ChatRole.assistant,
        text: '',
        isStreaming: true,
      );
      setState(() {
        _transcript
          ..add(userEntry)
          ..add(assistantEntry);
        _pendingImage = null;
        _streamingEntryIndex = _transcript.length - 1;
      });
      _scrollToEnd();

      final generationId = ++_generationId;
      final stream = engine.chat(
        messages: messages,
        config: const GenerationConfig(
          maxTokens: 256,
          temperature: 0.7,
          streamChunkTokens: 4,
        ),
      );
      _generationSubscription = stream.listen(
        (chunk) => _handleGenerationChunk(generationId, chunk),
        onError: (Object error, StackTrace stackTrace) {
          _handleGenerationError(generationId, error);
        },
        onDone: () => _handleGenerationDone(generationId),
        cancelOnError: true,
      );
    } catch (error) {
      if (mounted) {
        final index = _streamingEntryIndex;
        setState(() {
          if (index != null && index < _transcript.length) {
            if (_transcript[index].text.isEmpty) {
              _transcript.removeAt(index);
            } else {
              _transcript[index].isStreaming = false;
            }
          }
          _isGenerating = false;
          _streamingEntryIndex = null;
          _error = _errorText(error);
        });
      }
    }
  }

  void _handleGenerationChunk(int generationId, GenerationChunk chunk) {
    if (!mounted || generationId != _generationId) {
      return;
    }
    final index = _streamingEntryIndex;
    if (index == null || index >= _transcript.length) {
      return;
    }
    setState(() {
      _transcript[index].text += chunk.text;
      _lastTelemetry = chunk.telemetry ?? _lastTelemetry;
      if (chunk.isDone) {
        _transcript[index].isStreaming = false;
      }
    });
    _scrollToEnd();
  }

  void _handleGenerationDone(int generationId) {
    if (!mounted || generationId != _generationId) {
      return;
    }
    final index = _streamingEntryIndex;
    setState(() {
      if (index != null && index < _transcript.length) {
        _transcript[index].isStreaming = false;
      }
      _isGenerating = false;
      _streamingEntryIndex = null;
      _generationSubscription = null;
    });
  }

  void _handleGenerationError(int generationId, Object error) {
    if (!mounted || generationId != _generationId) {
      return;
    }
    final index = _streamingEntryIndex;
    setState(() {
      if (index != null && index < _transcript.length) {
        if (_transcript[index].text.isEmpty) {
          _transcript.removeAt(index);
        } else {
          _transcript[index].isStreaming = false;
        }
      }
      _generationId++;
      _isGenerating = false;
      _streamingEntryIndex = null;
      _generationSubscription = null;
      _error = _errorText(error);
    });
  }

  Future<void> _cancelGeneration({bool markCancelled = true}) async {
    final subscription = _generationSubscription;
    if (subscription == null && !_isGenerating) {
      return;
    }
    _generationId++;
    _generationSubscription = null;
    await subscription?.cancel();
    if (!mounted) {
      return;
    }
    final index = _streamingEntryIndex;
    setState(() {
      if (index != null && index < _transcript.length) {
        final entry = _transcript[index];
        entry.isStreaming = false;
        if (markCancelled && entry.text.isEmpty) {
          entry.text = 'Generation cancelled.';
        }
      }
      _streamingEntryIndex = null;
      _isGenerating = false;
    });
  }

  Future<void> _clearConversation() async {
    final engine = _engine;
    if (engine == null || _isLoading) {
      return;
    }
    try {
      await _cancelGeneration(markCancelled: false);
      await engine.reset();
      if (mounted) {
        setState(() {
          _transcript.clear();
          _lastTelemetry = null;
          _error = null;
        });
      }
    } catch (error) {
      _showError(error);
    }
  }

  ChatMessage _toChatMessage(_TranscriptEntry entry) {
    if (entry.role == ChatRole.user && entry.attachmentPath != null) {
      return ChatMessage.content(
        role: ChatRole.user,
        parts: <ChatContentPart>[
          TextPart(entry.text),
          ImagePart.fromFile(entry.attachmentPath!),
        ],
      );
    }
    return switch (entry.role) {
      ChatRole.system => ChatMessage.system(entry.text),
      ChatRole.user => ChatMessage.user(entry.text),
      ChatRole.assistant => ChatMessage.assistant(entry.text),
      ChatRole.tool => ChatMessage.tool(entry.text),
    };
  }

  void _showError(Object error) {
    if (mounted) {
      setState(() => _error = _errorText(error));
    }
  }

  void _scrollToEnd() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted || !_scrollController.hasClients) {
        return;
      }
      _scrollController.animateTo(
        _scrollController.position.maxScrollExtent,
        duration: const Duration(milliseconds: 180),
        curve: Curves.easeOut,
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    final contextInfo = _contextInfo;
    return Scaffold(
      appBar: AppBar(
        title: const Row(
          mainAxisSize: MainAxisSize.min,
          children: <Widget>[
            Icon(Icons.memory),
            SizedBox(width: 10),
            Text('fllamer'),
          ],
        ),
        actions: <Widget>[
          _RuntimeStatus(capabilities: _runtimeCapabilities),
          const SizedBox(width: 12),
        ],
      ),
      body: SafeArea(
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 1000),
            child: Column(
              children: <Widget>[
                if (_engine == null)
                  _ModelSetup(
                    modelFile: _modelFile,
                    projectorFile: _projectorFile,
                    isLoading: _isLoading,
                    onSelectModel: () => _selectGguf(projector: false),
                    onSelectProjector: () => _selectGguf(projector: true),
                    onClearProjector: () {
                      setState(() => _projectorFile = null);
                    },
                    onLoad: _modelFile == null ? null : _loadModel,
                  )
                else
                  _LoadedModelBar(
                    modelName: _modelFile?.name ?? 'Model',
                    contextInfo: contextInfo!,
                    isLoading: _isLoading,
                    onClear: _clearConversation,
                    onUnload: _unloadModel,
                  ),
                if (_isLoading) const LinearProgressIndicator(minHeight: 2),
                if (_error case final error?)
                  _ErrorBar(
                    message: error,
                    onDismiss: () => setState(() => _error = null),
                  ),
                Expanded(
                  child: _TranscriptView(
                    entries: _transcript,
                    scrollController: _scrollController,
                  ),
                ),
                if (_lastTelemetry case final telemetry?)
                  _TelemetryBar(telemetry: telemetry),
                _Composer(
                  controller: _promptController,
                  enabled: _engine != null && !_isLoading,
                  canAttachImage: contextInfo?.supportsVision ?? false,
                  canSend: _canSend,
                  isGenerating: _isGenerating,
                  pendingImage: _pendingImage,
                  onAttachImage: _selectImage,
                  onRemoveImage: () => setState(() => _pendingImage = null),
                  onSend: _sendPrompt,
                  onCancel: _cancelGeneration,
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _RuntimeStatus extends StatelessWidget {
  const _RuntimeStatus({required this.capabilities});

  final LlamaRuntimeCapabilities capabilities;

  @override
  Widget build(BuildContext context) {
    final available = capabilities.nativeBridgeAvailable;
    final label = available
        ? 'Native bridge ABI ${capabilities.bridgeAbiVersion}'
        : 'Native bridge unavailable';
    final showLabel = MediaQuery.sizeOf(context).width >= 600;
    return Tooltip(
      message: label,
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: <Widget>[
          Icon(
            available ? Icons.check_circle : Icons.error_outline,
            size: 18,
            color: available
                ? const Color(0xFF2E7D32)
                : Theme.of(context).colorScheme.error,
          ),
          if (showLabel) ...<Widget>[
            const SizedBox(width: 6),
            Text(label, style: Theme.of(context).textTheme.labelMedium),
          ],
        ],
      ),
    );
  }
}

class _ModelSetup extends StatelessWidget {
  const _ModelSetup({
    required this.modelFile,
    required this.projectorFile,
    required this.isLoading,
    required this.onSelectModel,
    required this.onSelectProjector,
    required this.onClearProjector,
    required this.onLoad,
  });

  final _SelectedFile? modelFile;
  final _SelectedFile? projectorFile;
  final bool isLoading;
  final VoidCallback onSelectModel;
  final VoidCallback onSelectProjector;
  final VoidCallback onClearProjector;
  final VoidCallback? onLoad;

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        border: Border(
          bottom: BorderSide(
            color: Theme.of(context).colorScheme.outlineVariant,
          ),
        ),
      ),
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 12, 16, 14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: <Widget>[
            Text('Local model', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            _FileRow(
              key: const ValueKey<String>('model-file-row'),
              icon: Icons.data_object,
              label: 'Model',
              file: modelFile,
              selectTooltip: 'Select GGUF model',
              onSelect: isLoading ? null : onSelectModel,
            ),
            const Divider(height: 1),
            _FileRow(
              key: const ValueKey<String>('projector-file-row'),
              icon: Icons.visibility_outlined,
              label: 'Projector',
              file: projectorFile,
              selectTooltip: 'Select multimodal projector',
              onSelect: isLoading ? null : onSelectProjector,
              onClear: projectorFile == null || isLoading
                  ? null
                  : onClearProjector,
            ),
            const SizedBox(height: 10),
            Align(
              alignment: Alignment.centerRight,
              child: FilledButton.icon(
                key: const ValueKey<String>('load-model-button'),
                onPressed: isLoading ? null : onLoad,
                icon: const Icon(Icons.play_arrow),
                label: const Text('Load model'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _FileRow extends StatelessWidget {
  const _FileRow({
    super.key,
    required this.icon,
    required this.label,
    required this.file,
    required this.selectTooltip,
    required this.onSelect,
    this.onClear,
  });

  final IconData icon;
  final String label;
  final _SelectedFile? file;
  final String selectTooltip;
  final VoidCallback? onSelect;
  final VoidCallback? onClear;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 58,
      child: Row(
        children: <Widget>[
          Icon(icon, size: 21),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                Text(label, style: Theme.of(context).textTheme.labelLarge),
                Text(
                  file?.name ?? 'Not selected',
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: Theme.of(context).textTheme.bodySmall,
                ),
              ],
            ),
          ),
          if (onClear != null)
            IconButton(
              tooltip: 'Clear $label',
              onPressed: onClear,
              icon: const Icon(Icons.close),
            ),
          IconButton.filledTonal(
            key: ValueKey<String>('select-${label.toLowerCase()}-button'),
            tooltip: selectTooltip,
            onPressed: onSelect,
            icon: const Icon(Icons.folder_open),
          ),
        ],
      ),
    );
  }
}

class _LoadedModelBar extends StatelessWidget {
  const _LoadedModelBar({
    required this.modelName,
    required this.contextInfo,
    required this.isLoading,
    required this.onClear,
    required this.onUnload,
  });

  final String modelName;
  final LlamaContextInfo contextInfo;
  final bool isLoading;
  final VoidCallback onClear;
  final VoidCallback onUnload;

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 62,
      padding: const EdgeInsets.symmetric(horizontal: 16),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        border: Border(
          bottom: BorderSide(
            color: Theme.of(context).colorScheme.outlineVariant,
          ),
        ),
      ),
      child: Row(
        children: <Widget>[
          const Icon(Icons.memory, size: 21),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                Text(
                  modelName,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: Theme.of(context).textTheme.labelLarge,
                ),
                Text(
                  '${contextInfo.gpuBackend.name.toUpperCase()} | '
                  '${contextInfo.contextSize} context',
                  style: Theme.of(context).textTheme.bodySmall,
                ),
              ],
            ),
          ),
          if (contextInfo.supportsVision)
            const Tooltip(message: 'Image input', child: Icon(Icons.image)),
          if (contextInfo.supportsAudio) ...<Widget>[
            const SizedBox(width: 8),
            const Tooltip(message: 'Audio input', child: Icon(Icons.mic)),
          ],
          const SizedBox(width: 8),
          IconButton(
            tooltip: 'Clear conversation',
            onPressed: isLoading ? null : onClear,
            icon: const Icon(Icons.delete_sweep_outlined),
          ),
          IconButton(
            tooltip: 'Unload model',
            onPressed: isLoading ? null : onUnload,
            icon: const Icon(Icons.power_settings_new),
          ),
        ],
      ),
    );
  }
}

class _ErrorBar extends StatelessWidget {
  const _ErrorBar({required this.message, required this.onDismiss});

  final String message;
  final VoidCallback onDismiss;

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    return Container(
      color: colors.errorContainer,
      padding: const EdgeInsets.only(left: 16),
      child: Row(
        children: <Widget>[
          Icon(Icons.error_outline, color: colors.onErrorContainer),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              message,
              maxLines: 3,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(color: colors.onErrorContainer),
            ),
          ),
          IconButton(
            tooltip: 'Dismiss error',
            onPressed: onDismiss,
            icon: const Icon(Icons.close),
          ),
        ],
      ),
    );
  }
}

class _TranscriptView extends StatelessWidget {
  const _TranscriptView({
    required this.entries,
    required this.scrollController,
  });

  final List<_TranscriptEntry> entries;
  final ScrollController scrollController;

  @override
  Widget build(BuildContext context) {
    if (entries.isEmpty) {
      return const Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: <Widget>[
            Icon(Icons.forum_outlined, size: 36),
            SizedBox(height: 10),
            Text('Local session'),
          ],
        ),
      );
    }
    return ListView.separated(
      controller: scrollController,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 18),
      itemCount: entries.length,
      separatorBuilder: (_, _) => const SizedBox(height: 10),
      itemBuilder: (context, index) => _MessageBubble(entry: entries[index]),
    );
  }
}

class _MessageBubble extends StatelessWidget {
  const _MessageBubble({required this.entry});

  final _TranscriptEntry entry;

  @override
  Widget build(BuildContext context) {
    final isUser = entry.role == ChatRole.user;
    final colors = Theme.of(context).colorScheme;
    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 720),
        child: DecoratedBox(
          decoration: BoxDecoration(
            color: isUser ? colors.secondaryContainer : colors.surface,
            border: Border.all(color: colors.outlineVariant),
            borderRadius: BorderRadius.circular(6),
          ),
          child: Padding(
            padding: const EdgeInsets.all(12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                Row(
                  mainAxisSize: MainAxisSize.min,
                  children: <Widget>[
                    Icon(
                      isUser ? Icons.person_outline : Icons.memory,
                      size: 17,
                    ),
                    const SizedBox(width: 6),
                    Text(
                      isUser ? 'You' : 'Model',
                      style: Theme.of(context).textTheme.labelMedium,
                    ),
                  ],
                ),
                if (entry.attachmentPath case final path?) ...<Widget>[
                  const SizedBox(height: 10),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(4),
                    child: Image.file(
                      File(path),
                      width: 240,
                      height: 150,
                      fit: BoxFit.cover,
                      errorBuilder: (_, _, _) => Container(
                        width: 240,
                        height: 72,
                        alignment: Alignment.center,
                        color: colors.surfaceContainerHighest,
                        child: Text(entry.attachmentName ?? 'Image'),
                      ),
                    ),
                  ),
                ],
                if (entry.text.isNotEmpty) ...<Widget>[
                  const SizedBox(height: 8),
                  SelectableText(entry.text),
                ],
                if (entry.isStreaming) ...<Widget>[
                  const SizedBox(height: 10),
                  const SizedBox(
                    width: 16,
                    height: 16,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  ),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _TelemetryBar extends StatelessWidget {
  const _TelemetryBar({required this.telemetry});

  final GenerationTelemetry telemetry;

  @override
  Widget build(BuildContext context) {
    final text =
        '${telemetry.generatedTokens} tokens | '
        '${telemetry.decodeTokensPerSecond.toStringAsFixed(1)} tok/s | '
        'TTFT ${telemetry.timeToFirstTokenMs.toStringAsFixed(0)} ms';
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
      color: Theme.of(context).colorScheme.surfaceContainerLow,
      child: Text(
        text,
        textAlign: TextAlign.right,
        style: Theme.of(context).textTheme.labelSmall,
      ),
    );
  }
}

class _Composer extends StatelessWidget {
  const _Composer({
    required this.controller,
    required this.enabled,
    required this.canAttachImage,
    required this.canSend,
    required this.isGenerating,
    required this.pendingImage,
    required this.onAttachImage,
    required this.onRemoveImage,
    required this.onSend,
    required this.onCancel,
  });

  final TextEditingController controller;
  final bool enabled;
  final bool canAttachImage;
  final bool canSend;
  final bool isGenerating;
  final _SelectedFile? pendingImage;
  final VoidCallback onAttachImage;
  final VoidCallback onRemoveImage;
  final VoidCallback onSend;
  final VoidCallback onCancel;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.fromLTRB(12, 8, 12, 12),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        border: Border(
          top: BorderSide(color: Theme.of(context).colorScheme.outlineVariant),
        ),
      ),
      child: Column(
        children: <Widget>[
          if (pendingImage case final image?)
            Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: Row(
                children: <Widget>[
                  const Icon(Icons.image_outlined, size: 19),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      image.name,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  IconButton(
                    tooltip: 'Remove image',
                    onPressed: onRemoveImage,
                    icon: const Icon(Icons.close),
                  ),
                ],
              ),
            ),
          Row(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: <Widget>[
              IconButton(
                key: const ValueKey<String>('attach-image-button'),
                tooltip: 'Attach image',
                onPressed: enabled && canAttachImage && !isGenerating
                    ? onAttachImage
                    : null,
                icon: const Icon(Icons.add_photo_alternate_outlined),
              ),
              const SizedBox(width: 4),
              Expanded(
                child: TextField(
                  key: const ValueKey<String>('prompt-field'),
                  controller: controller,
                  enabled: enabled && !isGenerating,
                  minLines: 1,
                  maxLines: 4,
                  textCapitalization: TextCapitalization.sentences,
                  decoration: const InputDecoration(
                    hintText: 'Message',
                    isDense: true,
                  ),
                ),
              ),
              const SizedBox(width: 8),
              if (isGenerating)
                IconButton.filled(
                  key: const ValueKey<String>('cancel-generation-button'),
                  tooltip: 'Stop generation',
                  onPressed: onCancel,
                  icon: const Icon(Icons.stop),
                )
              else
                IconButton.filled(
                  key: const ValueKey<String>('send-button'),
                  tooltip: 'Send',
                  onPressed: canSend ? onSend : null,
                  icon: const Icon(Icons.send),
                ),
            ],
          ),
        ],
      ),
    );
  }
}

final class _SelectedFile {
  const _SelectedFile({required this.path, required this.name});

  final String path;
  final String name;
}

final class _TranscriptEntry {
  _TranscriptEntry({
    required this.role,
    required this.text,
    this.attachmentPath,
    this.attachmentName,
    this.isStreaming = false,
  });

  final ChatRole role;
  String text;
  final String? attachmentPath;
  final String? attachmentName;
  bool isStreaming;
}

String _errorText(Object error) {
  return switch (error) {
    LlamaException() => error.message,
    FileSystemException() => error.message,
    _ => error.toString(),
  };
}
