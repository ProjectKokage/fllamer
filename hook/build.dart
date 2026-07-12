import 'dart:io';

import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';

const _libraryName = 'llama_dart_bridge';
const _assetName = 'llama_dart_bridge';
const _cmakeBuildType = 'RelWithDebInfo';
const minimumIosVersion = 15;
const maximumDefaultBuildJobs = 4;

Future<void> main(List<String> args) async {
  await build(args, (input, output) async {
    if (!input.config.buildCodeAssets) {
      return;
    }

    final code = input.config.code;
    final targetOS = code.targetOS;
    final targetArchitecture = code.targetArchitecture;
    final sourceDir = input.packageRoot.resolve('native/llama_dart_bridge/');
    final buildDir = input.outputDirectory.resolve(
      'cmake-${targetOS.name}-${targetArchitecture.name}/',
    );
    final linkMode = DynamicLoadingBundled();
    final outputName = targetOS.libraryFileName(_libraryName, linkMode);
    final assetFile = input.outputDirectory.resolve(outputName);

    final configureArgs = <String>[
      '-S',
      sourceDir.toFilePath(),
      '-B',
      buildDir.toFilePath(),
      '-DCMAKE_BUILD_TYPE=$_cmakeBuildType',
      '-DBUILD_TESTING=OFF',
      ..._targetArgs(code),
    ];
    await _run('cmake', configureArgs);
    await _run('cmake', [
      '--build',
      buildDir.toFilePath(),
      '--config',
      _cmakeBuildType,
      '--target',
      _libraryName,
      '--parallel',
      '${cmakeBuildParallelism()}',
    ]);

    final builtFile = await builtLibraryForCmakeOutput(buildDir, outputName);
    await Directory.fromUri(input.outputDirectory).create(recursive: true);
    await builtFile.copy(assetFile.toFilePath());

    output.dependencies.addAll(_nativeBuildDependencies(input.packageRoot));
    output.assets.code.add(
      CodeAsset(
        package: input.packageName,
        name: _assetName,
        linkMode: linkMode,
        file: assetFile,
      ),
    );
  });
}

List<Uri> _nativeBuildDependencies(Uri packageRoot) {
  final dependencies = <Uri>[
    packageRoot.resolve('third_party/llama.cpp/CMakeLists.txt'),
    packageRoot.resolve('third_party/llama.cpp/LICENSE'),
  ];
  for (final relativePath in const <String>[
    'native/llama_dart_bridge/',
    'third_party/llama.cpp/cmake/',
    'third_party/llama.cpp/common/',
    'third_party/llama.cpp/ggml/',
    'third_party/llama.cpp/include/',
    'third_party/llama.cpp/licenses/',
    'third_party/llama.cpp/src/',
    'third_party/llama.cpp/tools/mtmd/',
    'third_party/llama.cpp/vendor/cpp-httplib/',
    'third_party/llama.cpp/vendor/miniaudio/',
    'third_party/llama.cpp/vendor/nlohmann/',
    'third_party/llama.cpp/vendor/stb/',
  ]) {
    dependencies.addAll(_filesUnder(packageRoot.resolve(relativePath)));
  }
  dependencies.sort((a, b) => a.toString().compareTo(b.toString()));
  return dependencies;
}

List<Uri> _filesUnder(Uri directory) {
  final root = Directory.fromUri(directory);
  if (!root.existsSync()) {
    return const <Uri>[];
  }
  return root
      .listSync(recursive: true, followLinks: false)
      .whereType<File>()
      .map((file) => file.uri)
      .toList(growable: false);
}

List<String> _targetArgs(CodeConfig code) {
  final os = code.targetOS;
  final arch = code.targetArchitecture;
  if (os == OS.android) {
    final abi = androidAbiForNativeAssetsBuild(arch);
    final ndk = _androidNdk(code);
    if (ndk == null) {
      throw BuildError(message: 'Android NDK was not found for native build.');
    }
    return [
      '-DCMAKE_TOOLCHAIN_FILE=$ndk/build/cmake/android.toolchain.cmake',
      '-DANDROID_ABI=$abi',
      '-DANDROID_PLATFORM=android-${code.android.targetNdkApi}',
      '-DANDROID_STL=c++_static',
      '-DCMAKE_ANDROID_STL_TYPE=c++_static',
    ];
  }
  if (os == OS.iOS) {
    final deploymentTarget = iosDeploymentTargetForNativeAssetsBuild(
      code.iOS.targetVersion,
    );
    return [
      '-DCMAKE_SYSTEM_NAME=iOS',
      '-DCMAKE_OSX_ARCHITECTURES=${_appleArch(arch)}',
      '-DCMAKE_OSX_SYSROOT=${code.iOS.targetSdk.type}',
      '-DCMAKE_OSX_DEPLOYMENT_TARGET=$deploymentTarget',
    ];
  }
  if (os == OS.macOS) {
    return [
      '-DCMAKE_OSX_ARCHITECTURES=${_appleArch(arch)}',
      '-DCMAKE_OSX_DEPLOYMENT_TARGET=${code.macOS.targetVersion}.0',
    ];
  }
  if (os == OS.linux) {
    return const <String>[];
  }
  throw BuildError(message: 'Native build is not configured for ${os.name}.');
}

String androidAbiForNativeAssetsBuild(Architecture architecture) {
  if (architecture == Architecture.arm64) {
    return 'arm64-v8a';
  }
  if (architecture == Architecture.x64) {
    return 'x86_64';
  }
  throw BuildError(
    message:
        'Android ABI is not configured for ${architecture.name}. '
        'Supported Android ABIs are arm64-v8a and x86_64. '
        'Build with --target-platform android-arm64,android-x64.',
  );
}

String iosDeploymentTargetForNativeAssetsBuild(int targetVersion) {
  final effectiveVersion = targetVersion < minimumIosVersion
      ? minimumIosVersion
      : targetVersion;
  return '$effectiveVersion.0';
}

int cmakeBuildParallelism({
  Map<String, String>? environment,
  int? processorCount,
}) {
  final configured = int.tryParse(
    (environment ?? Platform.environment)['FLLAMER_BUILD_JOBS'] ?? '',
  );
  if (configured != null && configured > 0) {
    return configured;
  }
  final available = processorCount ?? Platform.numberOfProcessors;
  if (available <= 0) {
    return 1;
  }
  return available > maximumDefaultBuildJobs
      ? maximumDefaultBuildJobs
      : available;
}

String _appleArch(Architecture architecture) {
  if (architecture == Architecture.arm64) {
    return 'arm64';
  }
  if (architecture == Architecture.x64) {
    return 'x86_64';
  }
  throw BuildError(
    message: 'Apple architecture is not configured for ${architecture.name}.',
  );
}

String? androidNdkForNativeAssetsCompiler(Uri? compiler) {
  if (compiler == null || !compiler.isScheme('file')) {
    return null;
  }

  var directory = File.fromUri(compiler).parent;
  while (true) {
    final path = directory.path;
    if (!_isUsableBuildPath(path)) {
      return null;
    }
    if (_isAndroidNdk(path)) {
      return path;
    }

    final parent = directory.parent;
    if (parent.path == path) {
      return null;
    }
    directory = parent;
  }
}

String? _androidNdk(CodeConfig code) {
  final compilerNdk = androidNdkForNativeAssetsCompiler(
    code.cCompiler?.compiler,
  );
  if (compilerNdk != null) {
    return compilerNdk;
  }

  for (final name in const [
    'ANDROID_NDK',
    'ANDROID_NDK_HOME',
    'ANDROID_NDK_LATEST_HOME',
    'ANDROID_NDK_ROOT',
  ]) {
    final path = Platform.environment[name];
    if (path != null && _isUsableBuildPath(path) && _isAndroidNdk(path)) {
      return path;
    }
  }

  final androidHome = Platform.environment['ANDROID_HOME'];
  if (androidHome == null || !_isUsableBuildPath(androidHome)) {
    return null;
  }
  return newestAndroidNdkInSdk(androidHome);
}

String? newestAndroidNdkInSdk(String androidHome) {
  if (!_isUsableBuildPath(androidHome)) {
    return null;
  }
  final ndkRoot = Directory('$androidHome/ndk');
  if (!ndkRoot.existsSync()) {
    return null;
  }
  final candidates =
      ndkRoot
          .listSync()
          .whereType<Directory>()
          .where(
            (dir) => _isUsableBuildPath(dir.path) && _isAndroidNdk(dir.path),
          )
          .toList()
        ..sort(_compareAndroidNdkDirectories);
  return candidates.isEmpty ? null : candidates.last.path;
}

int _compareAndroidNdkDirectories(Directory a, Directory b) {
  final byVersion = _compareVersionParts(
    _versionParts(_basename(a.path)),
    _versionParts(_basename(b.path)),
  );
  return byVersion == 0 ? a.path.compareTo(b.path) : byVersion;
}

String _basename(String path) {
  final normalized = path.replaceAll('\\', '/');
  final trimmed = normalized.endsWith('/')
      ? normalized.substring(0, normalized.length - 1)
      : normalized;
  final slash = trimmed.lastIndexOf('/');
  return slash < 0 ? trimmed : trimmed.substring(slash + 1);
}

List<int> _versionParts(String value) {
  return <int>[
    for (final match in RegExp(r'\d+').allMatches(value))
      int.tryParse(match.group(0)!) ?? 0,
  ];
}

int _compareVersionParts(List<int> a, List<int> b) {
  final length = a.length > b.length ? a.length : b.length;
  for (var i = 0; i < length; i += 1) {
    final left = i < a.length ? a[i] : 0;
    final right = i < b.length ? b[i] : 0;
    if (left != right) {
      return left.compareTo(right);
    }
  }
  return 0;
}

bool _isUsableBuildPath(String path) {
  return path.trim().isNotEmpty &&
      !path.contains('\u0000') &&
      !path.contains('\n') &&
      !path.contains('\r');
}

bool _isAndroidNdk(String path) {
  return File('$path/build/cmake/android.toolchain.cmake').existsSync();
}

Future<File> builtLibraryForCmakeOutput(Uri buildDir, String outputName) async {
  for (final candidate in [
    buildDir.resolve(outputName),
    buildDir.resolve('$_cmakeBuildType/$outputName'),
    buildDir.resolve('Release/$outputName'),
  ]) {
    final file = File.fromUri(candidate);
    if (await file.exists()) {
      return file;
    }
  }
  final root = Directory.fromUri(buildDir);
  if (await root.exists()) {
    final matches = <File>[];
    await for (final entity in root.list(recursive: true, followLinks: false)) {
      if (entity is File && _basename(entity.path) == outputName) {
        matches.add(entity);
      }
    }
    if (matches.isNotEmpty) {
      matches.sort((a, b) {
        final aPreferred = _isCmakeConfigurationOutput(a.path, _cmakeBuildType);
        final bPreferred = _isCmakeConfigurationOutput(b.path, _cmakeBuildType);
        if (aPreferred != bPreferred) {
          return aPreferred ? -1 : 1;
        }
        final aRelease = _isCmakeConfigurationOutput(a.path, 'Release');
        final bRelease = _isCmakeConfigurationOutput(b.path, 'Release');
        if (aRelease != bRelease) {
          return aRelease ? -1 : 1;
        }
        return a.path.compareTo(b.path);
      });
      return matches.first;
    }
  }
  throw BuildError(message: 'CMake did not produce $outputName.');
}

bool _isCmakeConfigurationOutput(String path, String configuration) {
  return path
      .replaceAll('\\', '/')
      .split('/')
      .any(
        (part) => part == configuration || part.startsWith('$configuration-'),
      );
}

Future<void> _run(String executable, List<String> args) async {
  final process = await Process.start(executable, args);
  final output = stdout.addStream(process.stdout);
  final errors = stderr.addStream(process.stderr);
  final exitCode = await process.exitCode;
  await Future.wait<void>([output, errors]);
  if (exitCode != 0) {
    throw BuildError(message: 'Command failed: $executable ${args.join(' ')}');
  }
}
