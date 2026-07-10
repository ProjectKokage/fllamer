if(NOT NM OR NOT LIBRARY)
  message(FATAL_ERROR "NM and LIBRARY are required")
endif()

if(APPLE_PLATFORM)
  execute_process(
    COMMAND "${NM}" -gU "${LIBRARY}"
    RESULT_VARIABLE nm_result
    OUTPUT_VARIABLE nm_output
    ERROR_VARIABLE nm_error
  )
  set(expected_pattern "^_llama_dart_")
else()
  execute_process(
    COMMAND "${NM}" -D --defined-only "${LIBRARY}"
    RESULT_VARIABLE nm_result
    OUTPUT_VARIABLE nm_output
    ERROR_VARIABLE nm_error
  )
  set(expected_pattern "^llama_dart_")
endif()

if(NOT nm_result EQUAL 0)
  message(FATAL_ERROR "nm failed: ${nm_error}")
endif()

set(export_count 0)
string(REPLACE "\n" ";" nm_lines "${nm_output}")
foreach(line IN LISTS nm_lines)
  string(STRIP "${line}" line)
  if(line STREQUAL "")
    continue()
  endif()
  string(REGEX MATCH "[^ ]+$" symbol "${line}")
  if(NOT symbol MATCHES "${expected_pattern}")
    message(FATAL_ERROR "unexpected exported symbol: ${symbol}")
  endif()
  math(EXPR export_count "${export_count} + 1")
endforeach()

if(export_count EQUAL 0)
  message(FATAL_ERROR "bridge exports no llama_dart symbols")
endif()

message(STATUS "validated ${export_count} llama_dart bridge exports")
