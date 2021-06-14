#pragma once

#include <chrono>

#include <fmt/format.h>
#include <fmt/color.h>

#ifndef SCYLLA_BLAS_LOGLEVEL
#define SCYLLA_BLAS_LOGLEVEL TRACE
#endif

#define MAX_FUN_NAME_LEN_STR "10"

namespace scylla_blas {
namespace logging {
enum class LoggingLevel {
    TRACE = 0,
    DEBUG = 1,
    INFO = 2,
    WARN = 3,
    ERROR = 4,
    CRITICAL = 5,
    NONE = 6
};

static inline constexpr LoggingLevel LOGLEVEL = LoggingLevel::SCYLLA_BLAS_LOGLEVEL;
static inline constexpr size_t MAX_FUN_NAME_LEN = 10;
static inline const auto start_time = std::chrono::steady_clock::now();
static inline auto fmt_fg_color = fmt::terminal_color::blue;
static inline auto fmt_function_color = fmt::terminal_color::cyan;
static inline const std::string log_format = fmt::format("{}{}{}{{:^8s}}{} ",
    fmt::format(fmt::fg(fmt_fg_color), "[{{:06f}}|"), // format begin, time, separator
    fmt::format(fmt::fg(fmt_function_color), "{{:^" MAX_FUN_NAME_LEN_STR "s}}"), // function name
    fmt::format(fmt::fg(fmt_fg_color), "|"), // seperator
    fmt::format(fmt::fg(fmt_fg_color), "]") // format end
    );
static inline std::string levels_formats[] = {
        fmt::format(fmt::fg(fmt::color::gray),"{:^8s}", "TRACE"),
        fmt::format(fmt::fg(fmt::color::magenta),"{:^8s}", "DEBUG"),
        fmt::format(fmt::fg(fmt::color::white),"{:^8s}", "INFO"),
        fmt::format(fmt::fg(fmt::color::yellow),"{:^8s}", "WARN"),
        fmt::format(fmt::fg(fmt::color::dark_orange),"{:^8s}", "ERROR"),
        fmt::format(fmt::fg(fmt::color::red),"{:^8s}", "CRITICAL"),
};
}
}

#define blas_internal_log(log_level, str_fmt, args...) do { \
    std::chrono::duration<float> duration = std::chrono::steady_clock::now() - scylla_blas::logging::start_time; \
    fmt::print(stderr, scylla_blas::logging::log_format, \
        duration.count(), fmt::basic_string_view(__func__, std::min(sizeof(__func__)-1, scylla_blas::logging::MAX_FUN_NAME_LEN)), \
        scylla_blas::logging::levels_formats[static_cast<int>(log_level)]); \
    fmt::print(str_fmt "\n", ## args); \
} while(0)

#define LogTrace(fmt, args...) do { \
    if constexpr (scylla_blas::logging::LOGLEVEL <= scylla_blas::logging::LoggingLevel::TRACE) { \
        blas_internal_log(scylla_blas::logging::LoggingLevel::TRACE, fmt, ## args); \
    } \
} while(0)

#define LogDebug(fmt, args...) do { \
    if constexpr (scylla_blas::logging::LOGLEVEL <= scylla_blas::logging::LoggingLevel::DEBUG) { \
        blas_internal_log(scylla_blas::logging::LoggingLevel::DEBUG, fmt, ## args); \
    } \
} while(0)

#define LogInfo(fmt, args...) do { \
    if constexpr (scylla_blas::logging::LOGLEVEL <= scylla_blas::logging::LoggingLevel::INFO) { \
        blas_internal_log(scylla_blas::logging::LoggingLevel::INFO, fmt, ## args); \
    } \
} while(0)

#define LogWarn(fmt, args...) do { \
    if constexpr (scylla_blas::logging::LOGLEVEL <= scylla_blas::logging::LoggingLevel::WARN) { \
        blas_internal_log(scylla_blas::logging::LoggingLevel::WARN, fmt, ## args); \
    } \
} while(0)

#define LogError(fmt, args...) do { \
    if constexpr (scylla_blas::logging::LOGLEVEL <= scylla_blas::logging::LoggingLevel::ERROR) { \
        blas_internal_log(scylla_blas::logging::LoggingLevel::ERROR, fmt, ## args); \
    } \
} while(0)

#define LogCritical(fmt, args...) do { \
    if constexpr (scylla_blas::logging::LOGLEVEL <= scylla_blas::logging::LoggingLevel::CRITICAL) { \
        blas_internal_log(scylla_blas::logging::LoggingLevel::CRITICAL, fmt, ## args); \
    } \
} while(0)
