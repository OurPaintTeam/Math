// Header-only C++20 rewrite of COLAMD (Column Approximate Minimum Degree)
// for CSC sparsity patterns.
//
// Algorithm attribution:
//   Timothy A. Davis, Stefan I. Larimore, and Esmond G. Ng,
//   University of Florida / Oak Ridge National Laboratory.
//
// This file keeps the algorithmic structure of COLAMD as adapted in
// Eigen_Colamd.h, but rewrites the implementation for this project:
//   - header-only
//   - std::span/std::array/concepts based API
//   - no macros
//   - no global mutable state
//   - workspace-owned memory for thread-safe reuse
//
// Original COLAMD notice:
//   Copyright (c) 1998-2003 by the University of Florida.
//   All Rights Reserved.
//   Permission is granted to use, copy, modify, and/or distribute this
//   program provided that the copyright, this license, and the availability
//   of the original version are retained on all copies and made accessible
//   to end users of any package that includes COLAMD or a modified version.

#ifndef MATH_HEADERS_DECOMPOSITION_SPARSE_QR_SPARSEQRCOLAMD_H_
#define MATH_HEADERS_DECOMPOSITION_SPARSE_QR_SPARSEQRCOLAMD_H_

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace sparse_qr {

template <class T>
concept ColamdInteger = std::integral<T> && !std::same_as<std::remove_cv_t<T>, bool>;

template <class T>
concept ColamdSignedIndex = ColamdInteger<T> && std::signed_integral<T>;

inline constexpr std::size_t kColamdKnobCount = 20;
inline constexpr std::size_t kColamdStatCount = 20;

enum class ColamdStatSlot : std::size_t {
    dense_row = 0,
    dense_col = 1,
    defrag_count = 2,
    status = 3,
    info1 = 4,
    info2 = 5,
    info3 = 6
};

enum class ColamdStatus : int {
    ok = 0,
    ok_but_jumbled = 1,
    error_a_not_present = -1,
    error_p_not_present = -2,
    error_nrow_negative = -3,
    error_ncol_negative = -4,
    error_nnz_negative = -5,
    error_p0_nonzero = -6,
    error_a_too_small = -7,
    error_col_length_negative = -8,
    error_row_index_out_of_bounds = -9,
    error_out_of_memory = -10,
    error_index_overflow = -11,
    error_invalid_outer_size = -12,
    error_invalid_inner_size = -13,
    error_internal_error = -999
};

[[nodiscard]] constexpr bool isColamdSuccess(const ColamdStatus status) noexcept {
    using enum ColamdStatus;
    return status == ok || status == ok_but_jumbled;
}

using ColamdKnobs = std::array<double, kColamdKnobCount>;

template <ColamdSignedIndex Index>
using ColamdStats = std::array<Index, kColamdStatCount>;

struct ColamdOptions {
    double dense_row_ratio = 0.5;
    double dense_col_ratio = 0.5;
};

[[nodiscard]] constexpr ColamdKnobs defaultColamdKnobs() noexcept {
    ColamdKnobs knobs{};
    knobs[static_cast<std::size_t>(ColamdStatSlot::dense_row)] = 0.5;
    knobs[static_cast<std::size_t>(ColamdStatSlot::dense_col)] = 0.5;
    return knobs;
}

constexpr void setColamdDefaults(ColamdKnobs& knobs) noexcept {
    knobs = defaultColamdKnobs();
}

[[nodiscard]] constexpr ColamdKnobs makeColamdKnobs(const ColamdOptions& options) noexcept {
    ColamdKnobs knobs = defaultColamdKnobs();
    knobs[static_cast<std::size_t>(ColamdStatSlot::dense_row)] = options.dense_row_ratio;
    knobs[static_cast<std::size_t>(ColamdStatSlot::dense_col)] = options.dense_col_ratio;
    return knobs;
}

template <ColamdInteger Index>
struct CscPatternView {
    Index row_count{};
    Index col_count{};
    std::span<const Index> outer{};
    std::span<const Index> inner{};
};

namespace detail {

template <ColamdSignedIndex Index>
inline constexpr Index kEmpty = static_cast<Index>(-1);

[[nodiscard]] constexpr std::size_t toSlot(const ColamdStatSlot slot) noexcept {
    return static_cast<std::size_t>(slot);
}

template <ColamdSignedIndex Index>
constexpr void resetStats(ColamdStats<Index>& stats) noexcept {
    stats.fill(Index{0});
    stats[toSlot(ColamdStatSlot::status)] = static_cast<Index>(ColamdStatus::ok);
    stats[toSlot(ColamdStatSlot::info1)] = static_cast<Index>(-1);
    stats[toSlot(ColamdStatSlot::info2)] = static_cast<Index>(-1);
}

template <ColamdSignedIndex Index>
constexpr void setStatus(
    ColamdStats<Index>& stats,
    const ColamdStatus status,
    const Index info1 = static_cast<Index>(-1),
    const Index info2 = static_cast<Index>(-1),
    const Index info3 = Index{0}) noexcept
{
    stats[toSlot(ColamdStatSlot::status)] = static_cast<Index>(status);
    stats[toSlot(ColamdStatSlot::info1)] = info1;
    stats[toSlot(ColamdStatSlot::info2)] = info2;
    stats[toSlot(ColamdStatSlot::info3)] = info3;
}

template <ColamdSignedIndex Index>
[[nodiscard]] constexpr ColamdStatus getStatus(const ColamdStats<Index>& stats) noexcept {
    return static_cast<ColamdStatus>(stats[toSlot(ColamdStatSlot::status)]);
}

template <ColamdSignedIndex Index>
[[nodiscard]] constexpr bool checkedAdd(const Index lhs, const Index rhs, Index& result) noexcept {
    if (lhs < 0 || rhs < 0) {
        return false;
    }
    if (lhs > std::numeric_limits<Index>::max() - rhs) {
        return false;
    }
    result = static_cast<Index>(lhs + rhs);
    return true;
}

template <ColamdSignedIndex Index>
[[nodiscard]] constexpr bool checkedMul(const Index lhs, const Index rhs, Index& result) noexcept {
    if (lhs < 0 || rhs < 0) {
        return false;
    }
    if (lhs == 0 || rhs == 0) {
        result = Index{0};
        return true;
    }
    if (lhs > std::numeric_limits<Index>::max() / rhs) {
        return false;
    }
    result = static_cast<Index>(lhs * rhs);
    return true;
}

template <ColamdSignedIndex Index>
[[nodiscard]] constexpr bool checkedIncrement(const Index value, Index& result) noexcept {
    return checkedAdd(value, Index{1}, result);
}

template <ColamdSignedIndex Index>
[[nodiscard]] constexpr bool toSize(const Index value, std::size_t& result) noexcept {
    if (value < 0 || !std::in_range<std::size_t>(value)) {
        return false;
    }
    result = static_cast<std::size_t>(value);
    return true;
}

template <ColamdSignedIndex Index, ColamdInteger InputIndex>
[[nodiscard]] constexpr bool toSignedIndex(const InputIndex value, Index& result) noexcept {
    if constexpr (std::signed_integral<InputIndex>) {
        if (value < 0) {
            return false;
        }
    }
    if (!std::in_range<Index>(value)) {
        return false;
    }
    result = static_cast<Index>(value);
    return true;
}

template <ColamdSignedIndex Index>
[[nodiscard]] constexpr Index onesComplement(const Index value) noexcept {
    return static_cast<Index>(-value - 1);
}

template <ColamdSignedIndex Index>
struct Column {
    Index start = 0;
    Index length = 0;
    Index thickness = 1;
    Index parent = kEmpty<Index>;
    Index score = 0;
    Index order = kEmpty<Index>;
    Index head_hash = kEmpty<Index>;
    Index hash = 0;
    Index prev = kEmpty<Index>;
    Index degree_next = kEmpty<Index>;
    Index hash_next = kEmpty<Index>;

    [[nodiscard]] constexpr bool is_alive() const noexcept { return start >= 0; }
    [[nodiscard]] constexpr bool is_dead() const noexcept { return start < 0; }
    [[nodiscard]] constexpr bool is_dead_principal() const noexcept { return start == static_cast<Index>(-1); }
    constexpr void kill_principal() noexcept { start = static_cast<Index>(-1); }
    constexpr void kill_non_principal() noexcept { start = static_cast<Index>(-2); }
};

template <ColamdSignedIndex Index>
struct Row {
    Index start = 0;
    Index length = 0;
    Index degree = 0;
    Index p = 0;
    Index mark = 0;
    Index first_column = kEmpty<Index>;

    [[nodiscard]] constexpr bool is_alive() const noexcept { return mark >= 0; }
    [[nodiscard]] constexpr bool is_dead() const noexcept { return mark < 0; }
    constexpr void kill() noexcept { mark = static_cast<Index>(-1); }
};

template <ColamdSignedIndex Index>
[[nodiscard]] constexpr Index minimumIndexCapacity(const Index nnz, const Index n_col) noexcept {
    Index twice_nnz = 0;
    Index result = 0;
    if (!checkedMul(nnz, Index{2}, twice_nnz)) {
        return static_cast<Index>(-1);
    }
    if (!checkedAdd(twice_nnz, n_col, result)) {
        return static_cast<Index>(-1);
    }
    return result;
}

template <ColamdSignedIndex Index>
[[nodiscard]] constexpr Index recommendedIndexCapacity(
    const Index nnz,
    const Index /*n_row*/,
    const Index n_col) noexcept
{
    const Index base = minimumIndexCapacity(nnz, n_col);
    if (base < 0) {
        return static_cast<Index>(-1);
    }
    Index result = 0;
    if (!checkedAdd(base, static_cast<Index>(nnz / 5), result)) {
        return static_cast<Index>(-1);
    }
    return result;
}

template <ColamdSignedIndex Index>
[[nodiscard]] constexpr Index denseThreshold(const double ratio, const Index extent) noexcept {
    if (extent <= 0) {
        return Index{0};
    }
    const double scaled = ratio * static_cast<double>(extent);
    if (!(scaled > 0.0)) {
        return Index{0};
    }
    if (scaled >= static_cast<double>(extent)) {
        return extent;
    }
    return static_cast<Index>(scaled);
}

template <typename T>
[[nodiscard]] bool resizeNoThrow(std::vector<T>& buffer, const std::size_t size) noexcept {
    try {
        buffer.resize(size);
        return true;
    } catch (...) {
        return false;
    }
}

template <ColamdSignedIndex Index>
[[nodiscard]] bool initRowsCols(
    const Index n_row,
    const Index n_col,
    const std::span<Row<Index>> rows,
    const std::span<Column<Index>> cols,
    const std::span<Index> a,
    const std::span<Index> p,
    ColamdStats<Index>& stats) noexcept
{
    for (Index col = 0; col < n_col; ++col) {
        Column<Index>& col_state = cols[static_cast<std::size_t>(col)];
        col_state.start = p[static_cast<std::size_t>(col)];
        col_state.length =
            static_cast<Index>(p[static_cast<std::size_t>(col + 1)] - p[static_cast<std::size_t>(col)]);
        if (col_state.length < 0) {
            setStatus(stats, ColamdStatus::error_col_length_negative, col, col_state.length);
            return false;
        }
        col_state.thickness = Index{1};
        col_state.parent = kEmpty<Index>;
        col_state.score = Index{0};
        col_state.order = kEmpty<Index>;
        col_state.head_hash = kEmpty<Index>;
        col_state.hash = Index{0};
        col_state.prev = kEmpty<Index>;
        col_state.degree_next = kEmpty<Index>;
        col_state.hash_next = kEmpty<Index>;
    }

    stats[toSlot(ColamdStatSlot::info3)] = Index{0};
    for (Index row = 0; row < n_row; ++row) {
        rows[static_cast<std::size_t>(row)].length = Index{0};
        rows[static_cast<std::size_t>(row)].mark = static_cast<Index>(-1);
    }

    for (Index col = 0; col < n_col; ++col) {
        Index last_row = static_cast<Index>(-1);
        const Index start = p[static_cast<std::size_t>(col)];
        const Index end = p[static_cast<std::size_t>(col + 1)];
        for (Index pos = start; pos < end; ++pos) {
            const Index row = a[static_cast<std::size_t>(pos)];
            if (row < 0 || row >= n_row) {
                setStatus(stats, ColamdStatus::error_row_index_out_of_bounds, col, row, n_row);
                return false;
            }

            Row<Index>& row_state = rows[static_cast<std::size_t>(row)];
            if (row <= last_row || row_state.mark == col) {
                stats[toSlot(ColamdStatSlot::status)] = static_cast<Index>(ColamdStatus::ok_but_jumbled);
                stats[toSlot(ColamdStatSlot::info1)] = col;
                stats[toSlot(ColamdStatSlot::info2)] = row;
                ++stats[toSlot(ColamdStatSlot::info3)];
            }

            if (row_state.mark != col) {
                ++row_state.length;
            } else {
                --cols[static_cast<std::size_t>(col)].length;
            }

            row_state.mark = col;
            last_row = row;
        }
    }

    if (n_row > 0) {
        rows[0].start = p[static_cast<std::size_t>(n_col)];
        rows[0].p = rows[0].start;
        rows[0].mark = static_cast<Index>(-1);
        for (Index row = 1; row < n_row; ++row) {
            Row<Index>& row_state = rows[static_cast<std::size_t>(row)];
            const Row<Index>& prev_row = rows[static_cast<std::size_t>(row - 1)];
            row_state.start = static_cast<Index>(prev_row.start + prev_row.length);
            row_state.p = row_state.start;
            row_state.mark = static_cast<Index>(-1);
        }
    }

    if (getStatus(stats) == ColamdStatus::ok_but_jumbled) {
        for (Index col = 0; col < n_col; ++col) {
            const Index start = p[static_cast<std::size_t>(col)];
            const Index end = p[static_cast<std::size_t>(col + 1)];
            for (Index pos = start; pos < end; ++pos) {
                const Index row = a[static_cast<std::size_t>(pos)];
                Row<Index>& row_state = rows[static_cast<std::size_t>(row)];
                if (row_state.mark != col) {
                    a[static_cast<std::size_t>(row_state.p++)] = col;
                    row_state.mark = col;
                }
            }
        }
    } else {
        for (Index col = 0; col < n_col; ++col) {
            const Index start = p[static_cast<std::size_t>(col)];
            const Index end = p[static_cast<std::size_t>(col + 1)];
            for (Index pos = start; pos < end; ++pos) {
                const Index row = a[static_cast<std::size_t>(pos)];
                a[static_cast<std::size_t>(rows[static_cast<std::size_t>(row)].p++)] = col;
            }
        }
    }

    for (Index row = 0; row < n_row; ++row) {
        Row<Index>& row_state = rows[static_cast<std::size_t>(row)];
        row_state.mark = Index{0};
        row_state.degree = row_state.length;
    }

    if (getStatus(stats) == ColamdStatus::ok_but_jumbled) {
        if (n_col > 0) {
            cols[0].start = Index{0};
            p[0] = cols[0].start;
            for (Index col = 1; col < n_col; ++col) {
                cols[static_cast<std::size_t>(col)].start =
                    static_cast<Index>(cols[static_cast<std::size_t>(col - 1)].start +
                                       cols[static_cast<std::size_t>(col - 1)].length);
                p[static_cast<std::size_t>(col)] = cols[static_cast<std::size_t>(col)].start;
            }
        }

        for (Index row = 0; row < n_row; ++row) {
            const Row<Index>& row_state = rows[static_cast<std::size_t>(row)];
            const Index start = row_state.start;
            const Index end = static_cast<Index>(row_state.start + row_state.length);
            for (Index pos = start; pos < end; ++pos) {
                const Index col = a[static_cast<std::size_t>(pos)];
                a[static_cast<std::size_t>(p[static_cast<std::size_t>(col)]++)] = row;
            }
        }
    }

    return true;
}

template <ColamdSignedIndex Index>
void initScoring(
    const Index n_row,
    const Index n_col,
    const std::span<Row<Index>> rows,
    const std::span<Column<Index>> cols,
    const std::span<Index> a,
    const std::span<Index> head,
    const ColamdKnobs& knobs,
    Index& n_row2,
    Index& n_col2,
    Index& max_deg) noexcept
{
    const Index dense_row_count =
        std::max(Index{0}, std::min(denseThreshold(knobs[toSlot(ColamdStatSlot::dense_row)], n_col), n_col));
    const Index dense_col_count =
        std::max(Index{0}, std::min(denseThreshold(knobs[toSlot(ColamdStatSlot::dense_col)], n_row), n_row));

    max_deg = Index{0};
    n_col2 = n_col;
    n_row2 = n_row;

    for (Index c = n_col; c-- > 0;) {
        if (cols[static_cast<std::size_t>(c)].length == 0) {
            cols[static_cast<std::size_t>(c)].order = --n_col2;
            cols[static_cast<std::size_t>(c)].kill_principal();
        }
    }

    for (Index c = n_col; c-- > 0;) {
        Column<Index>& col = cols[static_cast<std::size_t>(c)];
        if (col.is_dead()) {
            continue;
        }
        if (col.length > dense_col_count) {
            col.order = --n_col2;
            const Index start = col.start;
            const Index end = static_cast<Index>(col.start + col.length);
            for (Index pos = start; pos < end; ++pos) {
                --rows[static_cast<std::size_t>(a[static_cast<std::size_t>(pos)])].degree;
            }
            col.kill_principal();
        }
    }

    for (Index r = 0; r < n_row; ++r) {
        Row<Index>& row = rows[static_cast<std::size_t>(r)];
        if (row.degree > dense_row_count || row.degree == 0) {
            row.kill();
            --n_row2;
        } else {
            max_deg = std::max(max_deg, row.degree);
        }
    }

    for (Index c = n_col; c-- > 0;) {
        Column<Index>& col = cols[static_cast<std::size_t>(c)];
        if (col.is_dead()) {
            continue;
        }

        Index score = Index{0};
        const Index start = col.start;
        const Index end = static_cast<Index>(col.start + col.length);
        Index new_pos = start;
        for (Index pos = start; pos < end; ++pos) {
            const Index row = a[static_cast<std::size_t>(pos)];
            if (rows[static_cast<std::size_t>(row)].is_dead()) {
                continue;
            }
            a[static_cast<std::size_t>(new_pos++)] = row;
            score = static_cast<Index>(score + rows[static_cast<std::size_t>(row)].degree - 1);
            score = std::min(score, n_col);
        }

        const Index col_length = static_cast<Index>(new_pos - start);
        if (col_length == 0) {
            col.order = --n_col2;
            col.kill_principal();
        } else {
            col.length = col_length;
            col.score = score;
        }
    }

    std::fill(head.begin(), head.end(), kEmpty<Index>);
    for (Index c = n_col; c-- > 0;) {
        Column<Index>& col = cols[static_cast<std::size_t>(c)];
        if (!col.is_alive()) {
            continue;
        }
        const Index score = col.score;
        const Index next_col = head[static_cast<std::size_t>(score)];
        col.prev = kEmpty<Index>;
        col.degree_next = next_col;
        if (next_col != kEmpty<Index>) {
            cols[static_cast<std::size_t>(next_col)].prev = c;
        }
        head[static_cast<std::size_t>(score)] = c;
    }
}

template <ColamdSignedIndex Index>
Index clearMark(const Index n_row, const std::span<Row<Index>> rows) noexcept {
    for (Index row = 0; row < n_row; ++row) {
        if (rows[static_cast<std::size_t>(row)].is_alive()) {
            rows[static_cast<std::size_t>(row)].mark = Index{0};
        }
    }
    return Index{1};
}

template <ColamdSignedIndex Index>
void detectSuperCols(
    const std::span<Column<Index>> cols,
    const std::span<Index> a,
    const std::span<Index> head,
    const Index row_start,
    const Index row_length) noexcept
{
    const Index row_end = static_cast<Index>(row_start + row_length);
    for (Index rp = row_start; rp < row_end; ++rp) {
        const Index col = a[static_cast<std::size_t>(rp)];
        if (cols[static_cast<std::size_t>(col)].is_dead()) {
            continue;
        }

        const Index hash = cols[static_cast<std::size_t>(col)].hash;
        const Index head_column = head[static_cast<std::size_t>(hash)];
        const Index first_col = head_column > kEmpty<Index>
            ? cols[static_cast<std::size_t>(head_column)].head_hash
            : static_cast<Index>(-(head_column + 2));

        for (Index super_col = first_col;
             super_col != kEmpty<Index>;
             super_col = cols[static_cast<std::size_t>(super_col)].hash_next)
        {
            const Index length = cols[static_cast<std::size_t>(super_col)].length;
            Index prev_col = super_col;

            for (Index candidate = cols[static_cast<std::size_t>(super_col)].hash_next;
                 candidate != kEmpty<Index>;)
            {
                const Index next_candidate = cols[static_cast<std::size_t>(candidate)].hash_next;
                if (cols[static_cast<std::size_t>(candidate)].length != length ||
                    cols[static_cast<std::size_t>(candidate)].score != cols[static_cast<std::size_t>(super_col)].score)
                {
                    prev_col = candidate;
                    candidate = next_candidate;
                    continue;
                }

                bool identical = true;
                const Index lhs_start = cols[static_cast<std::size_t>(super_col)].start;
                const Index rhs_start = cols[static_cast<std::size_t>(candidate)].start;
                for (Index offset = 0; offset < length; ++offset) {
                    if (a[static_cast<std::size_t>(lhs_start + offset)] !=
                        a[static_cast<std::size_t>(rhs_start + offset)])
                    {
                        identical = false;
                        break;
                    }
                }

                if (!identical) {
                    prev_col = candidate;
                    candidate = next_candidate;
                    continue;
                }

                cols[static_cast<std::size_t>(super_col)].thickness =
                    static_cast<Index>(cols[static_cast<std::size_t>(super_col)].thickness +
                                       cols[static_cast<std::size_t>(candidate)].thickness);
                cols[static_cast<std::size_t>(candidate)].parent = super_col;
                cols[static_cast<std::size_t>(candidate)].kill_non_principal();
                cols[static_cast<std::size_t>(candidate)].order = kEmpty<Index>;
                cols[static_cast<std::size_t>(prev_col)].hash_next = next_candidate;
                candidate = next_candidate;
            }
        }

        if (head_column > kEmpty<Index>) {
            cols[static_cast<std::size_t>(head_column)].head_hash = kEmpty<Index>;
        } else {
            head[static_cast<std::size_t>(hash)] = kEmpty<Index>;
        }
    }
}

template <ColamdSignedIndex Index>
Index garbageCollection(
    const Index n_row,
    const Index n_col,
    const std::span<Row<Index>> rows,
    const std::span<Column<Index>> cols,
    const std::span<Index> a,
    const Index pfree) noexcept
{
    Index pdest = Index{0};
    for (Index col = 0; col < n_col; ++col) {
        Column<Index>& col_state = cols[static_cast<std::size_t>(col)];
        if (!col_state.is_alive()) {
            continue;
        }

        const Index old_start = col_state.start;
        const Index length = col_state.length;
        col_state.start = pdest;
        for (Index offset = 0; offset < length; ++offset) {
            const Index row = a[static_cast<std::size_t>(old_start + offset)];
            if (rows[static_cast<std::size_t>(row)].is_alive()) {
                a[static_cast<std::size_t>(pdest++)] = row;
            }
        }
        col_state.length = static_cast<Index>(pdest - col_state.start);
    }

    for (Index row = 0; row < n_row; ++row) {
        Row<Index>& row_state = rows[static_cast<std::size_t>(row)];
        if (!row_state.is_alive()) {
            continue;
        }
        if (row_state.length == 0) {
            row_state.kill();
            continue;
        }
        row_state.first_column = a[static_cast<std::size_t>(row_state.start)];
        a[static_cast<std::size_t>(row_state.start)] = onesComplement(row);
    }

    Index psrc = pdest;
    while (psrc < pfree) {
        if (a[static_cast<std::size_t>(psrc)] >= 0) {
            ++psrc;
            continue;
        }

        const Index row = onesComplement(a[static_cast<std::size_t>(psrc)]);
        Row<Index>& row_state = rows[static_cast<std::size_t>(row)];
        a[static_cast<std::size_t>(psrc)] = row_state.first_column;

        const Index length = row_state.length;
        row_state.start = pdest;
        for (Index offset = 0; offset < length; ++offset) {
            const Index col = a[static_cast<std::size_t>(psrc++)];
            if (cols[static_cast<std::size_t>(col)].is_alive()) {
                a[static_cast<std::size_t>(pdest++)] = col;
            }
        }
        row_state.length = static_cast<Index>(pdest - row_state.start);
    }

    return pdest;
}

template <ColamdSignedIndex Index>
Index findOrdering(
    const Index n_row,
    const Index n_col,
    const Index a_length,
    const std::span<Row<Index>> rows,
    const std::span<Column<Index>> cols,
    const std::span<Index> a,
    const std::span<Index> head,
    const Index n_col2,
    Index max_deg,
    Index pfree) noexcept
{
    const Index max_mark = static_cast<Index>(std::numeric_limits<Index>::max() - n_col);
    Index tag_mark = clearMark(n_row, rows);
    Index min_score = Index{0};
    Index garbage_count = Index{0};

    for (Index k = 0; k < n_col2;) {
        while (min_score < n_col && head[static_cast<std::size_t>(min_score)] == kEmpty<Index>) {
            ++min_score;
        }

        const Index pivot_col = head[static_cast<std::size_t>(min_score)];
        const Index next_col = cols[static_cast<std::size_t>(pivot_col)].degree_next;
        head[static_cast<std::size_t>(min_score)] = next_col;
        if (next_col != kEmpty<Index>) {
            cols[static_cast<std::size_t>(next_col)].prev = kEmpty<Index>;
        }

        const Index pivot_col_score = cols[static_cast<std::size_t>(pivot_col)].score;
        cols[static_cast<std::size_t>(pivot_col)].order = k;
        const Index pivot_col_thickness = cols[static_cast<std::size_t>(pivot_col)].thickness;
        k = static_cast<Index>(k + pivot_col_thickness);

        const Index needed_memory = std::min(pivot_col_score, static_cast<Index>(n_col - k));
        if (pfree >= a_length - needed_memory) {
            pfree = garbageCollection(n_row, n_col, rows, cols, a, pfree);
            ++garbage_count;
            tag_mark = clearMark(n_row, rows);
        }

        const Index pivot_row_start = pfree;
        Index pivot_row_degree = Index{0};
        cols[static_cast<std::size_t>(pivot_col)].thickness = static_cast<Index>(-pivot_col_thickness);

        const Index pivot_col_start = cols[static_cast<std::size_t>(pivot_col)].start;
        const Index pivot_col_end =
            static_cast<Index>(pivot_col_start + cols[static_cast<std::size_t>(pivot_col)].length);
        for (Index pos = pivot_col_start; pos < pivot_col_end; ++pos) {
            const Index row = a[static_cast<std::size_t>(pos)];
            if (rows[static_cast<std::size_t>(row)].is_dead()) {
                continue;
            }

            const Index row_start = rows[static_cast<std::size_t>(row)].start;
            const Index row_end =
                static_cast<Index>(row_start + rows[static_cast<std::size_t>(row)].length);
            for (Index row_pos = row_start; row_pos < row_end; ++row_pos) {
                const Index col = a[static_cast<std::size_t>(row_pos)];
                const Index thickness = cols[static_cast<std::size_t>(col)].thickness;
                if (thickness > 0 && cols[static_cast<std::size_t>(col)].is_alive()) {
                    cols[static_cast<std::size_t>(col)].thickness = static_cast<Index>(-thickness);
                    a[static_cast<std::size_t>(pfree++)] = col;
                    pivot_row_degree = static_cast<Index>(pivot_row_degree + thickness);
                }
            }
        }

        cols[static_cast<std::size_t>(pivot_col)].thickness = pivot_col_thickness;
        max_deg = std::max(max_deg, pivot_row_degree);

        for (Index pos = pivot_col_start; pos < pivot_col_end; ++pos) {
            rows[static_cast<std::size_t>(a[static_cast<std::size_t>(pos)])].kill();
        }

        const Index pivot_row_length = static_cast<Index>(pfree - pivot_row_start);
        const Index pivot_row =
            pivot_row_length > 0 ? a[static_cast<std::size_t>(pivot_col_start)] : kEmpty<Index>;

        for (Index pos = pivot_row_start; pos < static_cast<Index>(pivot_row_start + pivot_row_length); ++pos) {
            const Index col = a[static_cast<std::size_t>(pos)];
            const Index col_thickness = static_cast<Index>(-cols[static_cast<std::size_t>(col)].thickness);
            cols[static_cast<std::size_t>(col)].thickness = col_thickness;

            const Index cur_score = cols[static_cast<std::size_t>(col)].score;
            const Index prev_col = cols[static_cast<std::size_t>(col)].prev;
            const Index degree_next = cols[static_cast<std::size_t>(col)].degree_next;
            if (prev_col == kEmpty<Index>) {
                head[static_cast<std::size_t>(cur_score)] = degree_next;
            } else {
                cols[static_cast<std::size_t>(prev_col)].degree_next = degree_next;
            }
            if (degree_next != kEmpty<Index>) {
                cols[static_cast<std::size_t>(degree_next)].prev = prev_col;
            }

            const Index col_start = cols[static_cast<std::size_t>(col)].start;
            const Index col_end =
                static_cast<Index>(col_start + cols[static_cast<std::size_t>(col)].length);
            for (Index col_pos = col_start; col_pos < col_end; ++col_pos) {
                const Index row = a[static_cast<std::size_t>(col_pos)];
                if (rows[static_cast<std::size_t>(row)].is_dead()) {
                    continue;
                }
                Index set_difference = static_cast<Index>(rows[static_cast<std::size_t>(row)].mark - tag_mark);
                if (set_difference < 0) {
                    set_difference = rows[static_cast<std::size_t>(row)].degree;
                }
                set_difference = static_cast<Index>(set_difference - col_thickness);
                if (set_difference == 0) {
                    rows[static_cast<std::size_t>(row)].kill();
                } else {
                    rows[static_cast<std::size_t>(row)].mark =
                        static_cast<Index>(set_difference + tag_mark);
                }
            }
        }

        for (Index pos = pivot_row_start; pos < static_cast<Index>(pivot_row_start + pivot_row_length); ++pos) {
            const Index col = a[static_cast<std::size_t>(pos)];
            Index hash = Index{0};
            Index cur_score = Index{0};

            const Index col_start = cols[static_cast<std::size_t>(col)].start;
            const Index col_end =
                static_cast<Index>(col_start + cols[static_cast<std::size_t>(col)].length);
            Index new_pos = col_start;
            for (Index col_pos = col_start; col_pos < col_end; ++col_pos) {
                const Index row = a[static_cast<std::size_t>(col_pos)];
                if (rows[static_cast<std::size_t>(row)].is_dead()) {
                    continue;
                }

                a[static_cast<std::size_t>(new_pos++)] = row;
                hash = static_cast<Index>(hash + row);
                cur_score = static_cast<Index>(cur_score +
                                               rows[static_cast<std::size_t>(row)].mark - tag_mark);
                cur_score = std::min(cur_score, n_col);
            }

            Column<Index>& col_state = cols[static_cast<std::size_t>(col)];
            col_state.length = static_cast<Index>(new_pos - col_start);
            if (col_state.length == 0) {
                col_state.kill_principal();
                pivot_row_degree = static_cast<Index>(pivot_row_degree - col_state.thickness);
                col_state.order = k;
                k = static_cast<Index>(k + col_state.thickness);
                continue;
            }

            col_state.score = cur_score;
            hash = static_cast<Index>(hash % (n_col + 1));
            const Index head_column = head[static_cast<std::size_t>(hash)];
            Index first_col = kEmpty<Index>;
            if (head_column > kEmpty<Index>) {
                first_col = cols[static_cast<std::size_t>(head_column)].head_hash;
                cols[static_cast<std::size_t>(head_column)].head_hash = col;
            } else {
                first_col = static_cast<Index>(-(head_column + 2));
                head[static_cast<std::size_t>(hash)] = static_cast<Index>(-(col + 2));
            }
            col_state.hash_next = first_col;
            col_state.hash = hash;
        }

        detectSuperCols(cols, a, head, pivot_row_start, pivot_row_length);
        cols[static_cast<std::size_t>(pivot_col)].kill_principal();

        const Index tag_increment = static_cast<Index>(max_deg + 1);
        if (tag_mark >= static_cast<Index>(max_mark - tag_increment)) {
            tag_mark = clearMark(n_row, rows);
        } else {
            tag_mark = static_cast<Index>(tag_mark + tag_increment);
        }

        Index new_rp = pivot_row_start;
        for (Index pos = pivot_row_start; pos < static_cast<Index>(pivot_row_start + pivot_row_length); ++pos) {
            const Index col = a[static_cast<std::size_t>(pos)];
            Column<Index>& col_state = cols[static_cast<std::size_t>(col)];
            if (col_state.is_dead()) {
                continue;
            }

            a[static_cast<std::size_t>(new_rp++)] = col;
            a[static_cast<std::size_t>(col_state.start + col_state.length)] = pivot_row;
            ++col_state.length;

            Index cur_score = static_cast<Index>(col_state.score + pivot_row_degree);
            const Index max_score = static_cast<Index>(n_col - k - col_state.thickness);
            cur_score = static_cast<Index>(cur_score - col_state.thickness);
            cur_score = std::min(cur_score, max_score);
            col_state.score = cur_score;

            const Index degree_next = head[static_cast<std::size_t>(cur_score)];
            col_state.degree_next = degree_next;
            col_state.prev = kEmpty<Index>;
            if (degree_next != kEmpty<Index>) {
                cols[static_cast<std::size_t>(degree_next)].prev = col;
            }
            head[static_cast<std::size_t>(cur_score)] = col;
            min_score = std::min(min_score, cur_score);
        }

        if (pivot_row_degree > 0) {
            Row<Index>& row_state = rows[static_cast<std::size_t>(pivot_row)];
            row_state.start = pivot_row_start;
            row_state.length = static_cast<Index>(new_rp - pivot_row_start);
            row_state.degree = pivot_row_degree;
            row_state.mark = Index{0};
        }
    }

    return garbage_count;
}

template <ColamdSignedIndex Index>
void orderChildren(
    const Index n_col,
    const std::span<Column<Index>> cols,
    const std::span<Index> permutation) noexcept
{
    for (Index i = 0; i < n_col; ++i) {
        Column<Index>& col = cols[static_cast<std::size_t>(i)];
        if (col.is_dead_principal() || col.order != kEmpty<Index>) {
            continue;
        }

        Index parent = i;
        do {
            parent = cols[static_cast<std::size_t>(parent)].parent;
        } while (!cols[static_cast<std::size_t>(parent)].is_dead_principal());

        Index current = i;
        Index order = cols[static_cast<std::size_t>(parent)].order;
        do {
            Column<Index>& current_col = cols[static_cast<std::size_t>(current)];
            const Index next_parent = current_col.parent;
            current_col.order = order++;
            current_col.parent = parent;
            current = next_parent;
        } while (cols[static_cast<std::size_t>(current)].order == kEmpty<Index>);

        cols[static_cast<std::size_t>(parent)].order = order;
    }

    for (Index col = 0; col < n_col; ++col) {
        permutation[static_cast<std::size_t>(cols[static_cast<std::size_t>(col)].order)] = col;
    }
}

template <ColamdSignedIndex Index, ColamdInteger InputIndex>
[[nodiscard]] bool copyInput(
    const CscPatternView<InputIndex>& matrix,
    const std::span<Index> column_pointers,
    const std::span<Index> a,
    ColamdStats<Index>& stats) noexcept
{
    for (std::size_t pos = 0; pos < matrix.outer.size(); ++pos) {
        Index value = 0;
        if (!toSignedIndex(matrix.outer[pos], value)) {
            setStatus(stats, ColamdStatus::error_index_overflow, static_cast<Index>(pos));
            return false;
        }
        column_pointers[pos] = value;
    }

    for (std::size_t pos = 0; pos < matrix.inner.size(); ++pos) {
        if constexpr (!std::in_range<Index>(std::numeric_limits<InputIndex>::max())) {
            if (!std::in_range<Index>(matrix.inner[pos])) {
                setStatus(stats, ColamdStatus::error_index_overflow, static_cast<Index>(pos));
                return false;
            }
        }
        a[pos] = static_cast<Index>(matrix.inner[pos]);
    }
    return true;
}

template <ColamdSignedIndex Index, ColamdInteger InputIndex>
[[nodiscard]] bool validateHeader(
    const CscPatternView<InputIndex>& matrix,
    Index& n_row,
    Index& n_col,
    Index& nnz,
    ColamdStats<Index>& stats) noexcept
{
    if constexpr (std::signed_integral<InputIndex>) {
        if (matrix.row_count < 0) {
            setStatus(stats, ColamdStatus::error_nrow_negative, static_cast<Index>(matrix.row_count));
            return false;
        }
        if (matrix.col_count < 0) {
            setStatus(stats, ColamdStatus::error_ncol_negative, static_cast<Index>(matrix.col_count));
            return false;
        }
    }

    if (!toSignedIndex(matrix.row_count, n_row) || !toSignedIndex(matrix.col_count, n_col)) {
        setStatus(stats, ColamdStatus::error_index_overflow);
        return false;
    }

    const std::size_t expected_outer = static_cast<std::size_t>(n_col) + 1;
    if (matrix.outer.size() != expected_outer) {
        setStatus(
            stats,
            ColamdStatus::error_invalid_outer_size,
            static_cast<Index>(expected_outer),
            static_cast<Index>(matrix.outer.size()));
        return false;
    }

    if (matrix.outer.empty()) {
        setStatus(stats, ColamdStatus::error_p_not_present);
        return false;
    }

    if (!toSignedIndex(matrix.outer.back(), nnz)) {
        setStatus(stats, ColamdStatus::error_index_overflow);
        return false;
    }
    if (nnz < 0) {
        setStatus(stats, ColamdStatus::error_nnz_negative, nnz);
        return false;
    }

    if (matrix.inner.size() != static_cast<std::size_t>(nnz)) {
        setStatus(
            stats,
            ColamdStatus::error_invalid_inner_size,
            nnz,
            static_cast<Index>(matrix.inner.size()));
        return false;
    }

    Index first_pointer = 0;
    if (!toSignedIndex(matrix.outer.front(), first_pointer)) {
        setStatus(stats, ColamdStatus::error_index_overflow);
        return false;
    }
    if (first_pointer != 0) {
        setStatus(stats, ColamdStatus::error_p0_nonzero, first_pointer);
        return false;
    }

    return true;
}

template <ColamdSignedIndex Index>
[[nodiscard]] constexpr Index initialPfree(const Index nnz) noexcept {
    Index result = 0;
    return checkedMul(nnz, Index{2}, result) ? result : static_cast<Index>(-1);
}

} // namespace detail

template <ColamdSignedIndex Index = std::ptrdiff_t>
struct ColamdWorkspace {
    using index_type = Index;

    ColamdKnobs knobs = defaultColamdKnobs();
    ColamdStats<Index> stats{};
    std::vector<Index> column_pointers;
    std::vector<Index> indices;
    std::vector<detail::Row<Index>> rows;
    std::vector<detail::Column<Index>> columns;
    std::vector<Index> degree_heads;
    std::vector<Index> permutation;

    // Reinitializes the fixed-size statistics array before each run.
    constexpr void resetStatistics() noexcept { detail::resetStats(stats); }

    [[nodiscard]] constexpr ColamdStatus status() const noexcept {
        return detail::getStatus(stats);
    }

    [[nodiscard]] constexpr bool ok() const noexcept {
        return isColamdSuccess(status());
    }

    [[nodiscard]] std::span<const Index> ordering() const noexcept {
        return permutation;
    }

    [[nodiscard]] std::span<const Index, kColamdStatCount> statistics() const noexcept {
        return std::span<const Index, kColamdStatCount>(stats);
    }

    // Allocates or resizes all reusable buffers needed by the symbolic pass.
    [[nodiscard]] ColamdStatus prepare(
        const Index n_row,
        const Index n_col,
        const Index nnz,
        const Index elbow_room = detail::kEmpty<Index>) noexcept
    {
        Index index_capacity = detail::minimumIndexCapacity(nnz, n_col);
        if (index_capacity < 0) {
            return ColamdStatus::error_index_overflow;
        }

        const Index extra = elbow_room >= 0 ? elbow_room : static_cast<Index>(nnz / 5);
        if (!detail::checkedAdd(index_capacity, extra, index_capacity)) {
            return ColamdStatus::error_index_overflow;
        }

        Index row_count = 0;
        Index col_count = 0;
        if (!detail::checkedIncrement(n_row, row_count) || !detail::checkedIncrement(n_col, col_count)) {
            return ColamdStatus::error_index_overflow;
        }

        std::size_t p_size = 0;
        std::size_t a_size = 0;
        std::size_t row_size = 0;
        std::size_t col_size = 0;
        std::size_t perm_size = 0;
        if (!detail::toSize(col_count, p_size) ||
            !detail::toSize(index_capacity, a_size) ||
            !detail::toSize(row_count, row_size) ||
            !detail::toSize(col_count, col_size) ||
            !detail::toSize(n_col, perm_size))
        {
            return ColamdStatus::error_index_overflow;
        }

        if (!detail::resizeNoThrow(column_pointers, p_size) ||
            !detail::resizeNoThrow(indices, a_size) ||
            !detail::resizeNoThrow(rows, row_size) ||
            !detail::resizeNoThrow(columns, col_size) ||
            !detail::resizeNoThrow(degree_heads, p_size) ||
            !detail::resizeNoThrow(permutation, perm_size))
        {
            return ColamdStatus::error_out_of_memory;
        }

        return ColamdStatus::ok;
    }
};

template <ColamdSignedIndex Index>
[[nodiscard]] constexpr Index recommendedColamdIndexCapacity(
    const Index nnz,
    const Index n_row,
    const Index n_col) noexcept
{
    // Unlike SuiteSparse/Eigen, the row/column metadata lives in ColamdWorkspace
    // vectors, so this number only covers the mutable index arena.
    return detail::recommendedIndexCapacity(nnz, n_row, n_col);
}

template <ColamdInteger InputIndex, ColamdSignedIndex Index>
[[nodiscard]] bool computeColamdOrdering(
    const CscPatternView<InputIndex>& matrix,
    ColamdWorkspace<Index>& workspace,
    const ColamdKnobs& knobs) noexcept
{
    // The workspace owns all temporary state, so the routine is thread-safe
    // as long as each caller uses its own ColamdWorkspace instance.
    workspace.resetStatistics();

    Index n_row = 0;
    Index n_col = 0;
    Index nnz = 0;
    if (!detail::validateHeader(matrix, n_row, n_col, nnz, workspace.stats)) {
        return false;
    }

    const ColamdStatus prepare_status = workspace.prepare(n_row, n_col, nnz);
    if (prepare_status != ColamdStatus::ok) {
        detail::setStatus(workspace.stats, prepare_status);
        return false;
    }

    workspace.knobs = knobs;

    auto column_pointers = std::span<Index>(workspace.column_pointers).first(static_cast<std::size_t>(n_col + 1));
    auto a = std::span<Index>(workspace.indices);
    auto rows = std::span<detail::Row<Index>>(workspace.rows).first(static_cast<std::size_t>(n_row + 1));
    auto cols = std::span<detail::Column<Index>>(workspace.columns).first(static_cast<std::size_t>(n_col + 1));
    auto head = std::span<Index>(workspace.degree_heads).first(static_cast<std::size_t>(n_col + 1));
    auto permutation = std::span<Index>(workspace.permutation).first(static_cast<std::size_t>(n_col));

    if (!detail::copyInput(matrix, column_pointers, a, workspace.stats)) {
        return false;
    }

    const Index minimum_capacity = detail::minimumIndexCapacity(nnz, n_col);
    if (minimum_capacity < 0 || static_cast<Index>(a.size()) < minimum_capacity) {
        detail::setStatus(
            workspace.stats,
            ColamdStatus::error_a_too_small,
            minimum_capacity,
            static_cast<Index>(a.size()));
        return false;
    }

    if (!detail::initRowsCols(n_row, n_col, rows, cols, a, column_pointers, workspace.stats)) {
        return false;
    }

    Index n_row2 = 0;
    Index n_col2 = 0;
    Index max_deg = 0;
    detail::initScoring(n_row, n_col, rows, cols, a, head, workspace.knobs, n_row2, n_col2, max_deg);

    const Index pfree = detail::initialPfree(nnz);
    if (pfree < 0 || pfree > static_cast<Index>(a.size())) {
        detail::setStatus(workspace.stats, ColamdStatus::error_internal_error);
        return false;
    }

    const Index garbage_count =
        detail::findOrdering(n_row, n_col, static_cast<Index>(a.size()), rows, cols, a, head, n_col2, max_deg, pfree);
    detail::orderChildren(n_col, cols, permutation);

    workspace.stats[detail::toSlot(ColamdStatSlot::dense_row)] = static_cast<Index>(n_row - n_row2);
    workspace.stats[detail::toSlot(ColamdStatSlot::dense_col)] = static_cast<Index>(n_col - n_col2);
    workspace.stats[detail::toSlot(ColamdStatSlot::defrag_count)] = garbage_count;
    return isColamdSuccess(workspace.status());
}

template <ColamdInteger InputIndex, ColamdSignedIndex Index>
[[nodiscard]] bool computeColamdOrdering(
    const CscPatternView<InputIndex>& matrix,
    ColamdWorkspace<Index>& workspace,
    const ColamdOptions& options = {}) noexcept
{
    return computeColamdOrdering(matrix, workspace, makeColamdKnobs(options));
}

} // namespace sparse_qr

#endif // MATH_HEADERS_DECOMPOSITION_SPARSE_QR_SPARSEQRCOLAMD_H_
