#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include <Eigen/OrderingMethods>

#include "SparseMatrix.h"
#include "SparseQrColamd.h"

namespace {

struct CscPatternBuffers {
    std::vector<size_t> outer;
    std::vector<size_t> inner;
};

struct EigenColamdResult {
    bool success = false;
    std::array<int, Eigen::internal::Colamd::NStats> stats{};
    std::vector<std::ptrdiff_t> permutation;
};

struct StructuralSignature {
    uint64_t active_rows = 0;
    uint64_t elimination_parent_sum = 0;
    uint64_t elimination_depth_sum = 0;

    friend bool operator==(const StructuralSignature&, const StructuralSignature&) = default;
};

CscPatternBuffers BuildCscPattern(const SparseMatrix<>& matrix) {
    SparseMatrix<> buffer;
    const SparseMatrix<>* compressed = &matrix;
    if (matrix.rowMutableMode()) {
        buffer = matrix.compressed();
        compressed = &buffer;
    }

    return CscPatternBuffers{
        compressed->outerIndexData(),
        compressed->innerIndexData()
    };
}

bool IsPermutation(const std::span<const std::ptrdiff_t> ordering, const size_t n) {
    if (ordering.size() != n) {
        return false;
    }
    std::vector<unsigned char> seen(n, 0);
    for (const std::ptrdiff_t value : ordering) {
        if (value < 0 || static_cast<size_t>(value) >= n) {
            return false;
        }
        if (seen[static_cast<size_t>(value)] != 0) {
            return false;
        }
        seen[static_cast<size_t>(value)] = 1;
    }
    return true;
}

EigenColamdResult RunEigenColamd(
    const size_t row_count,
    const size_t col_count,
    const std::span<const size_t> outer,
    const std::span<const size_t> inner)
{
    EigenColamdResult result;
    result.stats.fill(0);

    if (row_count > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        col_count > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        inner.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        return result;
    }

    const int n_row = static_cast<int>(row_count);
    const int n_col = static_cast<int>(col_count);
    const int nnz = static_cast<int>(inner.size());
    const int alen = Eigen::internal::Colamd::recommended<int>(nnz, n_row, n_col);
    if (alen <= 0) {
        return result;
    }

    std::vector<int> a(static_cast<size_t>(alen), 0);
    std::vector<int> p(static_cast<size_t>(n_col) + 1, 0);
    for (size_t i = 0; i < outer.size(); ++i) {
        p[i] = static_cast<int>(outer[i]);
    }
    for (size_t i = 0; i < inner.size(); ++i) {
        a[i] = static_cast<int>(inner[i]);
    }

    double knobs[Eigen::internal::Colamd::NKnobs];
    Eigen::internal::Colamd::set_defaults(knobs);
    result.success = Eigen::internal::Colamd::compute_ordering(
        n_row,
        n_col,
        alen,
        a.data(),
        p.data(),
        knobs,
        result.stats.data());

    result.permutation.resize(col_count, std::ptrdiff_t{0});
    for (size_t i = 0; i < col_count; ++i) {
        result.permutation[i] = static_cast<std::ptrdiff_t>(p[i]);
    }
    return result;
}

std::vector<std::ptrdiff_t> BuildRandomColumnRows(
    std::mt19937_64& rng,
    const size_t row_count,
    const bool force_dense)
{
    std::vector<std::ptrdiff_t> rows;
    if (row_count == 0) {
        return rows;
    }

    std::uniform_int_distribution<size_t> row_dist(0, row_count - 1);
    std::bernoulli_distribution make_jumbled(0.35);
    std::bernoulli_distribution keep_duplicates(0.45);

    if (force_dense) {
        rows.resize(row_count);
        std::iota(rows.begin(), rows.end(), std::ptrdiff_t{0});
        std::shuffle(rows.begin(), rows.end(), rng);
        return rows;
    }

    const size_t max_len = std::max<size_t>(2, std::min<size_t>(row_count + 3, 18));
    std::uniform_int_distribution<size_t> len_dist(0, max_len);
    const size_t len = len_dist(rng);
    rows.reserve(len);
    for (size_t i = 0; i < len; ++i) {
        rows.push_back(static_cast<std::ptrdiff_t>(row_dist(rng)));
    }

    if (!make_jumbled(rng)) {
        std::sort(rows.begin(), rows.end());
        if (!keep_duplicates(rng)) {
            rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        }
    } else {
        std::shuffle(rows.begin(), rows.end(), rng);
    }

    return rows;
}

CscPatternBuffers MakeRandomCscPattern(
    std::mt19937_64& rng,
    const size_t row_count,
    const size_t col_count)
{
    CscPatternBuffers csc;
    csc.outer.assign(col_count + 1, 0);
    csc.inner.clear();

    std::bernoulli_distribution make_dense_col(0.12);

    for (size_t col = 0; col < col_count; ++col) {
        csc.outer[col] = csc.inner.size();
        const std::vector<std::ptrdiff_t> rows =
            BuildRandomColumnRows(rng, row_count, make_dense_col(rng));
        for (const std::ptrdiff_t row : rows) {
            csc.inner.push_back(static_cast<size_t>(row));
        }
    }
    csc.outer[col_count] = csc.inner.size();
    return csc;
}

StructuralSignature ComputeStructuralSignature(
    const std::span<const size_t> outer,
    const std::span<const size_t> inner,
    const size_t row_count,
    const std::span<const std::ptrdiff_t> permutation)
{
    StructuralSignature sig{};
    if (row_count == 0 || permutation.empty()) {
        return sig;
    }

    std::vector<size_t> inverse(permutation.size(), 0);
    for (size_t pos = 0; pos < permutation.size(); ++pos) {
        inverse[static_cast<size_t>(permutation[pos])] = pos;
    }

    std::vector<std::vector<size_t>> row_positions(row_count);
    for (size_t original_col = 0; original_col < permutation.size(); ++original_col) {
        const size_t permuted_pos = inverse[original_col];
        for (size_t p = outer[original_col]; p < outer[original_col + 1]; ++p) {
            const size_t row = inner[p];
            if (row < row_count) {
                row_positions[row].push_back(permuted_pos);
            }
        }
    }

    for (auto& positions : row_positions) {
        std::sort(positions.begin(), positions.end());
        positions.erase(std::unique(positions.begin(), positions.end()), positions.end());
        if (positions.empty()) {
            continue;
        }
        ++sig.active_rows;
    }

    const size_t n = permutation.size();
    std::vector<size_t> first_row_element(row_count, n);
    for (size_t col = 0; col < n; ++col) {
        const size_t original_col = static_cast<size_t>(permutation[col]);
        for (size_t p = outer[original_col]; p < outer[original_col + 1]; ++p) {
            const size_t row = inner[p];
            if (row < row_count) {
                first_row_element[row] = std::min(first_row_element[row], col);
            }
        }
    }

    const size_t diag_bound = std::min(row_count, n);
    for (size_t row = 0; row < diag_bound; ++row) {
        first_row_element[row] = std::min(first_row_element[row], row);
    }

    std::vector<size_t> etree(n, n);
    std::vector<size_t> root(n, n);
    std::vector<size_t> parent(n, n);

    auto find_root = [&parent](size_t node) {
        while (parent[node] != node) {
            parent[node] = parent[parent[node]];
            node = parent[node];
        }
        return node;
    };

    for (size_t col = 0; col < n; ++col) {
        bool found_diag = col >= row_count;
        parent[col] = col;
        size_t cset = col;
        root[cset] = col;
        etree[col] = n;

        const size_t original_col = static_cast<size_t>(permutation[col]);
        size_t p = outer[original_col];
        while (p < outer[original_col + 1] || !found_diag) {
            size_t row = col;
            if (p < outer[original_col + 1]) {
                row = inner[p++];
            } else {
                found_diag = true;
            }
            if (row == col) {
                found_diag = true;
            }
            if (row >= row_count) {
                continue;
            }

            const size_t first = first_row_element[row];
            if (first >= col) {
                continue;
            }
            const size_t rset = find_root(first);
            const size_t rroot = root[rset];
            if (rroot != col) {
                etree[rroot] = col;
                parent[cset] = rset;
                cset = rset;
                root[cset] = col;
            }
        }
    }

    for (size_t col = 0; col < n; ++col) {
        const size_t parent_col = etree[col];
        sig.elimination_parent_sum += static_cast<uint64_t>(parent_col);

        size_t depth = 0;
        size_t current = col;
        while (current < n && etree[current] < n) {
            ++depth;
            current = etree[current];
            if (depth > n) {
                break;
            }
        }
        sig.elimination_depth_sum += static_cast<uint64_t>(depth);
    }

    return sig;
}

void CompareWithEigenColamd(
    const size_t row_count,
    const size_t col_count,
    const CscPatternBuffers& csc)
{
    sparse_qr::CscPatternView<size_t> view{
        row_count,
        col_count,
        std::span<const size_t>(csc.outer),
        std::span<const size_t>(csc.inner)
    };
    sparse_qr::ColamdWorkspace<std::ptrdiff_t> ours;
    const bool ours_ok = sparse_qr::computeColamdOrdering(view, ours);

    const EigenColamdResult eigen =
        RunEigenColamd(row_count, col_count, std::span<const size_t>(csc.outer), std::span<const size_t>(csc.inner));

    EXPECT_EQ(ours_ok, eigen.success);
    EXPECT_EQ(static_cast<int>(ours.status()), eigen.stats[Eigen::internal::Colamd::Status]);

    const auto ours_stats = ours.statistics();
    EXPECT_EQ(
        ours_stats[static_cast<size_t>(sparse_qr::ColamdStatSlot::dense_row)],
        static_cast<std::ptrdiff_t>(eigen.stats[Eigen::internal::Colamd::DenseRow]));
    EXPECT_EQ(
        ours_stats[static_cast<size_t>(sparse_qr::ColamdStatSlot::dense_col)],
        static_cast<std::ptrdiff_t>(eigen.stats[Eigen::internal::Colamd::DenseCol]));
    EXPECT_EQ(
        ours_stats[static_cast<size_t>(sparse_qr::ColamdStatSlot::defrag_count)],
        static_cast<std::ptrdiff_t>(eigen.stats[Eigen::internal::Colamd::DefragCount]));

    if (!ours_ok || !eigen.success) {
        return;
    }

    const std::span<const std::ptrdiff_t> ours_perm = ours.ordering();
    ASSERT_EQ(ours_perm.size(), eigen.permutation.size());

    const bool exact_match = std::equal(ours_perm.begin(), ours_perm.end(), eigen.permutation.begin());
    if (exact_match) {
        return;
    }

    const bool ours_is_permutation = IsPermutation(ours_perm, col_count);
    const bool eigen_is_permutation =
        IsPermutation(std::span<const std::ptrdiff_t>(eigen.permutation), col_count);
    EXPECT_TRUE(ours_is_permutation);
    EXPECT_TRUE(eigen_is_permutation);
    if (!ours_is_permutation || !eigen_is_permutation) {
        return;
    }

    const StructuralSignature ours_sig =
        ComputeStructuralSignature(csc.outer, csc.inner, row_count, ours_perm);
    const StructuralSignature eigen_sig =
        ComputeStructuralSignature(csc.outer, csc.inner, row_count, eigen.permutation);
    EXPECT_EQ(ours_sig, eigen_sig);
}

} // namespace

TEST(SparseQrColamdTests, producesValidPermutationForRegularPattern) {
    Matrix<double> dense = {
        {1.0, 0.0, 2.0, 0.0},
        {0.0, 3.0, 0.0, 4.0},
        {5.0, 0.0, 6.0, 0.0},
        {0.0, 7.0, 0.0, 8.0},
        {9.0, 0.0, 1.0, 0.0}
    };
    SparseMatrix<> sparse(dense);
    const CscPatternBuffers csc = BuildCscPattern(sparse);

    sparse_qr::CscPatternView<size_t> view{
        sparse.rows_size(),
        sparse.cols_size(),
        std::span<const size_t>(csc.outer),
        std::span<const size_t>(csc.inner)
    };
    sparse_qr::ColamdWorkspace<std::ptrdiff_t> workspace;

    ASSERT_TRUE(sparse_qr::computeColamdOrdering(view, workspace));
    EXPECT_TRUE(IsPermutation(workspace.ordering(), sparse.cols_size()));
    EXPECT_EQ(workspace.status(), sparse_qr::ColamdStatus::ok);
}

TEST(SparseQrColamdTests, acceptsJumbledColumnsAndReportsNotice) {
    const std::vector<size_t> outer = {0, 3, 5, 7};
    const std::vector<size_t> inner = {
        2, 0, 0,
        1, 0,
        2, 1
    };

    sparse_qr::CscPatternView<size_t> view{
        3,
        3,
        std::span<const size_t>(outer),
        std::span<const size_t>(inner)
    };
    sparse_qr::ColamdWorkspace<std::ptrdiff_t> workspace;

    ASSERT_TRUE(sparse_qr::computeColamdOrdering(view, workspace));
    EXPECT_EQ(workspace.status(), sparse_qr::ColamdStatus::ok_but_jumbled);
    EXPECT_TRUE(IsPermutation(workspace.ordering(), 3));
}

TEST(SparseQrColamdTests, rejectsOutOfRangeRowIndex) {
    const std::vector<size_t> outer = {0, 1, 1};
    const std::vector<size_t> inner = {3};

    sparse_qr::CscPatternView<size_t> view{
        3,
        2,
        std::span<const size_t>(outer),
        std::span<const size_t>(inner)
    };
    sparse_qr::ColamdWorkspace<std::ptrdiff_t> workspace;

    EXPECT_FALSE(sparse_qr::computeColamdOrdering(view, workspace));
    EXPECT_EQ(workspace.status(), sparse_qr::ColamdStatus::error_row_index_out_of_bounds);
}

TEST(SparseQrColamdTests, randomCscPatternsMatchEigenColamdStatsAndOrderingOrStructure) {
    std::mt19937_64 rng(0xC01A6DULL);
    std::uniform_int_distribution<size_t> rows_dist(1, 60);
    std::uniform_int_distribution<size_t> cols_dist(1, 60);

    constexpr size_t kTrials = 300;
    for (size_t trial = 0; trial < kTrials; ++trial) {
        const size_t rows = rows_dist(rng);
        const size_t cols = cols_dist(rng);
        const CscPatternBuffers csc = MakeRandomCscPattern(rng, rows, cols);
        SCOPED_TRACE(::testing::Message()
                     << "trial=" << trial
                     << " rows=" << rows
                     << " cols=" << cols
                     << " nnz=" << csc.inner.size());
        CompareWithEigenColamd(rows, cols, csc);
    }
}
