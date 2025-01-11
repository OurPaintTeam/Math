#ifndef MINIMIZEROPTIMIZER_HEADERS_GRAPH_GRAPHOBJECTS_H_
#define MINIMIZEROPTIMIZER_HEADERS_GRAPH_GRAPHOBJECTS_H_

#include <unordered_map>
#include <optional>

enum class Representation {
    LIST,
    MATRIX
};

// Type of search algorithm
enum class SearchType {
    BFS,
    DFS,
    AStar,
    IDDFS
};

// Result of search algorithm
template <typename VertexType, typename WeightType = double>
struct PathResult {
    std::unordered_map<VertexType, WeightType> distances;
    std::unordered_map<VertexType, std::optional<VertexType>> predecessors;
};

// Base node
template <typename VertexType, typename WeightType>
struct Edge {
    VertexType from;
    VertexType to;
    WeightType weight;

    Edge(const VertexType& f, const VertexType& t, const WeightType& w)
            : from(f), to(t), weight(w) {}
};

#endif // ! MINIMIZEROPTIMIZER_HEADERS_GRAPH_GRAPHOBJECTS_H_