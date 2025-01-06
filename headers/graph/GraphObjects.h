#ifndef GRAPH_OBJECTS_H
#define GRAPH_OBJECTS_H

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

#endif // GRAPH_OBJECTS_H