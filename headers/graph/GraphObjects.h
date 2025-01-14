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

    bool operator==(const Edge& other) const {
      return from == other.from && to == other.to && weight == other.weight;
    }

    bool operator!=(const Edge& other) const {
      return from != other.from || to != other.to || weight != other.weight;
    }
};

namespace std {
template<typename VertexType, typename WeightType>
struct hash<Edge<VertexType, WeightType>> {
    std::size_t operator()(const Edge<VertexType, WeightType>& edge) const {
      std::size_t h1 = std::hash<VertexType>{}(edge.from);
      std::size_t h2 = std::hash<VertexType>{}(edge.to);
      std::size_t h3 = std::hash<WeightType>{}(edge.weight);
      return ((h1 ^ (h2 << 1)) >> 1) ^ (h3 << 1);
    }
};
}

#endif // ! MINIMIZEROPTIMIZER_HEADERS_GRAPH_GRAPHOBJECTS_H_