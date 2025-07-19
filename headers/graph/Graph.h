#ifndef MINIMIZEROPTIMIZER_HEADERS_GRAPH_GRAPH_H_
#define MINIMIZEROPTIMIZER_HEADERS_GRAPH_GRAPH_H_

#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <stdexcept>
#include <queue>
#include <algorithm>

#include "Politicians.h"
#include "GraphObjects.h"

#define GRAPH_TEMPLATE_PARAMS \
    typename VertexType, \
    typename WeightType, \
    typename DirectedPolicyType, \
    typename WeightedPolicyType


template<typename VertexType,
        typename WeightType = double,
        typename DirectedPolicyType = UndirectedPolicy,
        typename WeightedPolicyType = UnweightedPolicy>
class Graph :
        private DirectedPolicyType,
        private WeightedPolicyType {

    using EdgeType = Edge<VertexType, WeightType>;


public:
    Graph() = default;

    virtual ~Graph() = default;

    template<typename... Args>
    void addVertex(const Args& ... vertex);

    template<typename... Args>
    bool removeVertex(const Args& ... vertex);

    bool addEdge(const VertexType& from, const VertexType& to, const WeightType& weight = WeightType());

    bool removeEdge(const VertexType& from, const VertexType& to);

    bool setEdgeWeight(const VertexType& from, const VertexType& to, const WeightType& weight);

    WeightType getEdgeWeight(const VertexType& from, const VertexType& to) const;

    // TODO test and Imp
    std::vector<EdgeType> getAllEdges() const {
        std::unordered_set<EdgeType> targetEdges;
        for (const auto& [vertex, edges]: _adjacencyList) {
            for (const EdgeType& edge: edges) {
                EdgeType edge1(edge.to, edge.from, edge.weight);
                if (!targetEdges.contains(edge1)) {
                    targetEdges.emplace(edge);
                }
            }
        }
        return {targetEdges.begin(), targetEdges.end()};
    }

    bool isDirected() const;

    bool isWeighted() const;

    bool hasVertex(const VertexType& v) const;

    template<typename... Args>
    bool hasVertices(const Args& ... vertex);

    bool hasEdge(const VertexType& from, const VertexType& to) const;

    std::vector<VertexType> getVertices() const;

    std::unordered_map<VertexType, std::vector<EdgeType>> getAdjacencyList() const;

    // TODO test and Imp
    std::vector<EdgeType> getVertexEdges(const VertexType& v) const {
        auto it = _adjacencyList.find(v);
        if (it != _adjacencyList.end()) {
            return it->second;
        }
        throw std::runtime_error("Vertex not found");
    }

    void printGraph(std::ostream& os) const {
        for (const auto& [vertex, edges]: _adjacencyList) {
            os << vertex << " -> ";
            for (const auto& edge: edges) {
                os << edge.from << edge.to;
                if (WeightedPolicyType::isWeighted) {
                    os << "(" << edge.weight << ")";
                }
                os << ", ";
            }
            os << std::endl;
        }
    }

    // TODO test and Imp
    std::string printGraph() const {
        std::string str;
        for (const auto& [vertex, edges]: _adjacencyList) {
            str += vertex + " -> ";
            for (const auto& edge: edges) {
                str += edge.from + edge.to + " (" + edge.weight + ") ";
            }
            str += '\n';
        }
        return str;
    }

    std::vector<VertexType> findConnectedComponent(const VertexType& start) const {
        std::vector<VertexType> component;
        if (hasVertex(start)) {
            std::unordered_set<VertexType> visited;
            DFS(start, visited, component);
        }
        return component;
    }

    template<typename Predicate>
    std::vector<VertexType> findComponentByEdgeType(const VertexType& start, Predicate edgePredicate) const {
        std::vector<VertexType> component;
        if (!hasVertex(start)) {
            return component;
        }

        std::unordered_set<VertexType> visited;
        std::queue<VertexType> queue;
        queue.push(start);
        visited.insert(start);

        while (!queue.empty()) {
            VertexType current = queue.front();
            queue.pop();
            component.push_back(current);

            const auto& edges = _adjacencyList.at(current);
            for (const auto& edge : edges) {
                if (edgePredicate(edge) && visited.find(edge.to) == visited.end()) {
                    visited.insert(edge.to);
                    queue.push(edge.to);
                }
            }
        }

        return component;
    }

    size_t vertexCount() const {
        return _vertices.size();
    }

    size_t edgeCount() const {
        std::unordered_set<EdgeType> targetEdges;
        for (const auto& [vertex, edges]: _adjacencyList) {
            for (const auto& edge: edges) {
                targetEdges.emplace(edge);
            }
        }
        if constexpr (DirectedPolicyType::isDirected) {
            return targetEdges.size();
        } else {
            return targetEdges.size() / 2;
        }
    }

    // TODO test and Imp
    std::vector<VertexType> traverse(const VertexType& start, SearchType type) const {
        return {};
    }

    // TODO test and Imp
    // Является ли граф связным (для ориентированных)
    // и
    // сильно связным (для неориентированных)
    bool isConnected() const {
        return false;
    }

    // TODO test and Imp
    // Выделить все компоненты связности (или сильной связности)
    std::vector<std::vector<VertexType>> connectedComponents() const {
        std::vector<std::vector<VertexType>> components;
        std::unordered_set<VertexType> visited;

        for (const auto& vertex: _vertices) {
            if (visited.find(vertex) == visited.end()) {
                std::vector<VertexType> component;
                DFS(vertex, visited, component);
                components.push_back(std::move(component));
            }
        }

        return components;
    }

    // TODO test and Imp
    // Проверка графа на ацикличность (для ориентированных - DAG)
    bool isAcyclic() const {
        return false;
    }

    // TODO test and Imp
    // Топологическая сортировка (актуально для ориентированного ацикличного графа)
    std::vector<VertexType> topologicalSort() const {
        return {};
    };

    // TODO test and Imp
    // желательно использовать using
    PathResult<VertexType, WeightType> dijkstra(const VertexType& start) const {
        return PathResult<VertexType, WeightType>{};
    }

    // TODO test and Imp
    PathResult<VertexType, WeightType> bellmanFord(const VertexType& start) const {
        return PathResult<VertexType, WeightType>{};
    }

    // TODO test and Imp
    // Алгоритм Флойда — Уоршелла: возвращает матрицу кратчайших путей между всеми парами вершин
    std::unordered_map<VertexType, std::unordered_map<VertexType, WeightType>> floydWarshall() const {
        return {};
    }

    // TODO test and Imp
    std::vector<WeightType> kruskalMST() const {
        return {};
    }

    // TODO test and Imp
    std::vector<WeightType> primMST(const VertexType& start) const {
        return {};
    }

    // TODO test and Imp
    // Существует ли Эйлеров цикл / путь (неориентированный, ориентированный)
    bool hasEulerianPath() const {
        return false;
    }

    // TODO test and Imp
    // Построение Эйлерова пути или цикла, если он существует
    std::vector<VertexType> eulerianPath() const {
        return {};
    }

    // TODO test and Imp
    // Поиск гамильтонова пути (или цикла)
    // это NP полная задача
    std::vector<VertexType> hamiltonianPath() const {
        return {};
    }

    // TODO test and Imp
    // Алгоритм Форда-Фулкерсона (Edmonds-Karp) для вычисления максимального потока
    WeightType maxFlow(const VertexType& source, const VertexType& sink) {
        return WeightType{};
    }

    // TODO test and Imp
    // Поиск максимального паросочетания в двудольном графе (алгоритм Хопкрофта–Карпа)
    WeightType bipartiteMatching() const {
        return WeightType{};
    }

    // TODO test and Imp
    // Транспонирования графа (актально для ориентированных графов)
    Graph transpose() const {
        return {};
    }

    // TODO test and Imp
    // Построение дополнительного графа (для неориентированного, невзвешенного)
    Graph complement() const {
        return {};
    }

    // TODO test and Imp
    // Выделение подграфа по множеству вершин
    Graph<VertexType, WeightType, UndirectedPolicy, WeightedPolicy> subGraph(std::vector<VertexType>& vertices) const {
        Graph<VertexType, WeightType, UndirectedPolicy, WeightedPolicy> sub;
        for (const auto& v: vertices) {
            sub.addVertex(v);
        }

        std::unordered_set<VertexType> vertexSet(vertices.begin(), vertices.end());


        for (const auto& vertex: vertices) {
            auto it = _adjacencyList.find(vertex);
            if (it != _adjacencyList.end()) {
                for (const auto& edge: it->second) {
                    if (vertexSet.contains(edge.to)) {
                        // Для неориентированного графа добавляем только одно направление, чтобы избежать дублирования
                        if constexpr (DirectedPolicyType::isDirected) {
                            sub.addEdge(edge.from, edge.to, edge.weight);
                        } else {
                            if (edge.from < edge.to) {
                                sub.addEdge(edge.from, edge.to, edge.weight);
                            }
                        }
                    }
                }
            }
        }
        return sub;
    }

    // Maybe next: graph iterators


private:

    std::unordered_set<VertexType> _vertices;

    std::unordered_map<VertexType, std::vector<EdgeType>> _adjacencyList;

protected:

    void DFS(const VertexType& v, std::unordered_set<VertexType>& visited, std::vector<VertexType>& component) const {
        visited.insert(v);
        component.push_back(v);
        if (_adjacencyList.find(v) != _adjacencyList.end()) {
            for (const auto& e: _adjacencyList.at(v)) {
                if (visited.find(e.to) == visited.end()) {
                    DFS(e.to, visited, component);
                }
            }
        }
    }

private:

    bool remVertex(const VertexType& v) {
        // Find the vertex in the set of vertices
        auto it = _vertices.find(v);
        if (it == _vertices.end()) {
            return false; // Vertex not found
        }

        // Remove the vertex from the set
        _vertices.erase(it);

        if (DirectedPolicyType::isDirected) {
            // For a directed graph, remove outgoing edges
            _adjacencyList.erase(v); // Remove outgoing edges

            // Remove incoming edges by iterating over all vertices
            for (auto& [vertex, edges]: _adjacencyList) {
                edges.erase(
                        std::remove_if(edges.begin(), edges.end(), [&](const EdgeType& edge) { return edge.to == v; }),
                        edges.end()
                );
            }
        } else {
            // For an undirected graph, remove both outgoing and incoming edges
            auto adjIt = _adjacencyList.find(v);
            if (adjIt != _adjacencyList.end()) {
                auto& vec = adjIt->second;

                // For each vertex that has an edge to 'v', remove the edge to 'v'
                for (const auto& elem: vec) {
                    VertexType to = elem.to;

                    auto toIt = _adjacencyList.find(to);
                    if (toIt != _adjacencyList.end()) {
                        auto& vec2 = toIt->second;
                        vec2.erase(std::remove_if(
                                           vec2.begin(), vec2.end(), [&](const EdgeType& edge) { return edge.to == v; }),
                                   vec2.end());
                    }
                }
            }

            // Remove the vertex from the adjacency list
            _adjacencyList.erase(v);
        }

        return true; // Vertex successfully removed
    }
};


template<GRAPH_TEMPLATE_PARAMS>
template<typename... Args>
void Graph<VertexType, WeightType, DirectedPolicyType, WeightedPolicyType>::addVertex(const Args& ... vertex) {
    (void) std::initializer_list<int>{
            (
                    _vertices.insert(vertex),
                            _adjacencyList.emplace(vertex, std::vector<EdgeType>{}),
                            0
            )...
    };
}

template<GRAPH_TEMPLATE_PARAMS>
template<typename... Args>
bool Graph<VertexType, WeightType, DirectedPolicyType, WeightedPolicyType>::removeVertex(const Args& ... vertices) {
    bool success = true;
    (void) std::initializer_list<int>{
            (
                    success &= remVertex(vertices),
                            0
            )...
    };
    return success;
}

template<GRAPH_TEMPLATE_PARAMS>
bool Graph<VertexType, WeightType, DirectedPolicyType, WeightedPolicyType>::addEdge(const VertexType& from,
                                                                                    const VertexType& to,
                                                                                    const WeightType& weight) {
    if (_vertices.find(from) == _vertices.end() || _vertices.find(to) == _vertices.end()) {
        return false;
    }

    if constexpr (WeightedPolicyType::isWeighted) {
        if (weight == WeightType()) {
            throw std::invalid_argument("should be use weight arg in weighted graph.");
        }
    } else {
        if (weight != WeightType()) {
            throw std::invalid_argument("do not use weight arg in unweighted graph.");
        }
    }

    if constexpr (DirectedPolicyType::isDirected) {
        if (hasEdge(from, to)) {
            return false;
        }
    } else {
        if (hasEdge(from, to) || hasEdge(to, from)) {
            return false;
        }
    }

    _adjacencyList[from].emplace_back(from, to, weight);
    if constexpr (DirectedPolicyType::isDirected == false) {
        _adjacencyList[to].emplace_back(to, from, weight);
    }
    return true;
}

template<GRAPH_TEMPLATE_PARAMS>
bool Graph<VertexType, WeightType, DirectedPolicyType, WeightedPolicyType>::removeEdge(const VertexType& from,
                                                                                       const VertexType& to) {
    if (_vertices.find(from) == _vertices.end() || _vertices.find(to) == _vertices.end()) {
        return false;
    }

    bool removed = false;

    // Remove edge from "from" to "to"
    auto& fromEdges = _adjacencyList[from];
    size_t originalSize = fromEdges.size();
    fromEdges.erase(
            std::remove_if(fromEdges.begin(), fromEdges.end(), [&](const EdgeType& edge) { return edge.to == to; }),
            fromEdges.end()
    );
    if (fromEdges.size() != originalSize) {
        removed = true;
    }

    // If the graph is undirected, also remove the edge from "to" to "from"
    if constexpr (!DirectedPolicyType::isDirected) {
        auto& toEdges = _adjacencyList[to];
        originalSize = toEdges.size();
        toEdges.erase(
                std::remove_if(toEdges.begin(), toEdges.end(), [&](const EdgeType& edge) { return edge.to == from; }),
                toEdges.end()
        );
        if (toEdges.size() != originalSize) {
            removed = true;
        }
    }

    return removed;
}

template<GRAPH_TEMPLATE_PARAMS>
bool Graph<VertexType, WeightType, DirectedPolicyType, WeightedPolicyType>::isWeighted() const {
    return WeightedPolicyType::isWeighted;
}

template<GRAPH_TEMPLATE_PARAMS>
bool Graph<VertexType, WeightType, DirectedPolicyType, WeightedPolicyType>::isDirected() const {
    return DirectedPolicyType::isDirected;
}

template<GRAPH_TEMPLATE_PARAMS>
bool Graph<VertexType, WeightType, DirectedPolicyType, WeightedPolicyType>::setEdgeWeight(const VertexType& from,
                                                                                          const VertexType& to,
                                                                                          const WeightType& weight) {
    if (_vertices.find(from) == _vertices.end() || _vertices.find(to) == _vertices.end()) {
        return false;
    }

    if constexpr (!WeightedPolicyType::isWeighted) {
        throw std::invalid_argument("Cannot set weight on an unweighted graph.");
    }

    auto itFrom = _adjacencyList.find(from);

    bool found = false;
    for (auto& edge: itFrom->second) {
        if (edge.to == to) {
            edge.weight = weight;
            found = true;
            break;
        }
    }

    if constexpr (!DirectedPolicyType::isDirected) {
        auto itTo = _adjacencyList.find(to);
        for (auto& edge: itTo->second) {
            if (edge.to == from) {
                edge.weight = weight;
                break;
            }
        }
    }

    return found;
}

template<GRAPH_TEMPLATE_PARAMS>
WeightType Graph<VertexType, WeightType, DirectedPolicyType, WeightedPolicyType>::getEdgeWeight(const VertexType& from,
                                                                                                const VertexType& to) const {
    if (_vertices.find(from) == _vertices.end() || _vertices.find(to) == _vertices.end()) {
        throw std::invalid_argument("One or both vertices do not exist.");
    }

    if constexpr (!WeightedPolicyType::isWeighted) {
        throw std::invalid_argument("Cannot get weight on an unweighted graph.");
    }

    auto it = _adjacencyList.find(from);
    if (it != _adjacencyList.end()) {
        for (const auto& edge: it->second) {
            if (edge.to == to) {
                return edge.weight;
            }
        }
    }

    throw std::invalid_argument("EdgeType does not exist.");
}

template<GRAPH_TEMPLATE_PARAMS>
bool Graph<VertexType, WeightType, DirectedPolicyType, WeightedPolicyType>::hasVertex(const VertexType& v) const {
    return _vertices.find(v) != _vertices.end();
}

template<GRAPH_TEMPLATE_PARAMS>
template<typename... Args>
bool Graph<VertexType, WeightType, DirectedPolicyType, WeightedPolicyType>::hasVertices(const Args& ... vertex) {
    return ((_vertices.find(vertex) != _vertices.end()) && ...);
}

template<GRAPH_TEMPLATE_PARAMS>
bool Graph<VertexType, WeightType, DirectedPolicyType, WeightedPolicyType>::hasEdge(const VertexType& from,
                                                                                    const VertexType& to) const {
    auto it = _adjacencyList.find(from);
    if (it != _adjacencyList.end()) {
        return std::any_of(it->second.begin(), it->second.end(),
                           [&](const EdgeType& e) { return e.to == to; });
    }
    return false;
}

template<GRAPH_TEMPLATE_PARAMS>
std::vector<VertexType> Graph<VertexType, WeightType, DirectedPolicyType, WeightedPolicyType>::getVertices() const {
    return std::vector<VertexType>(_vertices.begin(), _vertices.end());
}

template<GRAPH_TEMPLATE_PARAMS>
std::unordered_map<VertexType, std::vector<Edge<VertexType, WeightType>>>
Graph<VertexType, WeightType, DirectedPolicyType, WeightedPolicyType>::getAdjacencyList() const {
    return _adjacencyList;
}

#endif // ! MINIMIZEROPTIMIZER_HEADERS_GRAPH_GRAPH_H_