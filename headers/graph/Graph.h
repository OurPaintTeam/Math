#ifndef GRAPH_H
#define GRAPH_H

#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <stdexcept>

#include "Politicians.h"
#include "GraphObjects.h"


template <
        typename VertexType,
        typename WeightType = double,
        typename DirectedPolicyType = UndirectedPolicy<VertexType, WeightType>,
        typename WeightedPolicyType = UnweightedPolicy<VertexType, WeightType>
        >
class Graph :
        private DirectedPolicyType,
        private WeightedPolicyType {

private:
    std::unordered_set<VertexType> _vertices;

    std::unordered_map<VertexType, std::vector<Edge<VertexType, WeightType>>> _adjacencyList;

protected:
    void DFS(const VertexType& v, std::unordered_set<VertexType>& visited, std::vector<VertexType>& component) const {
        visited.insert(v);
        component.push_back(v);
        if (_adjacencyList.find(v) != _adjacencyList.end()) {
            for (const auto& e : _adjacencyList.at(v)) {
                if (visited.find(e.to) == visited.end()) {
                    DFS(e.to, visited, component);
                }
            }
        }
    }

public:

    Graph() = default;
    virtual ~Graph() = default;


    void addVertex(const VertexType& v) {
        _vertices.insert(v);
    }

    bool removeVertex(const VertexType& v) {
        auto it = _vertices.find(v);
        if (it == _vertices.end()) {
            return false; // Vertex not found
        }

        _vertices.erase(it);
        _adjacencyList.erase(v); // Remove outgoing edges

        // Remove incoming edges
        for (auto& [vertex, edges] : _adjacencyList) {
            edges.erase(
                    std::remove_if(edges.begin(), edges.end(),
                                   [&](const Edge<VertexType, WeightType>& edge) { return edge.to == v; }),
                    edges.end()
            );
        }

        return true;
    }

    bool addEdge(const VertexType& from, const VertexType& to, const WeightType& weight = WeightType()) {
        if (_vertices.find(from) != _vertices.end() && _vertices.find(to) != _vertices.end()) {
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
                _adjacencyList[from].emplace_back(from, to, weight);
            } else {
                _adjacencyList[from].emplace_back(from, to, weight);
                _adjacencyList[to].emplace_back(to, from, weight);
            }
            return true;
        }
        return false;
    }

    bool removeEdge(const VertexType& from, const VertexType& to) {
        if (_vertices.find(from) == _vertices.end() || _vertices.find(to) == _vertices.end()) {
            return false;
        }

        bool removed = false;

        // Remove edge from "from" to "to"
        auto &fromEdges = _adjacencyList[from];
        size_t originalSize = fromEdges.size();
        fromEdges.erase(
                std::remove_if(fromEdges.begin(), fromEdges.end(),
                               [&](const Edge<VertexType, WeightType>& edge) { return edge.to == to; }),
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
                    std::remove_if(toEdges.begin(), toEdges.end(),
                                   [&](const Edge<VertexType, WeightType>& edge) { return edge.to == from; }),
                    toEdges.end()
            );
            if (toEdges.size() != originalSize) {
                removed = true;
            }
        }
        return removed;
    }


    bool setEdgeWeight(const VertexType& from, const VertexType& to, const WeightType& weight) {
        if (_vertices.find(from) != _vertices.end() && _vertices.find(to) != _vertices.end()) {
            if constexpr (!WeightedPolicyType::isWeighted) {
                throw std::invalid_argument("Cannot set weight on an unweighted graph.");
            }
            auto itFrom = _adjacencyList.find(from);

            bool found = false;
            for (auto& edge : itFrom->second) {
                if (edge.to == to) {
                    edge.weight = weight;
                    found = true;
                    break;
                }
            }
            if constexpr (!DirectedPolicyType::isDirected) {
                auto itTo = _adjacencyList.find(to);
                for (auto& edge : itTo->second) {
                    if (edge.to == from) {
                        edge.weight = weight;
                        break;
                    }
                }
            }
            return found;
        }
        return false;
    }

    WeightType getEdgeWeight(const VertexType& from, const VertexType& to) const {
        if (_vertices.find(from) != _vertices.end() && _vertices.find(to) != _vertices.end()) {
            if constexpr (!WeightedPolicyType::isWeighted) {
                throw std::invalid_argument("Cannot get weight on an unweighted graph.");
            }
            auto it = _adjacencyList.find(from);
            if (it != _adjacencyList.end()) {
                for (const auto& edge : it->second) {
                    if (edge.to == to) {
                        return edge.weight;
                    }
                }
            }
            throw std::invalid_argument("Edge does not exist.");
        }
        throw std::invalid_argument("One or both vertices do not exist.");
    }

    bool hasVertex(const VertexType& v) const {
        return _vertices.find(v) != _vertices.end();
    }

    bool hasEdge(const VertexType& from, const VertexType& to) const {
        auto it = _adjacencyList.find(from);
        if (it != _adjacencyList.end()) {
            return std::any_of(it->second.begin(), it->second.end(),
                               [&](const Edge<VertexType, WeightType>& e) { return e.to == to; });
        }
        return false;
    }

    std::unordered_set<VertexType> getVertices() const {
        return _vertices;
    };

    std::unordered_map<VertexType, std::vector<Edge<VertexType, WeightType>>> getAdjacencyList() const {
        return _adjacencyList;
    }

    void printGraph(std::ostream& os = std::cout) const {
        for (const auto& [vertex, edges] : _adjacencyList) {
            os << vertex << " -> ";
            for (const auto& edge : edges) {
                os << edge.to << " (" << edge.weight << ") ";
            }
            os << std::endl;
        }
    }

    std::vector<VertexType> findConnectedComponent(const VertexType& start) const {
        std::vector<VertexType> component;
        if (hasVertex(start)) {
            std::unordered_set<VertexType> visited;
            DFS(start, visited, component);
        }
        return component;
    }
};

#endif // ! GRAPH_H
