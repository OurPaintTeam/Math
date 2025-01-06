#ifndef GRAPH_H
#define GRAPH_H

#include <unordered_map>
#include <unordered_set>
#include <vector>

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

public:

    Graph() = default;
    virtual ~Graph() = default;


    void addVertex(const VertexType& v) {
        _vertices.insert(v);
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


private:
    std::unordered_set<VertexType> _vertices;

    std::unordered_map<VertexType, std::vector<Edge<VertexType, WeightType>>> _adjacencyList;
};

#endif // ! GRAPH_H
