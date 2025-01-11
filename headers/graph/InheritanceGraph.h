#ifndef MINIMIZEROPTIMIZER_HEADERS_GRAPH_INHERITANCEGRAPH_H_
#define MINIMIZEROPTIMIZER_HEADERS_GRAPH_INHERITANCEGRAPH_H_

#include "Graph.h"

// 1
template <typename VertexType, typename WeightType = double>
class DirectedWeightedGraph : public Graph <
        VertexType,
        WeightType,
        DirectedPolicy,
        WeightedPolicy
> {
public:
    DirectedWeightedGraph() = default;
    ~DirectedWeightedGraph() override = default;
};



// 2
template <typename VertexType, typename WeightType = double>
class DirectedUnweightedGraph : public Graph <
        VertexType,
        WeightType,
        DirectedPolicy,
        UnweightedPolicy
> {
public:
    DirectedUnweightedGraph() = default;
    ~DirectedUnweightedGraph() override = default;
};



// 3
template <typename VertexType, typename WeightType = double>
class UndirectedWeightedGraph : public Graph <
        VertexType,
        WeightType,
        UndirectedPolicy,
        WeightedPolicy
> {
public:
    UndirectedWeightedGraph() = default;
    ~UndirectedWeightedGraph() override = default;
};



// 4
template <typename VertexType, typename WeightType = double>
class UndirectedUnweightedGraph : public Graph <
        VertexType,
        WeightType,
        UndirectedPolicy,
        UnweightedPolicy
> {
public:
    UndirectedUnweightedGraph() = default;
    ~UndirectedUnweightedGraph() override = default;
};

#endif // ! MINIMIZEROPTIMIZER_HEADERS_GRAPH_INHERITANCEGRAPH_H_