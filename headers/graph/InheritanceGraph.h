#ifndef INHERITANCE_GRAPH_H
#define INHERITANCE_GRAPH_H

#include "Graph.h"

// 1
template <typename VertexType, typename WeightType = double>
class DirectedWeightedGraph : public Graph <
        VertexType,
        WeightType,
        DirectedPolicy<VertexType, WeightType>,
        WeightedPolicy<VertexType, WeightType>
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
        DirectedPolicy<VertexType, WeightType>,
        UnweightedPolicy<VertexType, WeightType>
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
        UndirectedPolicy<VertexType, WeightType>,
        WeightedPolicy<VertexType, WeightType>
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
        UndirectedPolicy<VertexType, WeightType>,
        UnweightedPolicy<VertexType, WeightType>
> {
public:
    UndirectedUnweightedGraph() = default;
    ~UndirectedUnweightedGraph() override = default;
};

#endif // INHERITANCE_GRAPH_H