#ifndef POLITICIANS_H_
#define POLITICIANS_H_

#include "GraphObjects.h"

template <typename VertexType, typename WeightType>
struct DirectedPolicy {
    static constexpr bool isDirected = true;
};

template <typename VertexType, typename WeightType>
struct UndirectedPolicy {
    static constexpr bool isDirected = false;
};

template <typename VertexType, typename WeightType>
struct WeightedPolicy {
    static constexpr bool isWeighted = true;
};

template <typename VertexType, typename WeightType>
struct UnweightedPolicy {
    static constexpr bool isWeighted = false;
};

#endif // ! POLITICIANS_H_