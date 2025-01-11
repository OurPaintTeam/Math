#ifndef MINIMIZEROPTIMIZER_HEADERS_GRAPH_POLITICIANS_H_
#define MINIMIZEROPTIMIZER_HEADERS_GRAPH_POLITICIANS_H_

struct DirectedPolicy {
    static constexpr bool isDirected = true;
};

struct UndirectedPolicy {
    static constexpr bool isDirected = false;
};

struct WeightedPolicy {
    static constexpr bool isWeighted = true;
};

struct UnweightedPolicy {
    static constexpr bool isWeighted = false;
};

#endif // ! MINIMIZEROPTIMIZER_HEADERS_GRAPH_POLITICIANS_H_