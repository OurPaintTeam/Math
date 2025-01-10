#ifndef POLITICIANS_H_
#define POLITICIANS_H_

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

#endif // ! POLITICIANS_H_