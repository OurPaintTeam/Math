#include <gtest/gtest.h>
#include "../headers/graph/Graph.h"
#include "../headers/graph/InheritanceGraph.h"

// Graph Undirected Unweighted
TEST(GraphTest, AddVertex) {
    Graph<int> graph;
    int vertex = 42;
    graph.addVertex(vertex);

    const auto& vertices = graph.getVertices();
    EXPECT_NE(vertices.find(vertex), vertices.end());
}

TEST(GraphTest, AddEdgeAddEdge2) {
    Graph<std::string, int> g;

    g.addVertex("A");
    g.addVertex("B");
    g.addVertex("C");
    g.addVertex("D");
    g.addVertex("E");

    EXPECT_TRUE(g.hasVertex("A"));
    EXPECT_FALSE(g.hasVertex("F"));

    EXPECT_TRUE(g.addEdge("A", "B"));
    EXPECT_TRUE(g.addEdge("A", "C"));
    EXPECT_TRUE(g.addEdge("B", "D"));
    EXPECT_TRUE(g.addEdge("C", "E"));

    EXPECT_TRUE(g.hasEdge("A", "B"));
    EXPECT_FALSE(g.hasEdge("A", "D"));
}

TEST(GraphTest, FindConnectedComponent) {
    Graph<std::string, int> g;

    g.addVertex("A");
    g.addVertex("B");
    g.addVertex("C");
    g.addVertex("D");
    g.addVertex("E");

    g.addEdge("A", "B");
    g.addEdge("A", "C");
    g.addEdge("B", "D");
    g.addEdge("C", "E");

    auto component = g.findConnectedComponent("A");
    std::unordered_set<std::string> expected_component = {"A", "B", "C", "D", "E"};
    std::unordered_set<std::string> actual_component(component.begin(), component.end());

    EXPECT_EQ(actual_component, expected_component);
}

TEST(GraphTest, FindConnectedComponentSingleVertex) {
    Graph<std::string, int> g;

    g.addVertex("A");

    auto component = g.findConnectedComponent("A");
    std::unordered_set<std::string> expected_component = {"A"};
    std::unordered_set<std::string> actual_component(component.begin(), component.end());

    ASSERT_EQ(actual_component, expected_component);
}

TEST(GraphTest, FindConnectedComponentNoEdges) {
    Graph<std::string, int> g;

    g.addVertex("A");
    g.addVertex("B");
    g.addVertex("C");

    auto component = g.findConnectedComponent("A");
    std::unordered_set<std::string> expected_component = {"A"};
    std::unordered_set<std::string> actual_component(component.begin(), component.end());

    ASSERT_EQ(actual_component, expected_component);
}

TEST(GraphTest, FindConnectedComponentDisjointSets) {
    Graph<std::string, int> g;

    g.addVertex("A");
    g.addVertex("B");
    g.addVertex("C");
    g.addVertex("D");
    g.addVertex("E");
    g.addVertex("F");

    g.addEdge("A", "B");
    g.addEdge("A", "C");
    g.addEdge("D", "E");

    auto componentA = g.findConnectedComponent("A");
    std::unordered_set<std::string> expected_componentA = {"A", "B", "C"};
    std::unordered_set<std::string> actual_componentA(componentA.begin(), componentA.end());

    ASSERT_EQ(actual_componentA, expected_componentA);

    auto componentD = g.findConnectedComponent("D");
    std::unordered_set<std::string> expected_componentD = {"D", "E"};
    std::unordered_set<std::string> actual_componentD(componentD.begin(), componentD.end());

    ASSERT_EQ(actual_componentD, expected_componentD);
}

// DirectedWeightedGraph



// DirectedUnweightedGraph



// UndirectedWeightedGraph



// UndirectedUnweightedGraph

