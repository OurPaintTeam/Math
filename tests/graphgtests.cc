#include <gtest/gtest.h>
#include <algorithm>
#include "../headers/graph/Graph.h"
#include "../headers/graph/InheritanceGraph.h"

/*

Tests implementations:

addVertex               = true
removeVertex            = false
addEdge                 = false
removeEdge              = false
setEdgeWeight           = false
getEdgeWeight           = false
getAllEdges             = false
isDirected              = no need
isWeighted              = no need
hasEdge                 = false
getVertices             = no need
getAdjacencyList        = false
getVertexEdges          = false
printGraph(1,2)         = false
findConnectedComponent  = false


*/

TEST(GraphTest, AddVertex) {
    Graph<int> graph;
    int vertex = 42;
    graph.addVertex(vertex);

    std::vector<int> vertices = graph.getVertices();
    auto it = std::find(vertices.begin(), vertices.end(), vertex);
    EXPECT_EQ(*it, vertex);
    EXPECT_EQ(vertices.size(), 1);

    auto adjList = graph.getAdjacencyList();
    EXPECT_EQ(adjList.find(vertex)->first, vertex);
    EXPECT_EQ(adjList.size(), 1);

    graph.addVertex(1, 2, 3, 4, 5);
    graph.hasVertices(1, 2, 3, 4, 5);

    adjList = graph.getAdjacencyList();
    EXPECT_EQ(adjList.find(1)->first, 1);
    EXPECT_EQ(adjList.find(2)->first, 2);
    EXPECT_EQ(adjList.find(3)->first, 3);
    EXPECT_EQ(adjList.find(4)->first, 4);
    EXPECT_EQ(adjList.find(5)->first, 5);

    EXPECT_EQ(adjList.size(), 6);
    EXPECT_EQ(graph.vertexCount(), 6);
}


/*

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

// Test for addEdge with different policy configurations
TEST(GraphTest, AddEdge_FullyCoversFunction) {
    // Weighted and Directed Graph
    {
        DirectedWeightedGraph<int, double> graph;
        graph.addVertex(1);
        graph.addVertex(2);

        // Add edge with valid weight
        EXPECT_NO_THROW({
                            bool result = graph.addEdge(1, 2, 5.0);
                            EXPECT_TRUE(result);
                        });

        // Verify edge exists
        const auto& adjacencyList = graph.getAdjacencyList();
        auto it = adjacencyList.find(1);
        ASSERT_NE(it, adjacencyList.end());
        ASSERT_EQ(it->second.size(), 1);
        EXPECT_EQ(it->second[0].to, 2);
        EXPECT_EQ(it->second[0].weight, 5.0);

        // Attempt to add edge with default weight (should throw)
        EXPECT_THROW({
                         graph.addEdge(1, 2);
                     }, std::invalid_argument);
    }

    // Weighted and Undirected Graph
    {
        UndirectedWeightedGraph<int, double> graph;
        graph.addVertex(1);
        graph.addVertex(2);

        // Add edge with valid weight
        EXPECT_NO_THROW({
                            bool result = graph.addEdge(1, 2, 3.5);
                            EXPECT_TRUE(result);
                        });

        // Verify edge exists in both directions
        const auto& adjacencyList = graph.getAdjacencyList();
        auto it1 = adjacencyList.find(1);
        auto it2 = adjacencyList.find(2);
        ASSERT_NE(it1, adjacencyList.end());
        ASSERT_NE(it2, adjacencyList.end());
        ASSERT_EQ(it1->second.size(), 1);
        ASSERT_EQ(it2->second.size(), 1);
        EXPECT_EQ(it1->second[0].to, 2);
        EXPECT_EQ(it1->second[0].weight, 3.5);
        EXPECT_EQ(it2->second[0].to, 1);
        EXPECT_EQ(it2->second[0].weight, 3.5);

        // Attempt to add edge with default weight (should throw)
        EXPECT_THROW({
                         graph.addEdge(1, 2);
                     }, std::invalid_argument);
    }

    // Unweighted and Directed Graph
    {
        DirectedUnweightedGraph<int, int> graph;
        graph.addVertex(1);
        graph.addVertex(2);

        // Add edge with default weight
        EXPECT_NO_THROW({
                            bool result = graph.addEdge(1, 2);
                            EXPECT_TRUE(result);
                        });

        // Verify edge exists
        const auto& adjacencyList = graph.getAdjacencyList();
        auto it = adjacencyList.find(1);
        ASSERT_NE(it, adjacencyList.end());
        ASSERT_EQ(it->second.size(), 1);
        EXPECT_EQ(it->second[0].to, 2);
        EXPECT_EQ(it->second[0].weight, 0); // Default int

        // Attempt to add edge with non-default weight (should throw)
        EXPECT_THROW({
                         graph.addEdge(1, 2, 10);
                     }, std::invalid_argument);
    }

    // Unweighted and Undirected Graph
    {
        UndirectedUnweightedGraph<int, int> graph;
        graph.addVertex(1);
        graph.addVertex(2);

        // Add edge with default weight
        EXPECT_NO_THROW({
                            bool result = graph.addEdge(1, 2);
                            EXPECT_TRUE(result);
                        });

        // Verify edge exists in both directions
        const auto& adjacencyList = graph.getAdjacencyList();
        auto it1 = adjacencyList.find(1);
        auto it2 = adjacencyList.find(2);
        ASSERT_NE(it1, adjacencyList.end());
        ASSERT_NE(it2, adjacencyList.end());
        ASSERT_EQ(it1->second.size(), 1);
        ASSERT_EQ(it2->second.size(), 1);
        EXPECT_EQ(it1->second[0].to, 2);
        EXPECT_EQ(it1->second[0].weight, 0); // Default int
        EXPECT_EQ(it2->second[0].to, 1);
        EXPECT_EQ(it2->second[0].weight, 0); // Default int

        // Attempt to add edge with non-default weight (should throw)
        EXPECT_THROW({
                         graph.addEdge(1, 2, 5);
                     }, std::invalid_argument);
    }

    // Attempt to add edge with non-existing vertices
    {
        using AnyGraph = Graph<int>;
        AnyGraph graph;
        graph.addVertex(1);

        // 'to' vertex does not exist
        EXPECT_FALSE(graph.addEdge(1, 2));

        // Both vertices do not exist
        EXPECT_FALSE(graph.addEdge(3, 4));
    }
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

// Test for setEdgeWeight
TEST(GraphTest, SetEdgeWeight_FullyCoversFunction) {
    // Weighted and Directed Graph
    {
        DirectedWeightedGraph<int, int> graph;
        graph.addVertex(1);
        graph.addVertex(2);
        graph.addEdge(1, 2, 10);

        // Set existing edge weight
        EXPECT_NO_THROW({
                            bool result = graph.setEdgeWeight(1, 2, 20);
                            EXPECT_TRUE(result);
                        });

        // Verify weight is updated
        const auto& adjacencyList = graph.getAdjacencyList();
        auto it = adjacencyList.find(1);
        ASSERT_NE(it, adjacencyList.end());
        ASSERT_EQ(it->second.size(), 1);
        EXPECT_EQ(it->second[0].weight, 20);

        // Attempt to set weight on non-existing edge
        EXPECT_NO_THROW({
                            bool result = graph.setEdgeWeight(1, 3, 30);
                            EXPECT_FALSE(result);
                        });

        // Attempt to set weight on unweighted graph
        DirectedUnweightedGraph<int, int> unweightedGraph;
        unweightedGraph.addVertex(1);
        unweightedGraph.addVertex(2);
        unweightedGraph.addEdge(1, 2);

        EXPECT_THROW({
                         unweightedGraph.setEdgeWeight(1, 2, 5);
                     }, std::invalid_argument);
    }

    // Weighted and Undirected Graph
    {
        UndirectedWeightedGraph<int, int> graph;
        graph.addVertex(1);
        graph.addVertex(2);
        graph.addEdge(1, 2, 15);

        // Set existing edge weight
        EXPECT_NO_THROW({
                            bool result = graph.setEdgeWeight(1, 2, 25);
                            EXPECT_TRUE(result);
                        });

        // Verify weight is updated in both directions
        const auto& adjacencyList = graph.getAdjacencyList();
        auto it1 = adjacencyList.find(1);
        auto it2 = adjacencyList.find(2);
        ASSERT_NE(it1, adjacencyList.end());
        ASSERT_NE(it2, adjacencyList.end());
        ASSERT_EQ(it1->second.size(), 1);
        ASSERT_EQ(it2->second.size(), 1);
        EXPECT_EQ(it1->second[0].weight, 25);
        EXPECT_EQ(it2->second[0].weight, 25);

        // Attempt to set weight on non-existing edge
        EXPECT_NO_THROW({
                            bool result = graph.setEdgeWeight(1, 3, 35);
                            EXPECT_FALSE(result);
                        });

        // Attempt to set weight on unweighted graph
        UndirectedUnweightedGraph<int, int> unweightedGraph;
        unweightedGraph.addVertex(1);
        unweightedGraph.addVertex(2);
        unweightedGraph.addEdge(1, 2);

        EXPECT_THROW({
                         unweightedGraph.setEdgeWeight(1, 2, 5);
                     }, std::invalid_argument);
    }

    // Unweighted and Directed Graph
    {
        DirectedUnweightedGraph<int, int> graph;
        graph.addVertex(1);
        graph.addVertex(2);
        graph.addEdge(1, 2);

        // Attempt to set weight on unweighted graph
        EXPECT_THROW({
                         graph.setEdgeWeight(1, 2, 10);
                     }, std::invalid_argument);

        // Attempt to set weight on non-existing vertices
        EXPECT_FALSE(graph.setEdgeWeight(1, 3, 20));
        EXPECT_FALSE(graph.setEdgeWeight(3, 4, 30));
    }

    // Unweighted and Undirected Graph
    {
        UndirectedUnweightedGraph<int, int> graph;
        graph.addVertex(1);
        graph.addVertex(2);
        graph.addEdge(1, 2);

        // Attempt to set weight on unweighted graph
        EXPECT_THROW({
                         graph.setEdgeWeight(1, 2, 10);
                     }, std::invalid_argument);

        // Attempt to set weight on non-existing vertices
        EXPECT_FALSE(graph.setEdgeWeight(1, 3, 20));
        EXPECT_FALSE(graph.setEdgeWeight(3, 4, 30));
    }
}

// Test for getEdgeWeight
TEST(GraphTest, GetEdgeWeight_FullyCoversFunction) {
    // Weighted and Directed Graph
    {
        DirectedWeightedGraph<int, int> graph;
        graph.addVertex(1);
        graph.addVertex(2);
        graph.addEdge(1, 2, 10);

        // Get existing edge weight
        EXPECT_NO_THROW({
                            int weight = graph.getEdgeWeight(1, 2);
                            EXPECT_EQ(weight, 10);
                        });

        // Attempt to get weight of non-existing edge
        EXPECT_THROW({
                         graph.getEdgeWeight(1, 3);
                     }, std::invalid_argument);

        // Attempt to get weight on unweighted graph
        DirectedUnweightedGraph<int, int> unweightedGraph;
        unweightedGraph.addVertex(1);
        unweightedGraph.addVertex(2);
        unweightedGraph.addEdge(1, 2);

        EXPECT_THROW({
                         unweightedGraph.getEdgeWeight(1, 2);
                     }, std::invalid_argument);

        // Attempt to get weight with non-existing vertices
        EXPECT_THROW({
                         graph.getEdgeWeight(3, 4);
                     }, std::invalid_argument);
    }

    // Weighted and Undirected Graph
    {
        UndirectedWeightedGraph<int, int> graph;
        graph.addVertex(1);
        graph.addVertex(2);
        graph.addEdge(1, 2, 15);

        // Get existing edge weight
        EXPECT_NO_THROW({
                            int weight = graph.getEdgeWeight(1, 2);
                            EXPECT_EQ(weight, 15);
                        });

        // Get weight from the opposite direction
        EXPECT_NO_THROW({
                            int weight = graph.getEdgeWeight(2, 1);
                            EXPECT_EQ(weight, 15);
                        });

        // Attempt to get weight of non-existing edge
        EXPECT_THROW({
                         graph.getEdgeWeight(1, 3);
                     }, std::invalid_argument);

        // Attempt to get weight on unweighted graph
        UndirectedUnweightedGraph<int, int> unweightedGraph;
        unweightedGraph.addVertex(1);
        unweightedGraph.addVertex(2);
        unweightedGraph.addEdge(1, 2);

        EXPECT_THROW({
                         unweightedGraph.getEdgeWeight(1, 2);
                     }, std::invalid_argument);

        // Attempt to get weight with non-existing vertices
        EXPECT_THROW({
                         graph.getEdgeWeight(3, 4);
                     }, std::invalid_argument);
    }

    // Unweighted and Directed Graph
    {
        DirectedUnweightedGraph<int, int> graph;
        graph.addVertex(1);
        graph.addVertex(2);
        graph.addEdge(1, 2);

        // Attempt to get weight on unweighted graph
        EXPECT_THROW({
                         graph.getEdgeWeight(1, 2);
                     }, std::invalid_argument);

        // Attempt to get weight with non-existing edge
        EXPECT_THROW({
                         graph.getEdgeWeight(1, 3);
                     }, std::invalid_argument);

        // Attempt to get weight with non-existing vertices
        EXPECT_THROW({
                         graph.getEdgeWeight(3, 4);
                     }, std::invalid_argument);
    }

    // Unweighted and Undirected Graph
    {
        UndirectedUnweightedGraph<int, int> graph;
        graph.addVertex(1);
        graph.addVertex(2);
        graph.addEdge(1, 2);

        // Attempt to get weight on unweighted graph
        EXPECT_THROW({
                         graph.getEdgeWeight(1, 2);
                     }, std::invalid_argument);

        // Attempt to get weight with non-existing edge
        EXPECT_THROW({
                         graph.getEdgeWeight(1, 3);
                     }, std::invalid_argument);

        // Attempt to get weight with non-existing vertices
        EXPECT_THROW({
                         graph.getEdgeWeight(3, 4);
                     }, std::invalid_argument);
    }
}


// Test for removeVertex
TEST(GraphTest, RemoveVertex_FullyCoversFunction) {
    // Directed and Weighted Graph
    {
        DirectedWeightedGraph<int, int> graph;
        graph.addVertex(1);
        graph.addVertex(2);
        graph.addVertex(3);
        graph.addEdge(1, 2, 10);
        graph.addEdge(1, 3, 20);
        graph.addEdge(2, 3, 30);

        // Remove existing vertex
        EXPECT_TRUE(graph.removeVertex(2));

        // Verify vertex is removed
        const auto& vertices = graph.getVertices();
        EXPECT_EQ(vertices.find(2), vertices.end());

        // Verify outgoing edges from removed vertex are gone
        const auto& adjacencyList = graph.getAdjacencyList();
        EXPECT_EQ(adjacencyList.find(2), adjacencyList.end());

        // Verify incoming edges to removed vertex are gone
        auto it1 = adjacencyList.find(1);
        ASSERT_NE(it1, adjacencyList.end());
        EXPECT_EQ(it1->second.size(), 1);
        EXPECT_EQ(it1->second[0].to, 3);

        // Verify other edges remain
        auto it3 = adjacencyList.find(3);
        if (it3 != adjacencyList.end()) {
            EXPECT_TRUE(it3->second.empty());
        } else {
            SUCCEED() << "Vertex 3 has no outgoing edges, which is acceptable.";
        }

        // Attempt to remove non-existing vertex
        EXPECT_FALSE(graph.removeVertex(4));
    }

    // Undirected and Weighted Graph
    {
        UndirectedWeightedGraph<int, int> graph;
        graph.addVertex(1);
        graph.addVertex(2);
        graph.addVertex(3);
        graph.addEdge(1, 2, 15);
        graph.addEdge(1, 3, 25);
        graph.addEdge(2, 3, 35);

        // Remove existing vertex
        EXPECT_TRUE(graph.removeVertex(2));

        // Verify vertex is removed
        const auto& vertices = graph.getVertices();
        EXPECT_EQ(vertices.find(2), vertices.end());

        // Verify outgoing edges from removed vertex are gone
        const auto& adjacencyList = graph.getAdjacencyList();
        EXPECT_EQ(adjacencyList.find(2), adjacencyList.end());

        // Verify incoming edges to removed vertex are gone
        auto it1 = adjacencyList.find(1);
        ASSERT_NE(it1, adjacencyList.end());
        EXPECT_EQ(it1->second.size(), 1);
        EXPECT_EQ(it1->second[0].to, 3);

        auto it3 = adjacencyList.find(3);
        ASSERT_NE(it3, adjacencyList.end());
        EXPECT_EQ(it3->second.size(), 1);
        EXPECT_EQ(it3->second[0].to, 1);

        // Attempt to remove non-existing vertex
        EXPECT_FALSE(graph.removeVertex(4));
    }

    // Directed and Unweighted Graph
    {
        DirectedUnweightedGraph<int, int> graph;
        graph.addVertex(1);
        graph.addVertex(2);
        graph.addVertex(3);
        graph.addEdge(1, 2);
        graph.addEdge(1, 3);
        graph.addEdge(2, 3);

        // Remove existing vertex
        EXPECT_TRUE(graph.removeVertex(2));

        // Verify vertex is removed
        const auto& vertices = graph.getVertices();
        EXPECT_EQ(vertices.find(2), vertices.end());

        // Verify outgoing edges from removed vertex are gone
        const auto& adjacencyList = graph.getAdjacencyList();
        EXPECT_EQ(adjacencyList.find(2), adjacencyList.end());

        // Verify incoming edges to removed vertex are gone
        auto it1 = adjacencyList.find(1);
        ASSERT_NE(it1, adjacencyList.end());
        EXPECT_EQ(it1->second.size(), 1);
        EXPECT_EQ(it1->second[0].to, 3);

        auto it3 = adjacencyList.find(3);
        if (it3 != adjacencyList.end()) {
            EXPECT_TRUE(it3->second.empty());
        } else {
            SUCCEED() << "Vertex 3 has no outgoing edges, which is acceptable.";
        }

        // Attempt to remove non-existing vertex
        EXPECT_FALSE(graph.removeVertex(4));
    }

    // Undirected and Unweighted Graph
    {
        UndirectedUnweightedGraph<int, int> graph;
        graph.addVertex(1);
        graph.addVertex(2);
        graph.addVertex(3);
        graph.addEdge(1, 2);
        graph.addEdge(1, 3);
        graph.addEdge(2, 3);

        // Remove existing vertex
        EXPECT_TRUE(graph.removeVertex(2));

        // Verify vertex is removed
        const auto& vertices = graph.getVertices();
        EXPECT_EQ(vertices.find(2), vertices.end());

        // Verify outgoing edges from removed vertex are gone
        const auto& adjacencyList = graph.getAdjacencyList();
        EXPECT_EQ(adjacencyList.find(2), adjacencyList.end());

        // Verify incoming edges to removed vertex are gone
        auto it1 = adjacencyList.find(1);
        ASSERT_NE(it1, adjacencyList.end());
        EXPECT_EQ(it1->second.size(), 1);
        EXPECT_EQ(it1->second[0].to, 3);

        auto it3 = adjacencyList.find(3);
        ASSERT_NE(it3, adjacencyList.end());
        EXPECT_EQ(it3->second.size(), 1);
        EXPECT_EQ(it3->second[0].to, 1);

        // Attempt to remove non-existing vertex
        EXPECT_FALSE(graph.removeVertex(4));
    }

    // Attempt to remove vertex from empty graph
    {
        using AnyGraph = Graph<int>;
        AnyGraph graph;

        EXPECT_FALSE(graph.removeVertex(1));
    }
}

// Test for graph printing
TEST(GraphTest, PrintGraph) {
    UndirectedUnweightedGraph<unsigned int, double> graph;
    graph.addVertex(1);
    graph.addVertex(2);
    graph.addVertex(3);
    graph.addVertex(4);
    graph.addVertex(5);
    graph.addVertex(6);
    graph.addVertex(7);
    graph.addVertex(8);
    graph.addVertex(9);
    graph.addVertex(10);
    graph.addVertex(11);
    graph.addVertex(12);
    graph.addVertex(13);
    graph.addVertex(14);
}

/*

// Test for removeEdge
TEST(GraphTest, RemoveEdge_FullyCoversFunction) {
    // Directed and Weighted Graph
    {
        DirectedWeightedGraph<int, int> graph;
        graph.addVertex(1);
        graph.addVertex(2);
        graph.addVertex(3);
        graph.addEdge(1, 2, 10);
        graph.addEdge(1, 3, 20);
        graph.addEdge(2, 3, 30);

        // Remove existing edge
        EXPECT_TRUE(graph.removeEdge(1, 2));

        // Verify edge is removed
        const auto& adjacencyList = graph.getAdjacencyList();
        auto it = adjacencyList.find(1);
        ASSERT_NE(it, adjacencyList.end());
        bool edgeFound = false;
        for (const auto& edge : it->second) {
            if (edge.to == 2) {
                edgeFound = true;
                break;
            }
        }
        EXPECT_FALSE(edgeFound);

        // Attempt to remove the same edge again
        EXPECT_FALSE(graph.removeEdge(1, 2));

        // Attempt to remove non-existing edge
        EXPECT_FALSE(graph.removeEdge(2, 1));

        // Remove another existing edge
        EXPECT_TRUE(graph.removeEdge(2, 3));

        // Verify edge is removed
        auto it2 = adjacencyList.find(2);
        ASSERT_NE(it2, adjacencyList.end());
        edgeFound = false;
        for (const auto& edge : it2->second) {
            if (edge.to == 3) {
                edgeFound = true;
                break;
            }
        }
        EXPECT_FALSE(edgeFound);

        // Attempt to remove edge with non-existing vertices
        EXPECT_FALSE(graph.removeEdge(4, 5));
    }

    // Undirected and Weighted Graph
    {
        UndirectedWeightedGraph<int, int> graph;
        graph.addVertex(1);
        graph.addVertex(2);
        graph.addVertex(3);
        graph.addEdge(1, 2, 15);
        graph.addEdge(1, 3, 25);
        graph.addEdge(2, 3, 35);

        // Remove existing edge
        EXPECT_TRUE(graph.removeEdge(1, 2));

        // Verify edge is removed from both directions
        const auto& adjacencyList = graph.getAdjacencyList();

        // Check from 1 to 2
        auto it1 = adjacencyList.find(1);
        ASSERT_NE(it1, adjacencyList.end());
        bool edgeFound = false;
        for (const auto& edge : it1->second) {
            if (edge.to == 2) {
                edgeFound = true;
                break;
            }
        }
        EXPECT_FALSE(edgeFound);

        // Check from 2 to 1
        auto it2 = adjacencyList.find(2);
        ASSERT_NE(it2, adjacencyList.end());
        edgeFound = false;
        for (const auto& edge : it2->second) {
            if (edge.to == 1) {
                edgeFound = true;
                break;
            }
        }
        EXPECT_FALSE(edgeFound);

        // Attempt to remove the same edge again
        EXPECT_FALSE(graph.removeEdge(1, 2));

        // Attempt to remove non-existing edge
        EXPECT_FALSE(graph.removeEdge(2, 1));

        // Remove another existing edge
        EXPECT_TRUE(graph.removeEdge(2, 3));

        // Verify edge is removed from both directions
        auto it3 = adjacencyList.find(2);
        ASSERT_NE(it3, adjacencyList.end());
        edgeFound = false;
        for (const auto& edge : it3->second) {
            if (edge.to == 3) {
                edgeFound = true;
                break;
            }
        }
        EXPECT_FALSE(edgeFound);

        auto it4 = adjacencyList.find(3);
        ASSERT_NE(it4, adjacencyList.end());
        edgeFound = false;
        for (const auto& edge : it4->second) {
            if (edge.to == 2) {
                edgeFound = true;
                break;
            }
        }
        EXPECT_FALSE(edgeFound);

        // Attempt to remove edge with non-existing vertices
        EXPECT_FALSE(graph.removeEdge(4, 5));
    }

    // Directed and Unweighted Graph
    {
        DirectedUnweightedGraph<int, int> graph;
        graph.addVertex(1);
        graph.addVertex(2);
        graph.addVertex(3);
        graph.addEdge(1, 2);
        graph.addEdge(1, 3);
        graph.addEdge(2, 3);

        // Remove existing edge
        EXPECT_TRUE(graph.removeEdge(1, 2));

        // Verify edge is removed
        const auto& adjacencyList = graph.getAdjacencyList();
        auto it = adjacencyList.find(1);
        ASSERT_NE(it, adjacencyList.end());
        bool edgeFound = false;
        for (const auto& edge : it->second) {
            if (edge.to == 2) {
                edgeFound = true;
                break;
            }
        }
        EXPECT_FALSE(edgeFound);

        // Attempt to remove the same edge again
        EXPECT_FALSE(graph.removeEdge(1, 2));

        // Attempt to remove non-existing edge
        EXPECT_FALSE(graph.removeEdge(2, 1));

        // Remove another existing edge
        EXPECT_TRUE(graph.removeEdge(2, 3));

        // Verify edge is removed
        auto it2 = adjacencyList.find(2);
        ASSERT_NE(it2, adjacencyList.end());
        edgeFound = false;
        for (const auto& edge : it2->second) {
            if (edge.to == 3) {
                edgeFound = true;
                break;
            }
        }
        EXPECT_FALSE(edgeFound);

        // Attempt to remove edge with non-existing vertices
        EXPECT_FALSE(graph.removeEdge(4, 5));
    }

    // Undirected and Unweighted Graph
    {
        UndirectedUnweightedGraph<int, int> graph;
        graph.addVertex(1);
        graph.addVertex(2);
        graph.addVertex(3);
        graph.addEdge(1, 2);
        graph.addEdge(1, 3);
        graph.addEdge(2, 3);

        // Remove existing edge
        EXPECT_TRUE(graph.removeEdge(1, 2));

        // Verify edge is removed from both directions
        const auto& adjacencyList = graph.getAdjacencyList();

        // Check from 1 to 2
        auto it1 = adjacencyList.find(1);
        ASSERT_NE(it1, adjacencyList.end());
        bool edgeFound = false;
        for (const auto& edge : it1->second) {
            if (edge.to == 2) {
                edgeFound = true;
                break;
            }
        }
        EXPECT_FALSE(edgeFound);

        // Check from 2 to 1
        auto it2 = adjacencyList.find(2);
        ASSERT_NE(it2, adjacencyList.end());
        edgeFound = false;
        for (const auto& edge : it2->second) {
            if (edge.to == 1) {
                edgeFound = true;
                break;
            }
        }
        EXPECT_FALSE(edgeFound);

        // Attempt to remove the same edge again
        EXPECT_FALSE(graph.removeEdge(1, 2));

        // Attempt to remove non-existing edge
        EXPECT_FALSE(graph.removeEdge(2, 1));

        // Remove another existing edge
        EXPECT_TRUE(graph.removeEdge(2, 3));

        // Verify edge is removed from both directions
        auto it3 = adjacencyList.find(2);
        ASSERT_NE(it3, adjacencyList.end());
        edgeFound = false;
        for (const auto& edge : it3->second) {
            if (edge.to == 3) {
                edgeFound = true;
                break;
            }
        }
        EXPECT_FALSE(edgeFound);

        auto it4 = adjacencyList.find(3);
        ASSERT_NE(it4, adjacencyList.end());
        edgeFound = false;
        for (const auto& edge : it4->second) {
            if (edge.to == 2) {
                edgeFound = true;
                break;
            }
        }
        EXPECT_FALSE(edgeFound);

        // Attempt to remove edge with non-existing vertices
        EXPECT_FALSE(graph.removeEdge(4, 5));
    }

    // Attempt to remove edge from empty graph
    {
        using AnyGraph = Graph<int>;
        AnyGraph graph;

        EXPECT_FALSE(graph.removeEdge(1, 2));
    }
}*/
