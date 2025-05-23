#include <gtest/gtest.h>
#include <algorithm>
#include "Graph.h"
#include "../headers/graph/InheritanceGraph.h"

/*

Tests implementations:

addVertex               = true
removeVertex            = true
addEdge                 = true
removeEdge              = true
setEdgeWeight           = true
getEdgeWeight           = true
getAllEdges             = false
isDirected              = no need
isWeighted              = no need
hasEdge                 = no need
hasVertex               = no need
getVertices             = no need
getAdjacencyList        = false
getVertexEdges          = false
printGraph(1,2)         = no need
findConnectedComponent  = true
vertexCount             = no need
edgeCount               = true
traverse                = false
isConnected             = false
connectedComponents     = false
isAcyclic               = false
topologicalSort         = false
dijkstra                = false
bellmanFord             = false
floydWarshall           = false
kruskalMST              = false
primMST                 = false
hasEulerianPath         = false
eulerianPath            = false
hamiltonianPath         = false
maxFlow                 = false
bipartiteMatching       = false
transpose               = false
complement              = false
subGraph                = false

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

TEST(GraphTest, RemoveVertex) {
    // Directed and Weighted Graph
    {
      DirectedWeightedGraph<char, int> graph;
      graph.addVertex('A', 'B', 'C', 'D', 'E');
      graph.addEdge('A', 'D', 1);
      graph.addEdge('A', 'C', 2);
      graph.addEdge('A', 'B', 3);
      graph.addEdge('B', 'C', 4);
      graph.addEdge('C', 'E', 5);

      EXPECT_EQ(graph.edgeCount(), 5);
      EXPECT_EQ(graph.vertexCount(), 5);

      graph.removeVertex('C', 'E');

      EXPECT_EQ(graph.edgeCount(), 2);
      EXPECT_EQ(graph.vertexCount(), 3);
    }

    // Undirected and Weighted Graph
    {
      UndirectedWeightedGraph<char, int> graph;
      graph.addVertex('A', 'B', 'C', 'D', 'E');
      graph.addEdge('A', 'D', 1);
      graph.addEdge('A', 'C', 2);
      graph.addEdge('A', 'B', 3);
      graph.addEdge('B', 'C', 4);
      graph.addEdge('C', 'E', 5);

      EXPECT_EQ(graph.edgeCount(), 5);
      EXPECT_EQ(graph.vertexCount(), 5);

      graph.removeVertex('C');

      EXPECT_EQ(graph.edgeCount(), 2);
      EXPECT_EQ(graph.vertexCount(), 4);
    }

    // Directed and Unweighted Graph
    {
      DirectedUnweightedGraph<char> graph;
      graph.addVertex('A', 'B', 'C', 'D', 'E');
      graph.addEdge('A', 'D');
      graph.addEdge('A', 'C');
      graph.addEdge('A', 'B');
      graph.addEdge('B', 'C');
      graph.addEdge('C', 'E');

      EXPECT_EQ(graph.edgeCount(), 5);
      EXPECT_EQ(graph.vertexCount(), 5);

      graph.removeVertex('C');

      EXPECT_EQ(graph.edgeCount(), 2);
      EXPECT_EQ(graph.vertexCount(), 4);

    }

    // Undirected and Unweighted Graph
    {
      UndirectedUnweightedGraph<char> graph;
      graph.addVertex('A', 'B', 'C', 'D', 'E');
      graph.addEdge('A', 'D');
      graph.addEdge('A', 'C');
      graph.addEdge('A', 'B');
      graph.addEdge('B', 'C');
      graph.addEdge('C', 'E');

      //graph.printGraph(std::cout);

      EXPECT_EQ(graph.edgeCount(), 5);
      EXPECT_EQ(graph.vertexCount(), 5);

      graph.removeVertex('C');

      EXPECT_EQ(graph.edgeCount(), 2);
      EXPECT_EQ(graph.vertexCount(), 4);
    }
}

TEST(GraphTest, AddEdge) {
    // Directed Weighted graph
    {
      DirectedWeightedGraph<int, double> graph;

      graph.addVertex(1);
      graph.addVertex(2);

      EXPECT_NO_THROW({
        bool result = graph.addEdge(1, 2, 5.0);
        EXPECT_TRUE(result);
      });

      EXPECT_TRUE(graph.hasEdge(1, 2));

      const auto& adjacencyList = graph.getAdjacencyList();
      auto it = adjacencyList.find(1);
      ASSERT_NE(it, adjacencyList.end());
      ASSERT_EQ(it->second.size(), 1);
      EXPECT_EQ(it->second[0].to, 2);
      EXPECT_EQ(it->second[0].weight, 5.0);
    }

    // Directed Unweighted graph
    {
      DirectedUnweightedGraph<int> graph;

      graph.addVertex(1);
      graph.addVertex(2);

      // Add edge with default weight
      EXPECT_NO_THROW({
        bool result = graph.addEdge(1, 2);
        EXPECT_TRUE(result);
      });

      EXPECT_TRUE(graph.hasEdge(1, 2));

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

    // Undirected Weighted graph
    {
      UndirectedWeightedGraph<int, double> graph;

      graph.addVertex(1);
      graph.addVertex(2);

      // Add edge with valid weight
      EXPECT_NO_THROW({
        bool result = graph.addEdge(1, 2, 3.5);
        EXPECT_TRUE(result);
      });

      EXPECT_TRUE(graph.hasEdge(1, 2));
      EXPECT_EQ(graph.getEdgeWeight(1, 2), 3.5);

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
    }

    // Undirected Unweighted graph
    {
      UndirectedUnweightedGraph<int> graph;

      graph.addVertex(1);
      graph.addVertex(2);

      // Add edge with default weight
      EXPECT_NO_THROW({
        bool result = graph.addEdge(1, 2);
        EXPECT_TRUE(result);
      });

      EXPECT_TRUE(graph.hasEdge(1, 2));

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
}

TEST(GraphTest, RemoveEdge) {
    // Directed and Weighted Graph
    {
      DirectedWeightedGraph<char, int> graph;
      graph.addVertex('A', 'B', 'C', 'D', 'E');
      graph.addEdge('A', 'D', 1);
      graph.addEdge('A', 'C', 2);
      graph.addEdge('A', 'B', 3);
      graph.addEdge('B', 'C', 4);
      graph.addEdge('C', 'E', 5);

      EXPECT_EQ(graph.edgeCount(), 5);
      EXPECT_EQ(graph.vertexCount(), 5);

      graph.removeEdge('A', 'C');

      EXPECT_EQ(graph.edgeCount(), 4);
      EXPECT_EQ(graph.vertexCount(), 5);
    }

    // Undirected and Weighted Graph
    {
      UndirectedWeightedGraph<char, int> graph;
      graph.addVertex('A', 'B', 'C', 'D', 'E');
      graph.addEdge('A', 'D', 1);
      graph.addEdge('A', 'C', 2);
      graph.addEdge('A', 'B', 3);
      graph.addEdge('B', 'C', 4);
      graph.addEdge('C', 'E', 5);

      EXPECT_EQ(graph.edgeCount(), 5);
      EXPECT_EQ(graph.vertexCount(), 5);

      graph.removeEdge('A', 'C');

      EXPECT_EQ(graph.edgeCount(), 4);
      EXPECT_EQ(graph.vertexCount(), 5);
    }

    // Directed and Unweighted Graph
    {
      DirectedUnweightedGraph<char> graph;
      graph.addVertex('A', 'B', 'C', 'D', 'E');
      graph.addEdge('A', 'D');
      graph.addEdge('A', 'C');
      graph.addEdge('A', 'B');
      graph.addEdge('B', 'C');
      graph.addEdge('C', 'E');

      EXPECT_EQ(graph.edgeCount(), 5);
      EXPECT_EQ(graph.vertexCount(), 5);

      graph.removeEdge('A', 'C');

      EXPECT_EQ(graph.edgeCount(), 4);
      EXPECT_EQ(graph.vertexCount(), 5);

    }

    // Undirected and Unweighted Graph
    {
      UndirectedUnweightedGraph<char> graph;
      graph.addVertex('A', 'B', 'C', 'D', 'E');
      graph.addEdge('A', 'D');
      graph.addEdge('A', 'C');
      graph.addEdge('A', 'B');
      graph.addEdge('B', 'C');
      graph.addEdge('C', 'E');

      //graph.printGraph(std::cout);

      EXPECT_EQ(graph.edgeCount(), 5);
      EXPECT_EQ(graph.vertexCount(), 5);

      graph.removeEdge('A', 'C');

      EXPECT_EQ(graph.edgeCount(), 4);
      EXPECT_EQ(graph.vertexCount(), 5);
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

TEST(GraphTest, EdgeCount) {
    UndirectedUnweightedGraph<int> graph;
    graph.addVertex(1, 2, 3, 4, 5);

    graph.addEdge(1, 2);
    graph.addEdge(2, 3);
    graph.addEdge(3, 4);
    graph.addEdge(4, 5);
    graph.addEdge(5, 1);
    graph.addEdge(1, 4);

    EXPECT_FALSE(graph.addEdge(1, 4));

    EXPECT_EQ(graph.edgeCount(), 6);
}

TEST(GraphTest, GetEdgeWeight) {
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
      DirectedUnweightedGraph<int> unweightedGraph;
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
      UndirectedUnweightedGraph<int> unweightedGraph;
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
      DirectedUnweightedGraph<int> graph;
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
      UndirectedUnweightedGraph<int> graph;
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

// TODO
TEST(GraphTest, getAllEdges) {}

TEST(GraphTest, SetEdgeWeight) {
    // Directed and Weighted Graph
    {
      DirectedWeightedGraph<char, int> graph;
      graph.addVertex('A', 'B', 'C', 'D', 'E');
      graph.addEdge('A', 'D', 1);
      graph.addEdge('A', 'C', 2);
      graph.addEdge('A', 'B', 3);
      graph.addEdge('B', 'C', 4);
      graph.addEdge('C', 'E', 5);

      EXPECT_EQ(graph.getEdgeWeight('A', 'D'), 1);
      EXPECT_EQ(graph.getEdgeWeight('A', 'C'), 2);
      EXPECT_EQ(graph.getEdgeWeight('A', 'B'), 3);
      EXPECT_EQ(graph.getEdgeWeight('B', 'C'), 4);
      EXPECT_EQ(graph.getEdgeWeight('C', 'E'), 5);

      graph.setEdgeWeight('A', 'D', 777);

      EXPECT_EQ(graph.getEdgeWeight('A', 'D'), 777);

      EXPECT_EQ(graph.edgeCount(), 5);
      EXPECT_EQ(graph.vertexCount(), 5);
    }

    // Undirected and Weighted Graph
    {
      UndirectedWeightedGraph<char, int> graph;
      graph.addVertex('A', 'B', 'C', 'D', 'E');
      graph.addEdge('A', 'D', 1);
      graph.addEdge('A', 'C', 2);
      graph.addEdge('A', 'B', 3);
      graph.addEdge('B', 'C', 4);
      graph.addEdge('C', 'E', 5);

      EXPECT_EQ(graph.getEdgeWeight('A', 'D'), 1);
      EXPECT_EQ(graph.getEdgeWeight('A', 'C'), 2);
      EXPECT_EQ(graph.getEdgeWeight('A', 'B'), 3);
      EXPECT_EQ(graph.getEdgeWeight('B', 'C'), 4);
      EXPECT_EQ(graph.getEdgeWeight('C', 'E'), 5);

      graph.setEdgeWeight('A', 'D', 777);

      EXPECT_EQ(graph.getEdgeWeight('A', 'D'), 777);

      EXPECT_EQ(graph.edgeCount(), 5);
      EXPECT_EQ(graph.vertexCount(), 5);
    }

    // Directed and Unweighted Graph
    {
      DirectedUnweightedGraph<char> graph;
      graph.addVertex('A', 'B', 'C', 'D', 'E');
      graph.addEdge('A', 'D');
      graph.addEdge('A', 'C');
      graph.addEdge('A', 'B');
      graph.addEdge('B', 'C');
      graph.addEdge('C', 'E');

      EXPECT_THROW(graph.setEdgeWeight('A', 'D', 777), std::invalid_argument);

      EXPECT_EQ(graph.edgeCount(), 5);
      EXPECT_EQ(graph.vertexCount(), 5);

    }

    // Undirected and Unweighted Graph
    {
      UndirectedUnweightedGraph<char> graph;
      graph.addVertex('A', 'B', 'C', 'D', 'E');
      graph.addEdge('A', 'D');
      graph.addEdge('A', 'C');
      graph.addEdge('A', 'B');
      graph.addEdge('B', 'C');
      graph.addEdge('C', 'E');

      EXPECT_THROW(graph.setEdgeWeight('A', 'D', 777), std::invalid_argument);

      EXPECT_EQ(graph.edgeCount(), 5);
      EXPECT_EQ(graph.vertexCount(), 5);
    }
}


/*

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
*/
