/*
 * graphUtils.h
 *
 *  Created on: Jun 4, 2015
 *      Author: vvminh
 *
 ******************************************************************************
 * http://www.geeksforgeeks.org/transitive-closure-of-a-graph/
 * // Program for transitive closure using Floyd Warshall Algorithm
 *
 * http://www.geeksforgeeks.org/tarjan-algorithm-find-strongly-connected-components/
 * // A C++ program to find strongly connected components in a given
 * // directed graph using Tarjan's algorithm (single DFS)
 *
 * http://www.geeksforgeeks.org/strongly-connected-components/
 */

#ifndef UTILS_GRAPHUTILS_H_
#define UTILS_GRAPHUTILS_H_

#include <iostream>
#include <list>
#include <stack>
#include <vector>
#include <utility>      // std::pair, std::make_pair

#define NIL -1
using namespace std;

// A class that represents an directed graph
class Graph {
	int V;				// No. of vertices
	list<int> *adj;		// A dynamic array of adjacency lists

	int** graph;		// matrix occurrence

	/* reach[][] will be the output matrix that will finally have the shortest
	 distances between every pair of vertices */
	int** reach;		// matrix contains transitive closure

	vector<vector<int> > components;

	// A recursive function that finds and prints strongly connected
	// components using DFS traversal
	// u --> The vertex to be visited next
	// disc[] --> Stores discovery times of visited vertices
	// low[] -- >> earliest visited vertex (the vertex with minimum
	//             discovery time) that can be reached from subtree
	//             rooted with current vertex
	// *st -- >> To store all the connected ancestors (could be part
	//           of SCC)
	// stackMember[] --> bit/index array for faster check whether
	//                  a node is in stack
	void SCCUtil(int u, int disc[], int low[], stack<int> *st, bool stackMember[]) {
		// A static variable is used for simplicity, we can avoid use
		// of static variable by passing a pointer.
		static int time = 0;

		// Initialize discovery time and low value
		disc[u] = low[u] = ++time;
		st->push(u);
		stackMember[u] = true;

		// Go through all vertices adjacent to this
		list<int>::iterator i;
		for (i = adj[u].begin(); i != adj[u].end(); ++i) {
			int v = *i;  // v is current adjacent of 'u'

			// If v is not visited yet, then recur for it
			if (disc[v] == -1) {
				SCCUtil(v, disc, low, st, stackMember);

				// Check if the subtree rooted with 'v' has a
				// connection to one of the ancestors of 'u'
				// Case 1 (per above discussion on Disc and Low value)
				low[u] = min(low[u], low[v]);
			}

			// Update low value of 'u' only of 'v' is still in stack
			// (i.e. it's a back edge, not cross edge).
			// Case 2 (per above discussion on Disc and Low value)
			else if (stackMember[v] == true)
				low[u] = min(low[u], disc[v]);
		}

		// head node found, pop the stack and print an SCC
		int w = 0;  // To store stack extracted vertices
		vector<int> vTemp;
		if (low[u] == disc[u]) {
			while (st->top() != u) {
				w = (int) st->top();
				//cout << w << " ";
				vTemp.push_back(w);
				stackMember[w] = false;
				st->pop();
			}
			w = (int) st->top();
			//cout << w << "\n";
			vTemp.push_back(w);
			if (vTemp.size() > 1) {
				components.push_back(vTemp);
			}

			stackMember[w] = false;
			st->pop();
		}
	}

public:
	Graph(int V) {
		this->V = V;
		adj = new list<int> [V];

		graph = new int*[V];
		reach = new int*[V];

		for (int i = 0; i < V; i++) {
			graph[i] = new int[V];
			reach[i] = new int[V];

			for (int j = 0; j < V; j++) {
				graph[i][j] = 0;
				reach[i][j] = 0;
			}
			graph[i][i] = 1;
		}
	}

	void addEdge(int v, int w) {
		graph[v][w] = 1;
		graph[w][v] = 1;
	}

	void buildAdjacencyLists() {
		for (int i = 0; i < V; i++) {
			for (int j = 0; j < V; j++) {
				if (1 == reach[i][j] && i != j) {
					adj[i].push_back(j);
				}
			}
		}
	}

	void buildTransitiveClosure() {
		/* reach[][] will be the output matrix that will finally have the shortest
		 distances between every pair of vertices */

		/* Initialize the solution matrix same as input graph matrix. Or
		 we can say the initial values of shortest distances are based
		 on shortest paths considering no intermediate vertex. */
		for (int i = 0; i < V; i++)
			for (int j = 0; j < V; j++)
				reach[i][j] = graph[i][j];

		/* Add all vertices one by one to the set of intermediate vertices.
		 ---> Before start of a iteration, we have reachability values for all
		 pairs of vertices such that the reachability values consider only the
		 vertices in set {0, 1, 2, .. k-1} as intermediate vertices.
		 ----> After the end of a iteration, vertex no. k is added to the set of
		 intermediate vertices and the set becomes {0, 1, 2, .. k} */
		for (int k = 0; k < V; k++) {
			// Pick all vertices as source one by one
			for (int i = 0; i < V; i++) {
				// Pick all vertices as destination for the
				// above picked source
				for (int j = 0; j < V; j++) {
					// If vertex k is on a path from i to j,
					// then make sure that the value of reach[i][j] is 1
					reach[i][j] = reach[i][j] || (reach[i][k] && reach[k][j]);
				}
			}
		}
	}

	vector<pair<int, int> > getLinksInTransitiveClosure() {
		vector<pair<int, int> > mustLinks;

		for (int i = 0; i < V; i++) {
			for (int j = i; j < V; j++) {
				//std::cout << reach[i][j] << " ";
				if (1 == reach[i][j] && i != j) {
					//std::cout << i << "->" << j << std::endl;
					mustLinks.push_back(make_pair(i, j));
				}
			}
		}
		return mustLinks;
	}

	void SCC() {
		int *disc = new int[V];
		int *low = new int[V];
		bool *stackMember = new bool[V];
		stack<int> *st = new stack<int>();

		// Initialize disc and low, and stackMember arrays
		for (int i = 0; i < V; i++) {
			disc[i] = NIL;
			low[i] = NIL;
			stackMember[i] = false;
		}

		// Call the recursive helper function to find strongly
		// connected components in DFS tree with vertex 'i'
		for (int i = 0; i < V; i++)
			if (disc[i] == NIL)
				SCCUtil(i, disc, low, st, stackMember);
	}

	void printConnectecComponents() {
		cout << "Strongly Connected Components: " << endl;
		int nComps = components.size();
		for (int i = 0; i < nComps; ++i) {
			int nElems = components[i].size();
			for (int j = 0; j < nElems; ++j) {
				cout << components[i][j] << "\t";
			}
			cout << endl;
		}
	}

	vector<vector<int> > getConnectedComponents() { return components; }

	void dump() {
		for (int i = 0; i < V; i++) {
			for (int j = 0; j < V; j++) {
				std::cout << graph[i][j] << " ";
			}
			std::cout << std::endl;
		}
	}
};

#endif /* UTILS_GRAPHUTILS_H_ */
