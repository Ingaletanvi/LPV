#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>
using namespace std;

void bfs(int start, const vector<vector<int>>& adj, vector<bool>& visited) {
    queue<int> q;
    visited[start] = true;
    q.push(start);

    while (!q.empty()) {
        int node = q.front();
        q.pop();
        cout << "Visited (BFS) Node: " << node << endl;

        vector<int> neighbors; 

        #pragma omp parallel for
        for (int i = 0; i < adj[node].size(); ++i) {
            int neighbor = adj[node][i];
            int thread_id = omp_get_thread_num(); 

            if (!visited[neighbor]) {
                #pragma omp critical
                {
                    if (!visited[neighbor]) {  
                        visited[neighbor] = true;
                        neighbors.push_back(neighbor);
                        cout << "Thread " << thread_id << " processing neighbor (BFS): " << neighbor << endl;
                    }
                }
            }
        }

        for (int neighbor : neighbors) {
            q.push(neighbor);
        }
    }
}

void dfs(int node, const vector<vector<int>>& adj, vector<bool>& visited) {
    int thread_id = omp_get_thread_num();

    #pragma omp critical
    {
        visited[node] = true;
        cout << "Thread " << thread_id << " visited (DFS) Node: " << node << endl;
    }

    for (int i = 0; i < adj[node].size(); ++i) {
        int neighbor = adj[node][i];
        
        bool need_visit = false;
        
        #pragma omp critical
        {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                need_visit = true;
                cout << "Thread " << thread_id << " preparing DFS for neighbor: " << neighbor << endl;
            }
        }

        if (need_visit) {
            #pragma omp task
            {
                dfs(neighbor, adj, visited);
            }
        }
    }

    #pragma omp taskwait
}

int main() {
    // omp_set_num_threads(8);

    int n, m; 
    cout << "Enter number of nodes and edges: ";
    cin >> n >> m;

    vector<vector<int>> adj(n);

    cout << "Enter edges (u v):" << endl;
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    vector<bool> visited(n, false);

    cout << "\nBFS starting from node 0:\n";
    double start_bfs = omp_get_wtime(); 
    bfs(0, adj, visited);
    double end_bfs = omp_get_wtime();
    cout << "Time taken for BFS: " << (end_bfs - start_bfs) << " seconds\n";

    visited.assign(n, false);

    cout << "\nDFS starting from node 0:\n";
    double start_dfs = omp_get_wtime(); 
    #pragma omp parallel
    {
        #pragma omp single
        dfs(0, adj, visited);
    }    
    double end_dfs = omp_get_wtime();
    cout << "Time taken for DFS: " << (end_dfs - start_dfs) << " seconds\n";

    return 0;
}
