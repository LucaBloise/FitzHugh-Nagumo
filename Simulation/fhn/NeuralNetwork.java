package fhn;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/** Sparse undirected connectivity: A_ij = 1, i != j. */
public class NeuralNetwork {
    private final int n;
    private final List<int[]> neighbors;
    private final boolean fullyConnected;

    public NeuralNetwork(int n, List<int[]> neighbors, boolean fullyConnected) {
        this.n = n;
        this.neighbors = neighbors;
        this.fullyConnected = fullyConnected;
    }

    public boolean isFullyConnected() {
        return fullyConnected;
    }

    public int size() {
        return n;
    }

    public int[] neighborsOf(int i) {
        return neighbors.get(i);
    }

    public static NeuralNetwork build(SimulationConfig cfg, Random rng) {
        return switch (cfg.network) {
            case FULL -> fullyConnected(cfg.n);
            case RANDOM -> randomErdosRenyi(cfg.n, cfg.connectionProbability, rng);
            case RING -> ring(cfg.n, cfg.ringHalfWidth);
        };
    }

    public static NeuralNetwork fullyConnected(int n) {
        List<int[]> adj = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            int[] nb = new int[n - 1];
            int k = 0;
            for (int j = 0; j < n; j++) {
                if (j != i) {
                    nb[k++] = j;
                }
            }
            adj.add(nb);
        }
        return new NeuralNetwork(n, adj, true);
    }

    public static NeuralNetwork randomErdosRenyi(int n, double p, Random rng) {
        List<List<Integer>> adj = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            adj.add(new ArrayList<>());
        }
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (rng.nextDouble() < p) {
                    adj.get(i).add(j);
                    adj.get(j).add(i);
                }
            }
        }
        List<int[]> frozen = new ArrayList<>(n);
        for (List<Integer> list : adj) {
            int[] arr = new int[list.size()];
            for (int t = 0; t < list.size(); t++) {
                arr[t] = list.get(t);
            }
            frozen.add(arr);
        }
        return new NeuralNetwork(n, frozen, false);
    }

    public static NeuralNetwork ring(int n, int halfWidth) {
        List<int[]> adj = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            List<Integer> list = new ArrayList<>(2 * halfWidth);
            for (int d = 1; d <= halfWidth; d++) {
                list.add((i - d + n) % n);
                list.add((i + d) % n);
            }
            int[] arr = new int[list.size()];
            for (int t = 0; t < list.size(); t++) {
                arr[t] = list.get(t);
            }
            adj.add(arr);
        }
        return new NeuralNetwork(n, adj, false);
    }

    /** Edge list for visualization: pairs (i, j) with i < j. */
    public List<int[]> edges() {
        List<int[]> edges = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            for (int j : neighbors.get(i)) {
                if (i < j) {
                    edges.add(new int[] {i, j});
                }
            }
        }
        return edges;
    }
}
