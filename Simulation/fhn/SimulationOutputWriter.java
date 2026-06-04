package fhn;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

/** Writes raw simulation state only (no observables). Format version 2. */
public class SimulationOutputWriter implements AutoCloseable {
    private final BufferedWriter writer;
    private final SimulationConfig config;
    private final NeuralNetwork network;

    public SimulationOutputWriter(SimulationConfig config, NeuralNetwork network) throws IOException {
        this.config = config;
        this.network = network;
        Path path = Path.of(config.outputPath);
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        this.writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8);
        writeHeader();
    }

    private void writeHeader() throws IOException {
        boolean storeEdges = config.network != NetworkType.FULL;
        writer.write("# FHN_SIMULATION_OUTPUT v2\n");
        writer.write(String.format(Locale.US, "model=fitzhugh_nagumo%n"));
        writer.write(String.format(Locale.US, "N=%d%n", config.n));
        writer.write(String.format(Locale.US, "K=%.10g%n", config.k));
        writer.write(String.format(Locale.US, "dt=%.10g%n", config.dt));
        writer.write(String.format(Locale.US, "t_max=%.10g%n", config.tMax));
        writer.write(String.format(Locale.US, "network=%s%n", config.network.name()));
        writer.write(String.format(Locale.US, "p=%.10g%n", config.connectionProbability));
        writer.write(String.format(Locale.US, "ring_k=%d%n", config.ringHalfWidth));
        writer.write(String.format(Locale.US, "seed=%d%n", config.seed));
        writer.write(String.format(Locale.US, "realization=%d%n", config.realization));
        writer.write(String.format(Locale.US, "sample_every=%d%n", config.sampleEvery));
        writer.write(String.format(Locale.US, "edges_stored=%s%n", storeEdges));
        writer.write(String.format(Locale.US, "I=%.10g%n", FHNConstants.I));
        writer.write(String.format(Locale.US, "epsilon=%.10g%n", FHNConstants.EPSILON));
        writer.write(String.format(Locale.US, "a=%.10g%n", FHNConstants.A));
        writer.write(String.format(Locale.US, "b=%.10g%n", FHNConstants.B));
        writer.write(String.format(Locale.US, "ic_low=%.10g%n", FHNConstants.IC_LOW));
        writer.write(String.format(Locale.US, "ic_high=%.10g%n", FHNConstants.IC_HIGH));
        if (storeEdges) {
            writer.write("# EDGES i j (undirected, i < j)\n");
            for (int[] e : network.edges()) {
                writer.write(String.format(Locale.US, "EDGE %d %d%n", e[0], e[1]));
            }
        }
        writer.write("# Raw state samples: membrane potential v and recovery w\n");
        writer.write("BEGIN_STATE_SAMPLES\n");
    }

    public void writeState(double t, double[] v, double[] w) throws IOException {
        writer.write(String.format(Locale.US, "BEGIN_STATE t=%.10g%n", t));
        writer.write("v");
        for (double vi : v) {
            writer.write(String.format(Locale.US, " %.10g", vi));
        }
        writer.write("\n");
        writer.write("w");
        for (double wi : w) {
            writer.write(String.format(Locale.US, " %.10g", wi));
        }
        writer.write("\n");
        writer.write("END_STATE\n");
    }

    public void close() throws IOException {
        writer.write("END_STATE_SAMPLES\n");
        writer.close();
    }
}
