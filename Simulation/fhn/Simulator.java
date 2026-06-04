package fhn;

import java.io.IOException;
import java.util.Random;

public class Simulator {
    private final SimulationConfig config;
    private final Random rng;
    private final NeuralNetwork network;
    private final FitzHughNagumoDynamics dynamics;
    private final double[] v;
    private final double[] w;

    public Simulator(SimulationConfig config) {
        this.config = config;
        this.rng = new Random(config.seed);
        this.network = NeuralNetwork.build(config, rng);
        this.dynamics = new FitzHughNagumoDynamics(network, config.k);
        this.v = new double[config.n];
        this.w = new double[config.n];
        initializeState();
    }

    private void initializeState() {
        double lo = FHNConstants.IC_LOW;
        double hi = FHNConstants.IC_HIGH;
        for (int i = 0; i < config.n; i++) {
            v[i] = lo + (hi - lo) * rng.nextDouble();
            w[i] = lo + (hi - lo) * rng.nextDouble();
        }
    }

    public void run() throws IOException {
        try (SimulationOutputWriter out = new SimulationOutputWriter(config, network)) {
            int steps = (int) Math.ceil(config.tMax / config.dt);
            int sampleEvery = Math.max(1, config.sampleEvery);

            out.writeState(0.0, v, w);

            for (int step = 1; step <= steps; step++) {
                dynamics.rk4Step(v, w, config.dt);
                assertFinite(step, v, w);
                if (step % sampleEvery == 0) {
                    out.writeState(step * config.dt, v, w);
                }
            }
        }
    }

    private static void assertFinite(int step, double[] v, double[] w) {
        for (int i = 0; i < v.length; i++) {
            if (!Double.isFinite(v[i]) || !Double.isFinite(w[i])) {
                throw new IllegalStateException(
                        "Non-finite state at integrator step "
                                + step
                                + ". Try a smaller --dt (e.g. 0.005 or 0.001), especially for"
                                + " network=FULL with large N and K.");
            }
        }
    }
}
