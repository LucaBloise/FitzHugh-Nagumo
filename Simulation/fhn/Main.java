package fhn;

public class Main {
    public static void main(String[] args) {
        try {
            SimulationConfig config = SimulationConfig.fromArgs(args);
            new Simulator(config).run();
            System.out.println("Simulation written to: " + config.outputPath);
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            printUsage();
            System.exit(1);
        }
    }

    private static void printUsage() {
        System.err.println(
                """
                Usage: java fhn.Main [options]
                  --n <int>              neurons (default 512, must be > 500)
                  --k <double>           coupling K (default 1.0)
                  --dt <double>          fixed integrator step (default 0.005)
                  --tmax <double>        final time in s (default 500)
                  --network FULL|RANDOM|RING
                  --p <double>           connection probability (RANDOM)
                  --ring-k <int>         ring half-width k in [1,10] (RING)
                  --seed <long>          RNG seed (ICs and random graph)
                  --realization <int>    label only, for batch post-processing
                  --sample-every <int>   steps between raw state dumps (default 10)
                  --output <path>        optional; default Simulation/output/...

                One invocation = one run = one .txt file. Repeat with different
                --seed / --realization for multiple realizaciones.
                """);
    }
}
