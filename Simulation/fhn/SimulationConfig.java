package fhn;

/**
 * Adjustable simulation parameters. Constants live in {@link FHNConstants}.
 *
 * <p>{@code realization} is only a label in the output file (e.g. "run 3 of a batch").
 * It does not repeat the simulation; launch Java once per realization.
 */
public class SimulationConfig {
    public int n = 512;
    public double k = 1.0;
    /** Default 0.005: stable for FULL, N>500, K=1 with RK4 (0.01 can blow up). */
    public double dt = 0.005;
    public double tMax = 500.0;
    public NetworkType network = NetworkType.FULL;
    public double connectionProbability = 0.5;
    public int ringHalfWidth = 3;
    public long seed = 42L;
    /** Label for this run when post-processing batches (>10 realizaciones). */
    public int realization = 1;
    /** Integrator steps between raw state dumps (v, w). */
    public int sampleEvery = 10;
    /** If null, {@link OutputFileName#buildDefault} is used at run time. */
    public String outputPath = null;
    public boolean outputPathExplicit = false;

    public static SimulationConfig fromArgs(String[] args) {
        SimulationConfig c = new SimulationConfig();
        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            if (!arg.startsWith("--")) {
                continue;
            }
            String key = arg.substring(2);
            if ("output".equals(key)) {
                if (i + 1 >= args.length) {
                    throw new IllegalArgumentException("Missing value for --output");
                }
                c.outputPath = args[++i];
                c.outputPathExplicit = true;
                continue;
            }
            if (i + 1 >= args.length) {
                throw new IllegalArgumentException("Missing value for --" + key);
            }
            String val = args[++i];
            switch (key) {
                case "n" -> c.n = Integer.parseInt(val);
                case "k" -> c.k = Double.parseDouble(val);
                case "dt" -> c.dt = Double.parseDouble(val);
                case "tmax" -> c.tMax = Double.parseDouble(val);
                case "network" -> c.network = NetworkType.parse(val);
                case "p" -> c.connectionProbability = Double.parseDouble(val);
                case "ring-k" -> c.ringHalfWidth = Integer.parseInt(val);
                case "seed" -> c.seed = Long.parseLong(val);
                case "realization" -> c.realization = Integer.parseInt(val);
                case "sample-every" -> c.sampleEvery = Integer.parseInt(val);
                default -> throw new IllegalArgumentException("Unknown option: --" + key);
            }
        }
        if (!c.outputPathExplicit) {
            c.outputPath = OutputFileName.buildDefault(c);
        }
        c.outputPath = SimulationPaths.resolveOutputPath(c.outputPath).toString();
        return c;
    }
}
