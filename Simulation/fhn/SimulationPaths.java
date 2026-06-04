package fhn;

import java.nio.file.Files;
import java.nio.file.Path;

/** Resolve paths relative to the Simulation/ directory, not the JVM cwd. */
public final class SimulationPaths {
    private SimulationPaths() {}

    public static Path simulationDirectory() {
        String override = System.getenv("FHN_SIMULATION_ROOT");
        if (override != null && !override.isBlank()) {
            return Path.of(override).toAbsolutePath().normalize();
        }

        Path cwd = Path.of("").toAbsolutePath().normalize();
        if (looksLikeSimulationDir(cwd)) {
            return cwd;
        }

        Path nested = cwd.resolve("Simulation");
        if (looksLikeSimulationDir(nested)) {
            return nested;
        }

        Path parent = cwd.getParent();
        while (parent != null) {
            Path sim = parent.resolve("Simulation");
            if (looksLikeSimulationDir(sim)) {
                return sim;
            }
            if (looksLikeSimulationDir(parent)) {
                return parent;
            }
            parent = parent.getParent();
        }

        return nested;
    }

    public static Path resolveOutputPath(String outputPath) {
        Path out = Path.of(outputPath);
        if (out.isAbsolute()) {
            return out.normalize();
        }
        return simulationDirectory().resolve(out).normalize();
    }

    private static boolean looksLikeSimulationDir(Path dir) {
        return Files.isDirectory(dir)
                && (Files.isRegularFile(dir.resolve("compile.bat"))
                        || Files.isDirectory(dir.resolve("bin"))
                        || Files.isDirectory(dir.resolve("fhn")));
    }
}
