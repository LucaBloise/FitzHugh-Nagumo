package fhn;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Locale;

/** Default output path: output/fhn_{params}_{timestamp}.txt */
public final class OutputFileName {
    private static final DateTimeFormatter TIMESTAMP =
            DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss");

    private OutputFileName() {}

    public static String buildDefault(SimulationConfig c) {
        String ts = LocalDateTime.now().format(TIMESTAMP);
        String network = c.network.name().toLowerCase(Locale.US);
        return String.format(
                Locale.US,
                "output/fhn_%s_K%.4g_p%.4g_ring%d_N%d_seed%d_real%d_%s.txt",
                network,
                c.k,
                c.connectionProbability,
                c.ringHalfWidth,
                c.n,
                c.seed,
                c.realization,
                ts);
    }
}
