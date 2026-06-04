package fhn;

/** RHS and RK4 step for coupled FitzHugh–Nagumo neurons. */
public class FitzHughNagumoDynamics {
    private final NeuralNetwork network;
    private final double couplingK;

    private final double[] k1v;
    private final double[] k2v;
    private final double[] k3v;
    private final double[] k4v;
    private final double[] k1w;
    private final double[] k2w;
    private final double[] k3w;
    private final double[] k4w;
    private final double[] tv;
    private final double[] tw;

    public FitzHughNagumoDynamics(NeuralNetwork network, double couplingK) {
        this.network = network;
        this.couplingK = couplingK;
        int n = network.size();
        k1v = new double[n];
        k2v = new double[n];
        k3v = new double[n];
        k4v = new double[n];
        k1w = new double[n];
        k2w = new double[n];
        k3w = new double[n];
        k4w = new double[n];
        tv = new double[n];
        tw = new double[n];
    }

    public void rk4Step(double[] v, double[] w, double dt) {
        int n = network.size();
        derivatives(v, w, k1v, k1w);
        addState(v, w, k1v, k1w, 0.5 * dt, tv, tw);
        derivatives(tv, tw, k2v, k2w);
        addState(v, w, k2v, k2w, 0.5 * dt, tv, tw);
        derivatives(tv, tw, k3v, k3w);
        addState(v, w, k3v, k3w, dt, tv, tw);
        derivatives(tv, tw, k4v, k4w);

        for (int i = 0; i < n; i++) {
            v[i] += dt * (k1v[i] + 2.0 * k2v[i] + 2.0 * k3v[i] + k4v[i]) / 6.0;
            w[i] += dt * (k1w[i] + 2.0 * k2w[i] + 2.0 * k3w[i] + k4w[i]) / 6.0;
        }
    }

    private void addState(
            double[] v,
            double[] w,
            double[] dv,
            double[] dw,
            double h,
            double[] outV,
            double[] outW) {
        int n = network.size();
        for (int i = 0; i < n; i++) {
            outV[i] = v[i] + h * dv[i];
            outW[i] = w[i] + h * dw[i];
        }
    }

    private void derivatives(double[] v, double[] w, double[] dv, double[] dw) {
        int n = network.size();
        if (network.isFullyConnected()) {
            double sumV = 0.0;
            for (int i = 0; i < n; i++) {
                sumV += v[i];
            }
            for (int i = 0; i < n; i++) {
                double coupling = sumV - n * v[i];
                dv[i] = v[i] - (v[i] * v[i] * v[i]) / 3.0 - w[i] + FHNConstants.I + couplingK * coupling;
                dw[i] = FHNConstants.EPSILON * (v[i] + FHNConstants.A - FHNConstants.B * w[i]);
            }
            return;
        }

        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j : network.neighborsOf(i)) {
                sum += v[j] - v[i];
            }
            dv[i] = v[i] - (v[i] * v[i] * v[i]) / 3.0 - w[i] + FHNConstants.I + couplingK * sum;
            dw[i] = FHNConstants.EPSILON * (v[i] + FHNConstants.A - FHNConstants.B * w[i]);
        }
    }
}
