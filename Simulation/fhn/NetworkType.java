package fhn;

public enum NetworkType {
    FULL,
    RANDOM,
    RING;

    public static NetworkType parse(String s) {
        return NetworkType.valueOf(s.trim().toUpperCase());
    }
}
