// SPDX-FileCopyrightText: 2026 Michael Bickford (Woflo Labs)
// SPDX-License-Identifier: LicenseRef-WLCSL-1.0
// See repository-root LICENSE.md and LICENSES/WLCSL-1.0.md.

package woflo.wdym.engram;

import java.util.ArrayList;
import java.util.List;


final class EngramSegment {

    
    static final int MIN_BLOCK = 2;

    
    private static final int MAX_BLOCK = 128;

    
    private static final int MAX_LAG = 128;

    private EngramSegment() {}

    

    
    static int[] derivePairing(float[] w, int t, int dim) {
        int p = dim / 2;

        if (t < 2 || dim < 2) {
            int[] seq = new int[dim];
            for (int i = 0; i < dim; i++) seq[i] = i;
            return seq;
        }

        
        double[] means = new double[dim];
        for (int row = 0; row < t; row++) {
            int base = row * dim;
            for (int d = 0; d < dim; d++) {
                means[d] += w[base + d];
            }
        }
        double invT = 1.0 / t;
        for (int d = 0; d < dim; d++) means[d] *= invT;

        
        double[] norms = new double[dim];
        for (int row = 0; row < t; row++) {
            int base = row * dim;
            for (int d = 0; d < dim; d++) {
                double v = w[base + d] - means[d];
                norms[d] += v * v;
            }
        }
        for (int d = 0; d < dim; d++) {
            norms[d] = Math.max(1e-30, Math.sqrt(norms[d]));
        }

        
        
        double invTm1 = 1.0 / Math.max(1.0, t - 1.0);
        double[] corr = new double[dim * dim];

        for (int row = 0; row < t; row++) {
            int base = row * dim;
            for (int i = 0; i < dim; i++) {
                double vi = (w[base + i] - means[i]) / norms[i];
                int rowI = i * dim; 
                for (int j = i + 1; j < dim; j++) {
                    double vj = (w[base + j] - means[j]) / norms[j];
                    corr[rowI + j] += vi * vj;
                }
            }
        }

        
        for (int i = 0; i < dim; i++) {
            corr[i * dim + i] = 0.0;
            for (int j = i + 1; j < dim; j++) {
                double v = Math.abs(corr[i * dim + j] * invTm1);
                corr[i * dim + j] = v;
                corr[j * dim + i] = v;
            }
        }

        
        int[] pairing = new int[dim];
        boolean[] used = new boolean[dim];
        int pairIdx = 0;

        for (int iter = 0; iter < p; iter++) {
            double bestVal = -1.0;
            int bestI = 0, bestJ = 0;

            for (int i = 0; i < dim; i++) {
                if (used[i]) continue;
                for (int j = i + 1; j < dim; j++) {
                    if (used[j]) continue;
                    if (corr[i * dim + j] > bestVal) {
                        bestVal = corr[i * dim + j];
                        bestI = i;
                        bestJ = j;
                    }
                }
            }

            pairing[pairIdx * 2] = bestI;
            pairing[pairIdx * 2 + 1] = bestJ;
            used[bestI] = true;
            used[bestJ] = true;
            pairIdx++;
        }

        return pairing;
    }

    
    static float[] applyPairing(float[] w, int t, int dim, int[] pairing) {
        float[] out = new float[t * dim];
        for (int row = 0; row < t; row++) {
            int base = row * dim;
            for (int d = 0; d < dim; d++) {
                out[base + d] = w[base + pairing[d]];
            }
        }
        return out;
    }

    

    
    static int[] deriveBlockSizes(float[] w, int t, int dim) {
        int mac = Math.min(MAX_BLOCK, Math.max(MIN_BLOCK, autocorrelationLength(w, t, dim)));
        int mic = Math.max(MIN_BLOCK, mac / 4);
        return new int[]{mac, mic};
    }

    
    private static int autocorrelationLength(float[] w, int t, int dim) {
        if (t < 3) return MIN_BLOCK;

        
        int velLen = t - 1;
        double[] velNormSq = new double[velLen];

        for (int i = 0; i < velLen; i++) {
            int baseCurr = i * dim;
            int baseNext = baseCurr + dim;
            double sqSum = 0.0;
            for (int d = 0; d < dim; d++) {
                double diff = w[baseNext + d] - w[baseCurr + d];
                sqSum += diff * diff;
            }
            velNormSq[i] = sqSum; 
        }

        
        double mean = 0.0;
        for (double v : velNormSq) mean += v;
        mean /= velLen;

        double[] centered = new double[velLen];
        for (int i = 0; i < velLen; i++) centered[i] = velNormSq[i] - mean;

        
        double var = 0.0;
        for (double v : centered) var += v * v;
        if (var < 1e-30) return MIN_BLOCK;

        
        int maxLag = Math.min(MAX_LAG, velLen / 2);
        for (int lag = 1; lag <= maxLag; lag++) {
            double acf = 0.0;
            for (int i = 0; i < velLen - lag; i++) {
                acf += centered[i] * centered[i + lag];
            }
            acf /= var;
            if (acf <= 1e-10) {
                return Math.max(MIN_BLOCK, lag);
            }
        }

        return Math.max(MIN_BLOCK, maxLag);
    }

    

    
    static List<int[]> segment(float[] w, int t, int dim, int maxBlock) {
        List<int[]> segments = new ArrayList<>();

        if (t <= MIN_BLOCK) {
            if (t > 0) segments.add(new int[]{0, t});
            return segments;
        }

        double[] errors = linearPredictionErrors(w, t, dim);
        if (errors.length == 0) {
            segments.add(new int[]{0, t});
            return segments;
        }

        
        
        
        double n = errors.length;
        double eMean = 0.0;
        for (double e : errors) eMean += e;
        eMean /= n;

        double eVar = 0.0;
        for (double e : errors) eVar += (e - eMean) * (e - eMean);
        eVar /= n;

        
        double threshold = eMean + 3.0 * Math.sqrt(eVar);

        int blockStart = 0;

        
        for (int ei = 0; ei < errors.length; ei++) {
            int tIdx = ei + 2;
            int blockLen = tIdx - blockStart;

            
            if (blockLen >= maxBlock) {
                segments.add(new int[]{blockStart, tIdx});
                blockStart = tIdx;
                continue;
            }

            
            if (blockLen >= MIN_BLOCK && errors[ei] > threshold) {
                segments.add(new int[]{blockStart, tIdx});
                blockStart = tIdx;
            }
        }

        
        if (blockStart < t) {
            
            if (t - blockStart < MIN_BLOCK && !segments.isEmpty()) {
                segments.get(segments.size() - 1)[1] = t;
            } else {
                segments.add(new int[]{blockStart, t});
            }
        }

        return segments;
    }

    
    private static double[] linearPredictionErrors(float[] w, int t, int dim) {
        if (t < 3) return new double[0];

        int n = t - 2;
        double[] errors = new double[n]; 

        for (int i = 2; i < t; i++) {
            int base  = i * dim;
            int base1 = base - dim;       
            int base2 = base1 - dim;      
            double sq = 0.0;
            for (int d = 0; d < dim; d++) {
                double pred = 2.0 * w[base1 + d] - w[base2 + d];
                double diff = w[base + d] - pred;
                sq += diff * diff;
            }
            errors[i - 2] = sq; 
        }

        return errors;
    }

    

    
    static float[] flatten(List<float[]> trajectory, int dim) {
        float[] flat = new float[trajectory.size() * dim];
        for (int i = 0; i < trajectory.size(); i++) {
            System.arraycopy(trajectory.get(i), 0, flat, i * dim, dim);
        }
        return flat;
    }

    
    static List<float[]> extractSegment(float[] flat, int start, int end, int dim) {
        List<float[]> segment = new ArrayList<>(end - start);
        for (int i = start; i < end; i++) {
            float[] row = new float[dim];
            System.arraycopy(flat, i * dim, row, 0, dim);
            segment.add(row);
        }
        return segment;
    }
}
