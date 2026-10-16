// SPDX-FileCopyrightText: 2026 Michael Bickford (Woflo Labs)
// SPDX-License-Identifier: LicenseRef-WLCSL-1.0
// See repository-root LICENSE.md and LICENSES/WLCSL-1.0.md.

package woflo.wdym.engram;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;


public class EngramBrain {

    
    private static final double MACHINE_EPS = 1.192_092_9e-7; 
    private static final double RIDGE_SCALE = 1e-4;
    
    private static final int MIN_TIMESTEPS_AR2 = 4;
    
    private static final int MIN_TIMESTEPS_AR1 = 2;

    
    private static final int DREAM_CAP_PER_DIM = 4;
    static final int MIN_WELL_OBS  = 10; 

    
    private static final int WELL_RESERVED_BYTES = 2048;

    
    private static final int ABSORPTION_HORIZON = 1024;

    
    private final int dim;
    private final int pairs; 
    private final float invPairs; 
    private float alpha = 0.005f;
    private String name = "";
    private String author = "woflo";
    private String pronouns = "they/them";
    private int dreamCapacityOverride = -1; 

    
    private final Map<String, Well> wells = new LinkedHashMap<>();

    
    private int[] referencePairing;

    
    private final ArrayDeque<DreamEntry> dream = new ArrayDeque<>();

    private int nextWellId   = 0;
    private int totalAbsorbed = 0;

    
    
    
    
    private final float[] _sAA, _sBB, _sTT;
    private final float[] _sABre, _sABim;
    private final float[] _sTAre, _sTAim;
    private final float[] _sTBre, _sTBim;

    
    private JsonObject annotations;

    
    
    private double cachedGlobalScale = -1.0; 
    private long cachedGlobalScaleGeneration = -1;
    private long wellMutationGeneration = 0; 

    

    
    static class Well {
        double[] sumK; 
        int count;

        Well(int pairs) {
            this.sumK = new double[pairs * 2];
        }

        
        void absorb(float[] kRe, float[] kIm) {
            if (count >= ABSORPTION_HORIZON) {
                
                double scale = (double) (ABSORPTION_HORIZON - 1) / count;
                for (int j = 0; j < sumK.length; j++) {
                    sumK[j] *= scale;
                }
                count = ABSORPTION_HORIZON - 1;
            }
            
            int si = 0;
            for (int j = 0; j < kRe.length; j++, si += 2) {
                sumK[si]     += kRe[j];
                sumK[si + 1] += kIm[j];
            }
            count++;
        }

        
        float[][] centroid() {
            int p = sumK.length / 2;
            float[] re = new float[p];
            float[] im = new float[p];
            if (count == 0) return new float[][]{re, im};
            double inv = 1.0 / count;
            
            int si = 0;
            for (int j = 0; j < p; j++, si += 2) {
                re[j] = (float) (sumK[si]     * inv);
                im[j] = (float) (sumK[si + 1] * inv);
            }
            return new float[][]{re, im};
        }

        
        Well copy(int pairs) {
            Well w = new Well(pairs);
            System.arraycopy(this.sumK, 0, w.sumK, 0, this.sumK.length);
            w.count = this.count;
            return w;
        }
    }

    

    
    record DreamEntry(float[] kRe, float[] kIm,
                      float[] gRe, float[] gIm,
                      float[] s) {}

    

    
    public record DreamProfile(
            double kMagMean,       
            double kMagStd,        
            double gMagMean,       
            double gMagStd,        
            double textureMean,    
            double textureStd,     
            double textureTrend,   
            double rhythmicity,    
            double lastKMag,       
            double lastGMag,       
            double lastTexture,    
            int count              
    ) {}

    

    public record MeasureResult(
            float drift,
            String nearestWell,
            float nearestDist,
            Map<String, Float> wellProfile,
            float[] kRe,           
            float[] kIm            
    ) {}

    

    public EngramBrain(int dim) {
        this.dim   = dim;
        this.pairs = Math.max(1, dim / 2);
        this.invPairs = 1.0f / this.pairs;
        
        _sAA  = new float[pairs]; _sBB  = new float[pairs]; _sTT  = new float[pairs];
        _sABre = new float[pairs]; _sABim = new float[pairs];
        _sTAre = new float[pairs]; _sTAim = new float[pairs];
        _sTBre = new float[pairs]; _sTBim = new float[pairs];
    }

    
    int dreamCapacity() {
        return dreamCapacityOverride > 0 ? dreamCapacityOverride : Math.max(500, dim * DREAM_CAP_PER_DIM);
    }
    public int getDreamCapacity() { return dreamCapacity(); }
    public int getDefaultDreamCapacity() { return Math.max(500, dim * DREAM_CAP_PER_DIM); }
    public void setDreamCapacity(int cap) { this.dreamCapacityOverride = cap <= 0 ? -1 : Math.max(100, cap); }

    public int getDreamSize()     { return dream.size(); }
    public int getDim()           { return dim; }
    public int getPairs()         { return pairs; }
    public int getWellCount()     { return wells.size(); }
    public int getTotalAbsorbed() { return totalAbsorbed; }
    public String getName()       { return name; }
    public void setName(String name) { this.name = name != null ? name : ""; }
    public float getAlpha()        { return alpha; }
    public void setAlpha(float alpha) { this.alpha = Math.max(0.001f, Math.min(0.05f, alpha)); }
    public String getAuthor()      { return author; }
    public void setAuthor(String author) { this.author = author != null ? author : ""; }
    public String getPronouns()    { return pronouns; }
    public void setPronouns(String pronouns) { this.pronouns = pronouns != null ? pronouns : ""; }

    

    
    public JsonObject getAnnotations() {
        return annotations != null ? annotations : new JsonObject();
    }

    
    public String getAnnotation(String key) {
        if (annotations == null || !annotations.has(key)) return null;
        var el = annotations.get(key);
        return el.isJsonPrimitive() ? el.getAsString() : el.toString();
    }

    
    public void setAnnotation(String key, String value) {
        if (annotations == null) annotations = new JsonObject();
        annotations.addProperty(key, value);
    }

    
    public void removeAnnotation(String key) {
        if (annotations != null) annotations.remove(key);
    }

    
    public EngramBrain copy() {
        EngramBrain b = new EngramBrain(dim);
        b.alpha = this.alpha;
        b.author = this.author;
        b.pronouns = this.pronouns;
        b.dreamCapacityOverride = this.dreamCapacityOverride;
        b.nextWellId = this.nextWellId;
        b.totalAbsorbed = 0;
        if (this.referencePairing != null) {
            b.referencePairing = this.referencePairing.clone();
        }
        for (var e : this.wells.entrySet()) {
            b.wells.put(e.getKey(), e.getValue().copy(this.pairs));
        }
        for (var de : this.dream) {
            b.dream.addLast(new DreamEntry(
                    de.kRe.clone(), de.kIm.clone(),
                    de.gRe.clone(), de.gIm.clone(),
                    de.s.clone()));
        }

        
        
        
        JsonObject defaults = SeedDefaults.get();

        
        if (this.annotations != null && !this.annotations.isEmpty()) {
            b.annotations = this.annotations.deepCopy();
            b.name = this.name;
            b.pronouns = this.pronouns;
        } else {
            b.annotations = new JsonObject();
            b.name = defaults.has("name") ? defaults.get("name").getAsString() : this.name;
            b.pronouns = defaults.has("pronouns") ? defaults.get("pronouns").getAsString() : this.pronouns;
            
            b.annotations.addProperty("creator_notes",
                    "roar. I'm a mod. i am not a mod. made by woflo");
        }

        
        for (String key : defaults.keySet()) {
            if (key.equals("name") || key.equals("pronouns")) continue;
            if (!b.annotations.has(key)) {
                b.annotations.add(key, defaults.get(key).deepCopy());
            }
        }

        return b;
    }

    

    
    static final class SeedDefaults {
        private static volatile JsonObject instance;

        static JsonObject get() {
            if (instance != null) return instance;
            synchronized (SeedDefaults.class) {
                if (instance != null) return instance;
                instance = load();
                return instance;
            }
        }

        private static JsonObject load() {
            try (var is = SeedDefaults.class.getClassLoader().getResourceAsStream("engram/seed.json")) {
                if (is != null) {
                    return JsonParser.parseString(
                            new String(is.readAllBytes(), java.nio.charset.StandardCharsets.UTF_8))
                            .getAsJsonObject();
                }
            } catch (Exception e) {
                woflo.wdym.Whatdoyoumean.LOGGER.warn("[Engram] Failed to load seed.json: {}", e.getMessage());
            }
            
            JsonObject fallback = new JsonObject();
            fallback.addProperty("name", "The Doctor");
            fallback.addProperty("system_card", "You are The Doctor.");
            return fallback;
        }
    }

    

    
    record FitResult(float[] kRe, float[] kIm,
                     float[] gRe, float[] gIm,
                     float[] pairRms) {}

    

    
    FitResult fitFlat(float[] paired, int start, int end) {
        int t = end - start;
        if (t < MIN_TIMESTEPS_AR1) return null;

        
        if (t < MIN_TIMESTEPS_AR2) {
            return fitAR1(paired, start, end);
        }

        
        return fitAR2(paired, start, end);
    }

    
    private FitResult fitAR1(float[] paired, int start, int end) {
        int t = end - start;
        int n = t - 1; 

        
        
        final float[] sAA   = _sAA;
        final float[] sTT   = _sTT;
        final float[] sTAre = _sTAre;
        final float[] sTAim = _sTAim;
        java.util.Arrays.fill(sAA,   0, pairs, 0.0f);
        java.util.Arrays.fill(sTT,   0, pairs, 0.0f);
        java.util.Arrays.fill(sTAre, 0, pairs, 0.0f);
        java.util.Arrays.fill(sTAim, 0, pairs, 0.0f);

        for (int i = 0; i < n; i++) {
            int base0 = (start + i) * dim;
            int base1 = base0 + dim;

            
            
            int ri = 0;
            for (int j = 0; j < pairs; j++, ri += 2) {
                int ii = ri + 1;

                float a_re = paired[base0 + ri], a_im = paired[base0 + ii];
                float t_re = paired[base1 + ri], t_im = paired[base1 + ii];

                
                
                sAA[j]   = Math.fma(a_re, a_re,   Math.fma(a_im,  a_im,  sAA[j]));
                sTT[j]   = Math.fma(t_re, t_re,   Math.fma(t_im,  t_im,  sTT[j]));
                sTAre[j] = Math.fma(t_re, a_re,   Math.fma(t_im,  a_im,  sTAre[j]));
                sTAim[j] = Math.fma(t_im, a_re,   Math.fma(-t_re, a_im,  sTAim[j]));
            }
        }

        float[] kRe = new float[pairs];
        float[] kIm = new float[pairs];
        float[] pairRms = new float[pairs];

        for (int j = 0; j < pairs; j++) {
            double saa = sAA[j];
            double sAA_r = saa + Math.max(MACHINE_EPS, RIDGE_SCALE * saa);

            if (sAA_r < MACHINE_EPS) {
                kRe[j] = 2.0f; 
                pairRms[j] = (float) Math.sqrt(Math.max(0.0, sTT[j]) / Math.max(1, n));
                continue;
            }

            
            double tar = sTAre[j], tai = sTAim[j];
            double kr = tar / sAA_r;
            double ki = tai / sAA_r;
            kRe[j] = (float) kr;
            kIm[j] = (float) ki;

            
            double kNormSq    = kr * kr + ki * ki;
            double kConjTA_re = kr * tar + ki * tai;
            double errSq = sTT[j] + kNormSq * saa - 2.0 * kConjTA_re;
            pairRms[j] = (float) Math.sqrt(Math.max(0.0, errSq) / n);
        }

        
        return new FitResult(kRe, kIm, new float[pairs], new float[pairs], pairRms);
    }

    
    private FitResult fitAR2(float[] paired, int start, int end) {
        int t = end - start;
        int n = t - 2;

        
        
        final float[] sAA   = _sAA;   final float[] sBB   = _sBB;
        final float[] sTT   = _sTT;   final float[] sABre = _sABre;
        final float[] sABim = _sABim; final float[] sTAre = _sTAre;
        final float[] sTAim = _sTAim; final float[] sTBre = _sTBre;
        final float[] sTBim = _sTBim;
        java.util.Arrays.fill(sAA,   0, pairs, 0.0f);
        java.util.Arrays.fill(sBB,   0, pairs, 0.0f);
        java.util.Arrays.fill(sTT,   0, pairs, 0.0f);
        java.util.Arrays.fill(sABre, 0, pairs, 0.0f);
        java.util.Arrays.fill(sABim, 0, pairs, 0.0f);
        java.util.Arrays.fill(sTAre, 0, pairs, 0.0f);
        java.util.Arrays.fill(sTAim, 0, pairs, 0.0f);
        java.util.Arrays.fill(sTBre, 0, pairs, 0.0f);
        java.util.Arrays.fill(sTBim, 0, pairs, 0.0f);

        for (int i = 0; i < n; i++) {
            int base0 = (start + i) * dim;
            int base1 = base0 + dim;
            int base2 = base1 + dim;

            
            int ri = 0;
            for (int j = 0; j < pairs; j++, ri += 2) {
                int ii = ri + 1;

                float b_re = paired[base0 + ri], b_im = paired[base0 + ii];
                float a_re = paired[base1 + ri], a_im = paired[base1 + ii];
                float t_re = paired[base2 + ri], t_im = paired[base2 + ii];

                
                sAA[j]   = Math.fma(a_re, a_re,   Math.fma(a_im,  a_im,  sAA[j]));
                sBB[j]   = Math.fma(b_re, b_re,   Math.fma(b_im,  b_im,  sBB[j]));
                sTT[j]   = Math.fma(t_re, t_re,   Math.fma(t_im,  t_im,  sTT[j]));
                sABre[j] = Math.fma(a_re, b_re,   Math.fma(a_im,  b_im,  sABre[j]));
                sABim[j] = Math.fma(a_im, b_re,   Math.fma(-a_re, b_im,  sABim[j]));
                sTAre[j] = Math.fma(t_re, a_re,   Math.fma(t_im,  a_im,  sTAre[j]));
                sTAim[j] = Math.fma(t_im, a_re,   Math.fma(-t_re, a_im,  sTAim[j]));
                sTBre[j] = Math.fma(t_re, b_re,   Math.fma(t_im,  b_im,  sTBre[j]));
                sTBim[j] = Math.fma(t_im, b_re,   Math.fma(-t_re, b_im,  sTBim[j]));
            }
        }

        float[] kRe = new float[pairs];
        float[] kIm = new float[pairs];
        float[] gRe = new float[pairs];
        float[] gIm = new float[pairs];
        float[] pairRms = new float[pairs];

        for (int j = 0; j < pairs; j++) {
            double saa = sAA[j], sbb = sBB[j];
            double trace = saa + sbb;
            double ridge = Math.max(MACHINE_EPS, RIDGE_SCALE * trace * 0.5);
            double sAA_r = saa + ridge;
            double sBB_r = sbb + ridge;

            double abr = sABre[j], abi = sABim[j];
            double abNormSq = abr * abr + abi * abi;
            double det = sAA_r * sBB_r - abNormSq;

            if (Math.abs(det) < MACHINE_EPS * trace * trace) {
                kRe[j] = 2.0f;
                gRe[j] = 1.0f;
                pairRms[j] = (float) Math.sqrt(Math.max(0.0, sTT[j]) / Math.max(1, n));
                continue;
            }

            double invDet = 1.0 / det;
            double tar = sTAre[j], tai = sTAim[j];
            double tbr = sTBre[j], tbi = sTBim[j];

            double cABtB_re =  abr * tbr + abi * tbi;
            double cABtB_im = -abi * tbr + abr * tbi;
            double kr = (tar * sBB_r - cABtB_re) * invDet;
            double ki = (tai * sBB_r - cABtB_im) * invDet;
            kRe[j] = (float) kr;
            kIm[j] = (float) ki;

            double abTA_re = abr * tar - abi * tai;
            double abTA_im = abr * tai + abi * tar;
            double gr = (abTA_re - sAA_r * tbr) * invDet;
            double gi = (abTA_im - sAA_r * tbi) * invDet;
            gRe[j] = (float) gr;
            gIm[j] = (float) gi;

            double kNormSq    = kr * kr + ki * ki;
            double gNormSq    = gr * gr + gi * gi;
            double kConjTA_re = kr * tar + ki * tai;
            double gConjTB_re = gr * tbr + gi * tbi;
            double kgConj_re  = kr * gr + ki * gi;
            double kgConj_im  = ki * gr - kr * gi;
            double kgAB_re    = kgConj_re * abr - kgConj_im * abi;

            double errSq = sTT[j]
                    + kNormSq * saa
                    + gNormSq * sbb
                    - 2.0 * kConjTA_re
                    + 2.0 * gConjTB_re
                    - 2.0 * kgAB_re;

            pairRms[j] = (float) Math.sqrt(Math.max(0.0, errSq) / n);
        }

        return new FitResult(kRe, kIm, gRe, gIm, pairRms);
    }

    

    
    public MeasureResult measure(List<float[]> trajectory) {
        if (trajectory.size() < MIN_TIMESTEPS_AR1) return null;

        
        float[][] centroid = computeKCentroid(trajectory);
        if (centroid == null) return null;

        float[] kRe = centroid[0];
        float[] kIm = centroid[1];

        if (wells.isEmpty()) {
            return new MeasureResult(50f, "unknown", 50f, Map.of(), kRe, kIm);
        }

        
        
        String bestName  = null;
        float bestDistSq = Float.MAX_VALUE;
        String bestEstablished = null;
        float bestEstablishedDistSq = Float.MAX_VALUE;
        Map<String, Float> profile = new LinkedHashMap<>();

        for (var entry : wells.entrySet()) {
            float[][] c    = entry.getValue().centroid();
            float     dSq  = rmsDistanceSq(kRe, kIm, c[0], c[1]);
            boolean established = entry.getValue().count >= MIN_WELL_OBS;

            if (established) {
                
                profile.put(entry.getKey(), (float) Math.sqrt(dSq));
                if (dSq < bestEstablishedDistSq) {
                    bestEstablishedDistSq = dSq;
                    bestEstablished = entry.getKey();
                }
            }
            if (dSq < bestDistSq) {
                bestDistSq = dSq;
                bestName = entry.getKey();
            }
        }

        
        
        if (bestEstablished != null && bestName != null
                && !bestName.equals(bestEstablished)
                && bestDistSq > bestEstablishedDistSq * 0.81f) {
            bestName = bestEstablished;
            bestDistSq = bestEstablishedDistSq;
        }

        float bestDist;
        if (bestName == null) {
            bestName = "unknown";
            bestDist = 50f;
        } else {
            bestDist = (float) Math.sqrt(bestDistSq);
        }

        float drift = normalizeDrift(kRe, kIm, bestDist);
        return new MeasureResult(drift, bestName, bestDist, profile, kRe, kIm);
    }

    private float normalizeDrift(float[] kRe, float[] kIm, float rawDist) {
        if (wells.isEmpty()) return 50f;
        ensureGlobalScale();
        return (float) Math.min(100.0, rawDist / cachedGlobalScale * 100.0);
    }

    

    
    public String absorb(List<float[]> trajectory) {
        if (trajectory.size() < MIN_TIMESTEPS_AR1) {
            totalAbsorbed++;
            return "skipped";
        }

        
        List<FitResult> fits = computeBlockFits(trajectory);
        if (fits.isEmpty()) {
            totalAbsorbed++;
            return "skipped";
        }

        float nInv = 1.0f / fits.size();

        
        float[] centroidKRe = new float[pairs];
        float[] centroidKIm = new float[pairs];
        for (FitResult fit : fits) {
            for (int j = 0; j < pairs; j++) {
                centroidKRe[j] += fit.kRe[j];
                centroidKIm[j] += fit.kIm[j];
            }
        }
        for (int j = 0; j < pairs; j++) {
            centroidKRe[j] *= nInv;
            centroidKIm[j] *= nInv;
        }

        
        String wellName = routeUnsupervised(centroidKRe, centroidKIm);
        wells.computeIfAbsent(wellName, k -> new Well(pairs));
        Well well = wells.get(wellName);

        
        for (FitResult fit : fits) {
            well.absorb(fit.kRe, fit.kIm);
        }
        totalAbsorbed++;
        wellMutationGeneration++; 

        
        
        float[] meanKRe = centroidKRe.clone();
        float[] meanKIm = centroidKIm.clone();
        float[] meanGRe = new float[pairs];
        float[] meanGIm = new float[pairs];
        float[] meanS   = new float[pairs];
        for (FitResult fit : fits) {
            for (int j = 0; j < pairs; j++) {
                meanGRe[j] += fit.gRe[j];
                meanGIm[j] += fit.gIm[j];
                meanS[j]   += fit.pairRms[j];
            }
        }
        for (int j = 0; j < pairs; j++) {
            meanGRe[j] *= nInv;
            meanGIm[j] *= nInv;
            meanS[j]   *= nInv;
        }

        int dreamCap = dreamCapacity();
        dream.addLast(new DreamEntry(meanKRe, meanKIm, meanGRe, meanGIm, meanS));
        while (dream.size() > dreamCap) dream.removeFirst();

        woflo.wdym.Whatdoyoumean.LOGGER.debug(
                "[Engram] Absorbed {} vecs ({} blocks) → well '{}' (total absorbs: {})",
                trajectory.size(), fits.size(), wellName, totalAbsorbed);
        return wellName;
    }

    

    
    private List<FitResult> computeBlockFits(List<float[]> trajectory) {
        float[] flat = EngramSegment.flatten(trajectory, dim);
        int t = trajectory.size();

        
        if (referencePairing == null) {
            referencePairing = EngramSegment.derivePairing(flat, t, dim);
        }

        
        float[] paired = EngramSegment.applyPairing(flat, t, dim, referencePairing);

        
        int[] blockSizes = EngramSegment.deriveBlockSizes(paired, t, dim);
        List<int[]> segments = EngramSegment.segment(paired, t, dim, blockSizes[0]);

        
        List<FitResult> fits = new ArrayList<>(segments.size());
        for (int[] seg : segments) {
            FitResult fit = fitFlat(paired, seg[0], seg[1]);
            if (fit != null) {
                fits.add(fit);
            }
        }
        return fits;
    }

    
    float[][] computeKCentroid(List<float[]> trajectory) {
        List<FitResult> fits = computeBlockFits(trajectory);
        if (fits.isEmpty()) return null;

        float[] centroidRe = new float[pairs];
        float[] centroidIm = new float[pairs];
        for (FitResult fit : fits) {
            for (int j = 0; j < pairs; j++) {
                centroidRe[j] += fit.kRe[j];
                centroidIm[j] += fit.kIm[j];
            }
        }
        float inv = 1.0f / fits.size();
        for (int j = 0; j < pairs; j++) {
            centroidRe[j] *= inv;
            centroidIm[j] *= inv;
        }
        return new float[][]{centroidRe, centroidIm};
    }

    

    private String routeUnsupervised(float[] kRe, float[] kIm) {
        if (wells.isEmpty()) {
            return "well_" + nextWellId++;
        }
        if (wells.size() < 2) {
            return "well_" + nextWellId++;
        }

        
        String nearestName = null;
        float nearestDistSq = Float.MAX_VALUE;
        for (var entry : wells.entrySet()) {
            float[][] c    = entry.getValue().centroid();
            float     dSq  = rmsDistanceSq(kRe, kIm, c[0], c[1]);
            if (dSq < nearestDistSq) {
                nearestDistSq = dSq;
                nearestName = entry.getKey();
            }
        }

        List<float[][]> established = new ArrayList<>();
        for (Well w : wells.values()) {
            if (w.count >= MIN_WELL_OBS) {
                established.add(w.centroid());
            }
        }

        if (established.size() < 2) {
            return nearestName;
        }

        
        float[] nnDistsSq = new float[established.size()];
        for (int i = 0; i < established.size(); i++) {
            float best = Float.MAX_VALUE;
            for (int j2 = 0; j2 < established.size(); j2++) {
                if (i == j2) continue;
                float dSq = rmsDistanceSq(
                        established.get(i)[0], established.get(i)[1],
                        established.get(j2)[0], established.get(j2)[1]);
                if (dSq < best) best = dSq;
            }
            nnDistsSq[i] = best;
        }
        Arrays.sort(nnDistsSq);
        float medianSq;
        int len = nnDistsSq.length;
        if (len % 2 == 0) {
            medianSq = (nnDistsSq[len / 2 - 1] + nnDistsSq[len / 2]) * 0.5f;
        } else {
            medianSq = nnDistsSq[len / 2];
        }

        
        if (nearestDistSq > medianSq * 4.0f) {
            return "well_" + nextWellId++;
        }
        return nearestName;
    }

    

    
    private float rmsDistance(float[] aRe, float[] aIm, float[] bRe, float[] bIm) {
        return (float) Math.sqrt(rmsDistanceSq(aRe, aIm, bRe, bIm));
    }

    
    private float rmsDistanceSq(float[] aRe, float[] aIm, float[] bRe, float[] bIm) {
        float sumSq = 0f;
        for (int j = 0; j < pairs; j++) {
            float dRe = aRe[j] - bRe[j];
            float dIm = aIm[j] - bIm[j];
            
            sumSq = Math.fma(dRe, dRe, Math.fma(dIm, dIm, sumSq));
        }
        return sumSq * invPairs;
    }

    

    
    float computeTexture(int window, int seedCount) {
        int playerEntries = dream.size() - seedCount;
        if (playerEntries < 3) return -1f; 

        
        ensureGlobalScale();
        if (cachedGlobalScale <= MACHINE_EPS) return -1f;

        
        int skip = Math.max(seedCount, dream.size() - window);
        double sumRms = 0.0;
        int count = 0;
        int idx = 0;

        for (DreamEntry de : dream) {
            if (idx++ < skip) continue;

            
            double entrySum = 0.0;
            for (int j = 0; j < pairs && j < de.s.length; j++) {
                entrySum += de.s[j];
            }
            sumRms += entrySum / pairs;
            count++;
        }

        if (count == 0) return -1f;

        
        return (float) ((sumRms / count) / cachedGlobalScale);
    }

    

    
    DreamProfile computeDreamProfile(int seedCount) {
        int playerCount = dream.size() - seedCount;
        if (playerCount < 3) return null;

        
        double kM1 = 0, kM2 = 0;
        double gM1 = 0, gM2 = 0;
        double sM1 = 0, sM2 = 0;

        
        double rSx = 0, rSy = 0, rSxy = 0, rSxx = 0;

        
        double[] kMags = new double[playerCount];
        double lastK = 0, lastG = 0, lastS = 0;

        int idx = 0, n = 0;
        for (DreamEntry de : dream) {
            if (idx++ < seedCount) continue;

            
            double kSq = 0;
            for (int j = 0; j < pairs && j < de.kRe.length; j++) {
                kSq += (double) de.kRe[j] * de.kRe[j] + (double) de.kIm[j] * de.kIm[j];
            }
            double kMag = Math.sqrt(kSq / pairs);

            
            double gSq = 0;
            for (int j = 0; j < pairs && j < de.gRe.length; j++) {
                gSq += (double) de.gRe[j] * de.gRe[j] + (double) de.gIm[j] * de.gIm[j];
            }
            double gMag = Math.sqrt(gSq / pairs);

            
            double sSum = 0;
            for (int j = 0; j < pairs && j < de.s.length; j++) {
                sSum += de.s[j];
            }
            double sMean = sSum / pairs;

            n++;

            
            double kD = kMag - kM1; kM1 += kD / n; kM2 += kD * (kMag - kM1);
            
            double gD = gMag - gM1; gM1 += gD / n; gM2 += gD * (gMag - gM1);
            
            double sD = sMean - sM1; sM1 += sD / n; sM2 += sD * (sMean - sM1);

            
            double x = n - 1;
            rSx += x; rSy += sMean; rSxy += x * sMean; rSxx += x * x;

            kMags[n - 1] = kMag;
            lastK = kMag; lastG = gMag; lastS = sMean;
        }

        double kStd = n > 1 ? Math.sqrt(kM2 / (n - 1)) : 0;
        double gStd = n > 1 ? Math.sqrt(gM2 / (n - 1)) : 0;
        double sStd = n > 1 ? Math.sqrt(sM2 / (n - 1)) : 0;

        
        double texTrend = 0;
        if (n >= 3) {
            double denom = n * rSxx - rSx * rSx;
            if (Math.abs(denom) > MACHINE_EPS) {
                texTrend = (n * rSxy - rSx * rSy) / denom;
            }
        }

        
        double acCov = 0, acVar = 0;
        for (int i = 1; i < n; i++) {
            double d0 = kMags[i - 1] - kM1;
            double d1 = kMags[i] - kM1;
            acCov += d0 * d1;
            acVar += d0 * d0;
        }
        double rhythm = (acVar > MACHINE_EPS) ? Math.max(-1, Math.min(1, acCov / acVar)) : 0;

        return new DreamProfile(kM1, kStd, gM1, gStd, sM1, sStd, texTrend,
                rhythm, lastK, lastG, lastS, n);
    }

    

    
    boolean checkResonance(float[] currentKRe, float[] currentKIm, int seedCount) {
        int playerEntries = dream.size() - seedCount;
        if (playerEntries < 5) return false; 

        ensureGlobalScale();
        if (cachedGlobalScale <= MACHINE_EPS) return false;

        
        
        float threshold = (float) (0.10 * cachedGlobalScale);

        int idx = 0;
        for (DreamEntry de : dream) {
            if (idx++ < seedCount) continue; 

            float dist = rmsDistance(currentKRe, currentKIm, de.kRe, de.kIm);
            if (dist < threshold) {
                return true; 
            }
        }

        return false;
    }

    
    private void ensureGlobalScale() {
        if (cachedGlobalScale >= 0 && cachedGlobalScaleGeneration == wellMutationGeneration) return;

        
        
        double[] gcRe = new double[pairs];
        double[] gcIm = new double[pairs];
        for (Well w : wells.values()) {
            if (w.count == 0) continue;
            double inv = 1.0 / w.count;
            int si = 0;
            for (int j = 0; j < pairs; j++, si += 2) {
                gcRe[j] += w.sumK[si]     * inv;
                gcIm[j] += w.sumK[si + 1] * inv;
            }
        }
        double n = wells.size();
        double normSq = 0;
        for (int j = 0; j < pairs; j++) {
            double re = gcRe[j] / n;
            double im = gcIm[j] / n;
            normSq += re * re + im * im;
        }
        cachedGlobalScale = Math.max(MACHINE_EPS, Math.sqrt(normSq / pairs));
        cachedGlobalScaleGeneration = wellMutationGeneration;
    }

    
    public void clear() {
        wells.clear();
        dream.clear();
        referencePairing = null;
        nextWellId    = 0;
        totalAbsorbed = 0;
        cachedGlobalScale = -1.0;
        cachedGlobalScaleGeneration = -1;
        wellMutationGeneration = 0;
    }

    

    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    
    public byte[] toEngramBytes() {
        try {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            try (ZipOutputStream zos = new ZipOutputStream(baos)) {
                
                zos.putNextEntry(new ZipEntry("manifest.json"));
                zos.write(buildManifestJson().getBytes(StandardCharsets.UTF_8));
                zos.closeEntry();

                
                zos.putNextEntry(new ZipEntry("wells.bin"));
                zos.write(buildWellsBin());
                zos.closeEntry();

                
                zos.putNextEntry(new ZipEntry("dream.bin"));
                zos.write(buildDreamBin());
                zos.closeEntry();

                
                zos.putNextEntry(new ZipEntry("annotations.json"));
                String annJson = annotations != null ? GSON.toJson(annotations) : "{}";
                zos.write(annJson.getBytes(StandardCharsets.UTF_8));
                zos.closeEntry();
            }
            return baos.toByteArray();
        } catch (IOException e) {
            woflo.wdym.Whatdoyoumean.LOGGER.warn("[Engram] Failed to serialize brain: {}", e.getMessage());
            return null;
        }
    }

    
    public static EngramBrain fromEngramBytes(byte[] data, int expectedDim) {
        if (data == null || data.length < 4) return null;

        try {
            byte[] manifestBytes = null, wellsBytes = null, dreamBytes = null, annBytes = null;

            try (ZipInputStream zis = new ZipInputStream(new ByteArrayInputStream(data))) {
                ZipEntry entry;
                while ((entry = zis.getNextEntry()) != null) {
                    byte[] content = zis.readAllBytes();
                    switch (entry.getName()) {
                        case "manifest.json"    -> manifestBytes = content;
                        case "wells.bin"        -> wellsBytes = content;
                        case "dream.bin"        -> dreamBytes = content;
                        case "annotations.json" -> annBytes = content;
                    }
                }
            }

            if (manifestBytes == null || wellsBytes == null) {
                woflo.wdym.Whatdoyoumean.LOGGER.warn("[Engram] Missing manifest.json or wells.bin in engram file");
                return null;
            }

            
            JsonObject manifest = JsonParser.parseString(
                    new String(manifestBytes, StandardCharsets.UTF_8)).getAsJsonObject();
            JsonObject physics = manifest.getAsJsonObject("physics");
            int fileDim = physics.get("dim").getAsInt();
            int filePairs = physics.get("pairs").getAsInt();
            float fileAlpha = physics.has("alpha") ? physics.get("alpha").getAsFloat() : 0.005f;

            
            if (expectedDim > 0 && fileDim != expectedDim) {
                woflo.wdym.Whatdoyoumean.LOGGER.warn(
                        "[Engram] Dimension mismatch: file has {}d but expected {}d", fileDim, expectedDim);
                return null;
            }

            JsonObject state = manifest.getAsJsonObject("state");
            int fileTotalAbsorbed = state.get("total_absorbed").getAsInt();
            int fileNextWellId = state.get("next_well_id").getAsInt();

            String fileName = "";
            String fileAuthor = "woflo";
            String filePronouns = "they/them";
            if (manifest.has("identity")) {
                JsonObject id = manifest.getAsJsonObject("identity");
                if (id.has("name")) fileName = id.get("name").getAsString();
                if (id.has("author")) fileAuthor = id.get("author").getAsString();
                if (id.has("pronouns")) filePronouns = id.get("pronouns").getAsString();
            }

            EngramBrain brain = new EngramBrain(fileDim);
            brain.alpha = fileAlpha;
            brain.name = fileName;
            brain.author = fileAuthor;
            brain.pronouns = filePronouns;
            if (state.has("dream_capacity_override")) {
                brain.dreamCapacityOverride = state.get("dream_capacity_override").getAsInt();
            }
            brain.totalAbsorbed = fileTotalAbsorbed;
            brain.nextWellId = fileNextWellId;

            
            parseWellsBin(brain, wellsBytes, fileDim, filePairs);

            
            if (dreamBytes != null && dreamBytes.length >= 4) {
                parseDreamBin(brain, dreamBytes, filePairs);
            }

            
            if (annBytes != null) {
                try {
                    brain.annotations = JsonParser.parseString(
                            new String(annBytes, StandardCharsets.UTF_8)).getAsJsonObject();
                } catch (Exception _e) {  }
            }

            return brain;

        } catch (Exception e) {
            woflo.wdym.Whatdoyoumean.LOGGER.warn("[Engram] Failed to parse engram file: {}", e.getMessage());
            return null;
        }
    }

    

    private String buildManifestJson() {
        JsonObject root = new JsonObject();
        root.addProperty("format", "engram");
        root.addProperty("version", 4);

        JsonObject identity = new JsonObject();
        identity.addProperty("name", name);
        identity.addProperty("author", author);
        identity.addProperty("pronouns", pronouns);
        root.add("identity", identity);

        JsonObject physics = new JsonObject();
        physics.addProperty("dim", dim);
        physics.addProperty("pairs", pairs);
        physics.addProperty("alpha", alpha);
        root.add("physics", physics);

        JsonObject state = new JsonObject();
        state.addProperty("total_absorbed", totalAbsorbed);
        state.addProperty("next_well_id", nextWellId);
        state.addProperty("n_wells", wells.size());
        state.addProperty("dream_buffer_size", dream.size());
        state.addProperty("dream_buffer_capacity", dreamCapacity());
        if (dreamCapacityOverride > 0) {
            state.addProperty("dream_capacity_override", dreamCapacityOverride);
        }
        state.addProperty("has_meta", false);
        root.add("state", state);

        
        JsonObject wellsObj = new JsonObject();
        for (var e : wells.entrySet()) {
            JsonObject wo = new JsonObject();
            wo.addProperty("count", e.getValue().count);
            wellsObj.add(e.getKey(), wo);
        }
        root.add("wells", wellsObj);

        return GSON.toJson(root);
    }

    

    
    private byte[] buildWellsBin() {
        
        int headerSize = dim * 2 + 4; 
        int wellDataSize = 0;
        for (var e : wells.entrySet()) {
            int nameLen = e.getKey().getBytes(StandardCharsets.UTF_8).length;
            wellDataSize += 2 + nameLen + pairs * 16 + WELL_RESERVED_BYTES + 4;
        }
        ByteBuffer buf = ByteBuffer.allocate(headerSize + wellDataSize);
        buf.order(ByteOrder.LITTLE_ENDIAN);

        
        for (int i = 0; i < dim; i++) {
            buf.putShort((short) (referencePairing != null ? referencePairing[i] : i));
        }

        
        buf.putInt(wells.size());

        
        for (var e : wells.entrySet()) {
            byte[] nameBytes = e.getKey().getBytes(StandardCharsets.UTF_8);
            buf.putShort((short) nameBytes.length);
            buf.put(nameBytes);

            Well w = e.getValue();
            
            for (int j = 0; j < pairs; j++) {
                buf.putDouble(w.sumK[2 * j]);     
                buf.putDouble(w.sumK[2 * j + 1]); 
            }

            
            for (int r = 0; r < WELL_RESERVED_BYTES; r++) {
                buf.put((byte) 0);
            }

            
            buf.putInt(w.count);
        }

        buf.flip();
        byte[] result = new byte[buf.remaining()];
        buf.get(result);
        return result;
    }

    private static void parseWellsBin(EngramBrain brain, byte[] data, int dim, int pairs) {
        ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);

        
        brain.referencePairing = new int[dim];
        for (int i = 0; i < dim; i++) {
            brain.referencePairing[i] = Short.toUnsignedInt(buf.getShort());
        }

        int nWells = buf.getInt();

        for (int w = 0; w < nWells; w++) {
            int nameLen = Short.toUnsignedInt(buf.getShort());
            byte[] nameBytes = new byte[nameLen];
            buf.get(nameBytes);
            String wellName = new String(nameBytes, StandardCharsets.UTF_8);

            Well well = new Well(pairs);
            for (int j = 0; j < pairs; j++) {
                well.sumK[2 * j]     = buf.getDouble(); 
                well.sumK[2 * j + 1] = buf.getDouble(); 
            }

            
            buf.position(buf.position() + WELL_RESERVED_BYTES);

            well.count = buf.getInt();
            brain.wells.put(wellName, well);
        }
    }

    

    
    private byte[] buildDreamBin() {
        int entrySize = pairs * 8 + pairs * 8 + pairs * 4; 
        ByteBuffer buf = ByteBuffer.allocate(4 + dream.size() * entrySize);
        buf.order(ByteOrder.LITTLE_ENDIAN);

        buf.putInt(dream.size());

        for (DreamEntry de : dream) {
            for (int j = 0; j < pairs; j++) {
                buf.putFloat(de.kRe[j]);
                buf.putFloat(de.kIm[j]);
            }
            for (int j = 0; j < pairs; j++) {
                buf.putFloat(de.gRe[j]);
                buf.putFloat(de.gIm[j]);
            }
            for (int j = 0; j < pairs; j++) {
                buf.putFloat(de.s[j]);
            }
        }

        buf.flip();
        byte[] result = new byte[buf.remaining()];
        buf.get(result);
        return result;
    }

    private static void parseDreamBin(EngramBrain brain, byte[] data, int pairs) {
        ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        int nDream = buf.getInt();

        for (int i = 0; i < nDream; i++) {
            float[] kRe = new float[pairs], kIm = new float[pairs];
            for (int j = 0; j < pairs; j++) {
                kRe[j] = buf.getFloat();
                kIm[j] = buf.getFloat();
            }
            float[] gRe = new float[pairs], gIm = new float[pairs];
            for (int j = 0; j < pairs; j++) {
                gRe[j] = buf.getFloat();
                gIm[j] = buf.getFloat();
            }
            float[] s = new float[pairs];
            for (int j = 0; j < pairs; j++) {
                s[j] = buf.getFloat();
            }
            brain.dream.addLast(new DreamEntry(kRe, kIm, gRe, gIm, s));
        }
    }
}
