// SPDX-FileCopyrightText: 2026 Michael Bickford (Woflo Labs)
// SPDX-License-Identifier: LicenseRef-WLCSL-1.0
// See repository-root LICENSE.md and LICENSES/WLCSL-1.0.md.

package woflo.wdym.engram;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Pattern;


public class GloveEmbedder {

    
    private static final Pattern STRIP_PUNCT = Pattern.compile("^[^a-zA-Z0-9]++|[^a-zA-Z0-9]++$");
    
    private static final Pattern WHITESPACE = Pattern.compile("\\s++");

    private final Map<String, float[]> vocab;
    private final int dim;

    private GloveEmbedder(Map<String, float[]> vocab, int dim) {
        this.vocab = vocab;
        this.dim = dim;
    }

    
    public static GloveEmbedder load(String resourcePath) {
        try (InputStream raw = GloveEmbedder.class.getClassLoader().getResourceAsStream(resourcePath)) {
            if (raw == null) {
                woflo.wdym.Whatdoyoumean.LOGGER.warn("[Engram] GloVe binary not found at classpath resource: {}", resourcePath);
                return null;
            }

            DataInputStream in = new DataInputStream(new BufferedInputStream(raw, 1 << 16));

            
            int wordCount = Integer.reverseBytes(in.readInt());
            int dim       = Integer.reverseBytes(in.readInt());

            if (wordCount <= 0 || wordCount > 500_000 || dim <= 0 || dim > 2048) {
                woflo.wdym.Whatdoyoumean.LOGGER.error(
                        "[Engram] Invalid GloVe header: wordCount={}, dim={}", wordCount, dim);
                return null;
            }

            Map<String, float[]> vocab = HashMap.newHashMap(wordCount);
            byte[] nameBuf = new byte[256];

            for (int i = 0; i < wordCount; i++) {
                
                int nameLen = Short.toUnsignedInt(Short.reverseBytes(in.readShort()));
                if (nameLen > nameBuf.length) {
                    nameBuf = new byte[nameLen];
                }
                in.readFully(nameBuf, 0, nameLen);
                String word = new String(nameBuf, 0, nameLen, StandardCharsets.UTF_8);

                
                float[] vec = new float[dim];
                for (int d = 0; d < dim; d++) {
                    vec[d] = Float.intBitsToFloat(Integer.reverseBytes(in.readInt()));
                }
                vocab.put(word, vec);
            }

            woflo.wdym.Whatdoyoumean.LOGGER.info("[Engram] GloVe loaded: {} words @ {}d", vocab.size(), dim);
            return new GloveEmbedder(vocab, dim);

        } catch (Exception e) {
            woflo.wdym.Whatdoyoumean.LOGGER.error("[Engram] Failed to load GloVe binary: {}", e.getMessage());
            return null;
        }
    }

    
    public int getDim() {
        return dim;
    }

    
    public int getVocabSize() {
        return vocab.size();
    }

    
    public float[] embed(String text) {
        if (text == null || text.isBlank()) return null;

        float[] sum = new float[dim];
        int hits = 0;

        for (String raw : WHITESPACE.split(text)) {
            
            String token = STRIP_PUNCT.matcher(raw).replaceAll("")
                              .toLowerCase(Locale.ROOT);
            if (token.isEmpty()) continue;

            float[] vec = vocab.get(token);
            if (vec == null) continue;

            for (int d = 0; d < dim; d++) {
                sum[d] += vec[d];
            }
            hits++;
        }

        if (hits == 0) return null;

        
        
        float invHits = 1.0f / hits;
        float normSq = 0f;
        for (int d = 0; d < dim; d++) {
            float v = sum[d] * invHits;
            sum[d] = v;
            normSq += v * v;
        }
        if (normSq < 1e-20f) return null; 

        return sum;
    }
}
