// SPDX-FileCopyrightText: 2026 Michael Bickford (Woflo Labs)
// SPDX-License-Identifier: LicenseRef-WLCSL-1.0
// See repository-root LICENSE.md and LICENSES/WLCSL-1.0.md.

package woflo.wdym.engram;

import woflo.wdym.core.SessionContext;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;


public class EngramService {

    private static final EngramService INSTANCE = new EngramService();

    
    private static final String GLOVE_RESOURCE = "engram/glove_20k_300d.bin";
    private static final String SEED_RESOURCE  = "engram/seed.engram";
    private static final String SESSION_DIR    = "the-doctor-sessions";

    
    private static final int MIN_TRAJ_FOR_FIT = 2;

    
    private volatile GloveEmbedder embedder;     
    private volatile EngramBrain   brain;        
    private volatile EngramBrain   seedBrain;    
    private static final int       MAX_ROLLING_WINDOW = 100;
    private final List<float[]>    measureWindow   = new ArrayList<>();
    private final List<float[]>    unabsorbedQueue = new ArrayList<>();

    
    private final AtomicLong epoch = new AtomicLong(0);

    private volatile boolean initialized     = false;
    private volatile String  currentSessionId = "default";

    
    private volatile String lastKnownFocus = null;

    
    private static final int TREND_SIZE = 8;
    private final float[] driftRing  = new float[TREND_SIZE];
    private int driftHead  = 0;
    private int driftCount = 0;

    
    private volatile EngramBrain.MeasureResult cachedMeasure;
    private volatile long cachedMeasureEpoch = -1;

    
    private volatile int seedDreamCount = 0;

    
    private static final int SENSE_UPDATE_INTERVAL = 3;
    private volatile float cachedTexture = -1f;
    private volatile long cachedSenseEpoch = -1;

    private volatile EngramBrain.DreamProfile cachedProfile;
    private volatile boolean cachedResonance = false;

    private EngramService() {}

    public static EngramService getInstance() { return INSTANCE; }
    public long    getEpoch()   { return epoch.get(); }
    public boolean isEnabled()  { return embedder != null; }

    
    public String getBrainName() {
        if (brain != null) return brain.getName();
        var defaults = EngramBrain.SeedDefaults.get();
        return defaults.has("name") ? defaults.get("name").getAsString() : "The Doctor";
    }

    
    public com.google.gson.JsonObject getAnnotations() {
        return brain != null ? brain.getAnnotations() : new com.google.gson.JsonObject();
    }

    
    public String getAnnotation(String key) {
        return brain != null ? brain.getAnnotation(key) : null;
    }

    
    public void setAnnotation(String key, String value) {
        if (brain != null) brain.setAnnotation(key, value);
    }

    
    public void setBrainName(String name) {
        if (brain != null) brain.setName(name);
    }

    
    public float getAlpha() {
        return brain != null ? brain.getAlpha() : 0.005f;
    }

    
    public void setAlpha(float alpha) {
        if (brain != null) brain.setAlpha(alpha);
    }

    
    public String getAuthor() {
        return brain != null ? brain.getAuthor() : "woflo";
    }

    
    public void setAuthor(String author) {
        if (brain != null) brain.setAuthor(author);
    }

    
    public String getPronouns() {
        return brain != null ? brain.getPronouns() : "they/them";
    }

    
    public void setPronouns(String pronouns) {
        if (brain != null) brain.setPronouns(pronouns);
    }

    
    public int getDreamCapacity() {
        return brain != null ? brain.getDreamCapacity() : 1200;
    }

    
    public int getDefaultDreamCapacity() {
        return brain != null ? brain.getDefaultDreamCapacity() : 1200;
    }

    
    public void setDreamCapacity(int cap) {
        if (brain != null) brain.setDreamCapacity(cap);
    }

    
    public void requestSave() {
        saveEngramToDisk();
    }

    
    public com.google.gson.JsonObject getSeedDefaults() {
        return EngramBrain.SeedDefaults.get();
    }

    

    
    public void onMessage(String content, SessionContext.MessageType type) {
        if (type != SessionContext.MessageType.USER
                && type != SessionContext.MessageType.AI_RESPONSE) {
            return;
        }
        ensureInit();
        if (embedder == null) return;

        float[] vec = embedder.embed(content);
        if (vec == null) return;

        synchronized (measureWindow) {
            measureWindow.add(vec);
            if (measureWindow.size() > MAX_ROLLING_WINDOW) {
                measureWindow.remove(0);
            }
            unabsorbedQueue.add(vec);
        }
        cachedMeasure = null;
        epoch.incrementAndGet();
    }

    
    public void onSave() {
        if (embedder == null || brain == null) return;

        List<float[]> snap;
        synchronized (measureWindow) {
            if (unabsorbedQueue.size() < MIN_TRAJ_FOR_FIT) return;
            snap = new ArrayList<>(unabsorbedQueue);
        }

        
        
        
        try {
            synchronized (brain) {
                brain.absorb(snap);
            }
            synchronized (measureWindow) {
                
                
                int consumed = snap.size();
                if (unabsorbedQueue.size() >= consumed) {
                    unabsorbedQueue.subList(0, consumed).clear();
                } else {
                    unabsorbedQueue.clear();
                }
            }
            
            if (lastKnownFocus != null) {
                brain.setAnnotation("last_focus", lastKnownFocus);
            }
            saveEngramToDisk();
        } catch (Exception e) {
            woflo.wdym.Whatdoyoumean.LOGGER.warn("[Engram] Absorb failed, vectors preserved for retry: {}",
                    e.getMessage());
        }
    }

    
    public void onSessionSwitch(String newSessionId) {
        if (embedder == null) return;
        absorbAndSave();

        
        synchronized (measureWindow) {
            if (!unabsorbedQueue.isEmpty()) {
                woflo.wdym.Whatdoyoumean.LOGGER.debug(
                        "[Engram] Dropping {} sub-threshold vectors on session switch (need {} for fit)",
                        unabsorbedQueue.size(), MIN_TRAJ_FOR_FIT);
            }
        }

        currentSessionId = newSessionId;
        loadEngramFromDisk();

        synchronized (measureWindow) {
            measureWindow.clear();
            unabsorbedQueue.clear();
        }
        synchronized (this) {
            driftCount = 0;
            driftHead  = 0;
        }
        cachedMeasure = null;
        cachedProfile = null;
        cachedResonance = false;
        epoch.incrementAndGet();
        try { woflo.wdym.core.context.ContextBuilder.invalidateCache(); } catch (Exception _e) {}
    }

    
    public void onPurge() {
        synchronized (measureWindow) {
            measureWindow.clear();
            unabsorbedQueue.clear();
        }
        synchronized (this) {
            driftCount = 0;
            driftHead  = 0;
        }

        
        if (seedBrain != null) {
            brain = seedBrain.copy();
            seedDreamCount = brain.getDreamSize(); 
            woflo.wdym.Whatdoyoumean.LOGGER.info("[Engram] Purged — reset to seed brain ({} wells)",
                    brain.getWellCount());
        } else if (brain != null) {
            brain.clear();
        }

        cachedMeasure = null;
        cachedProfile = null;
        cachedResonance = false;
        epoch.incrementAndGet();

        try {
            Path file = getEngramFile();
            if (Files.deleteIfExists(file)) {
                woflo.wdym.Whatdoyoumean.LOGGER.info("[Engram] Purged engram file: {}", file.getFileName());
            }
        } catch (Exception e) {
            woflo.wdym.Whatdoyoumean.LOGGER.warn("[Engram] Failed to delete engram file: {}", e.getMessage());
        }
    }

    
    public void shutdown() {
        absorbAndSave();
    }

    

    public record EngramState(
            boolean warmingUp,
            float drift,
            String nearestWell,
            float nearestDist,
            Map<String, Float> wellProfile,
            float[] driftTrend,
            int wellCount,
            int totalAbsorbed,
            float texture,       
            String lastFocus,    
            String shiftedFrom,  
            EngramBrain.DreamProfile profile,  
            boolean resonance    
    ) {}

    public EngramState getState() {
        if (embedder == null || brain == null) return null;

        List<float[]> snap;
        synchronized (measureWindow) {
            snap = new ArrayList<>(measureWindow);
        }

        if (snap.size() < MIN_TRAJ_FOR_FIT) {
            
            
            return new EngramState(
                    true, 50f, "unknown", 0f, Map.of(), getDriftTrend(),
                    brain.getWellCount(), brain.getTotalAbsorbed(), -1f,
                    lastKnownFocus, null, cachedProfile, false);
        }

        long currentEpoch = epoch.get();
        EngramBrain.MeasureResult cached = cachedMeasure;
        if (cached != null && cachedMeasureEpoch == currentEpoch) {
            return buildState(cached);
        }

        EngramBrain.MeasureResult result;
        synchronized (brain) {
            result = brain.measure(snap);

            
            
            
            
            if (result != null && (cachedSenseEpoch < 0
                    || currentEpoch - cachedSenseEpoch >= SENSE_UPDATE_INTERVAL)) {
                cachedTexture = brain.computeTexture(25, seedDreamCount);
                cachedProfile = brain.computeDreamProfile(seedDreamCount);
                if (result.kRe() != null) {
                    cachedResonance = brain.checkResonance(result.kRe(), result.kIm(), seedDreamCount);
                }
                cachedSenseEpoch = currentEpoch;
            }
        }
        if (result == null) {
            return new EngramState(
                    true, 50f, "unknown", 0f, Map.of(), getDriftTrend(),
                    brain.getWellCount(), brain.getTotalAbsorbed(), -1f,
                    lastKnownFocus, null, cachedProfile, false);
        }

        cachedMeasure      = result;
        cachedMeasureEpoch = currentEpoch;

        synchronized (this) {
            driftRing[driftHead % TREND_SIZE] = result.drift();
            driftHead++;
            if (driftCount < TREND_SIZE) driftCount++;
        }

        return buildState(result);
    }

    private EngramState buildState(EngramBrain.MeasureResult m) {
        
        String currentWell = m.nearestWell();
        String shifted = null;
        if (lastKnownFocus != null && !lastKnownFocus.equals(currentWell)
                && !lastKnownFocus.startsWith("well_") && !currentWell.startsWith("well_")) {
            shifted = lastKnownFocus; 
        }

        
        lastKnownFocus = currentWell;

        return new EngramState(
                false, m.drift(), currentWell, m.nearestDist(),
                m.wellProfile(), getDriftTrend(), brain.getWellCount(),
                brain.getTotalAbsorbed(), cachedTexture,
                lastKnownFocus, shifted, cachedProfile, cachedResonance);
    }

    private float[] getDriftTrend() {
        synchronized (this) {
            if (driftCount == 0) return new float[0];
            float[] out   = new float[driftCount];
            int start = (driftHead - driftCount + TREND_SIZE * 2) % TREND_SIZE;
            for (int i = 0; i < driftCount; i++) {
                out[i] = driftRing[(start + i) % TREND_SIZE];
            }
            return out;
        }
    }

    

    
    private synchronized void ensureInit() {
        if (initialized) return;
        initialized = true;

        embedder = GloveEmbedder.load(GLOVE_RESOURCE);
        if (embedder == null) {
            woflo.wdym.Whatdoyoumean.LOGGER.info(
                    "[Engram] Disabled — GloVe binary '{}' not found in classpath. "
                    + "Drop glove_20k_300d.bin into src/main/resources/engram/ to enable.", GLOVE_RESOURCE);
            return;
        }

        
        loadSeedBrain();

        currentSessionId = SessionContext.getInstance().getSessionId();
        loadEngramFromDisk();
    }

    
    private void loadSeedBrain() {
        try (InputStream is = getClass().getClassLoader().getResourceAsStream(SEED_RESOURCE)) {
            if (is == null) {
                woflo.wdym.Whatdoyoumean.LOGGER.info(
                        "[Engram] No seed engram '{}' in classpath — new sessions start blank", SEED_RESOURCE);
                return;
            }
            byte[] data = is.readAllBytes();
            seedBrain = EngramBrain.fromEngramBytes(data, embedder.getDim());
            if (seedBrain != null) {
                
                
                if (seedBrain.getDim() != embedder.getDim()) {
                    woflo.wdym.Whatdoyoumean.LOGGER.warn(
                            "[Engram] Seed dimension mismatch: seed={}d, GloVe={}d — ignoring seed",
                            seedBrain.getDim(), embedder.getDim());
                    seedBrain = null;
                } else {
                    woflo.wdym.Whatdoyoumean.LOGGER.info(
                            "[Engram] Seed brain loaded: {} wells, {} total absorbs",
                            seedBrain.getWellCount(), seedBrain.getTotalAbsorbed());
                }
            } else {
                woflo.wdym.Whatdoyoumean.LOGGER.warn("[Engram] Failed to parse seed engram");
            }
        } catch (Exception e) {
            woflo.wdym.Whatdoyoumean.LOGGER.warn("[Engram] Failed to load seed engram: {}", e.getMessage());
        }
    }

    
    private void absorbAndSave() {
        if (embedder == null || brain == null) return;

        List<float[]> snap;
        synchronized (measureWindow) {
            if (unabsorbedQueue.size() < MIN_TRAJ_FOR_FIT) return;
            snap = new ArrayList<>(unabsorbedQueue);
        }

        try {
            synchronized (brain) {
                brain.absorb(snap);
            }
            synchronized (measureWindow) {
                int consumed = snap.size();
                if (unabsorbedQueue.size() >= consumed) {
                    unabsorbedQueue.subList(0, consumed).clear();
                } else {
                    unabsorbedQueue.clear();
                }
            }
            saveEngramToDisk();
        } catch (Exception e) {
            woflo.wdym.Whatdoyoumean.LOGGER.warn("[Engram] Absorb failed during save: {}", e.getMessage());
        }
    }

    private Path getEngramFile() {
        Path configDir = woflo.wdym.config.WdymPaths.configDirectory();
        return configDir.resolve(SESSION_DIR).resolve(currentSessionId + ".engram");
    }

    private void loadEngramFromDisk() {
        if (embedder == null) return;
        Path file = getEngramFile();

        
        Path tmp = file.resolveSibling(file.getFileName() + ".tmp");
        try { Files.deleteIfExists(tmp); } catch (Exception _e) {}

        if (Files.exists(file)) {
            try {
                byte[] data = Files.readAllBytes(file);
                EngramBrain loaded = EngramBrain.fromEngramBytes(data, embedder.getDim());
                if (loaded != null) {
                    brain = loaded;
                    
                    
                    seedDreamCount = Math.max(0, brain.getDreamSize() - brain.getTotalAbsorbed());
                    lastKnownFocus = brain.getAnnotation("last_focus");
                    woflo.wdym.Whatdoyoumean.LOGGER.info(
                            "[Engram] Loaded '{}': {} wells, {} total absorbs, ~{} seed dreams, last focus: {}",
                            file.getFileName(), brain.getWellCount(), brain.getTotalAbsorbed(),
                            seedDreamCount, lastKnownFocus);
                    return;
                }
                woflo.wdym.Whatdoyoumean.LOGGER.warn(
                        "[Engram] Corrupt engram file '{}', falling back to seed",
                        file.getFileName());
            } catch (Exception e) {
                woflo.wdym.Whatdoyoumean.LOGGER.warn(
                        "[Engram] Failed to load '{}': {}", file.getFileName(), e.getMessage());
            }
        }

        
        if (seedBrain != null) {
            brain = seedBrain.copy();
            seedDreamCount = brain.getDreamSize();
            lastKnownFocus = null; 
            woflo.wdym.Whatdoyoumean.LOGGER.info(
                    "[Engram] New session '{}' — cloned seed brain ({} wells)",
                    currentSessionId, brain.getWellCount());
        } else {
            brain = new EngramBrain(embedder.getDim());
            seedDreamCount = 0;
            lastKnownFocus = null;
            woflo.wdym.Whatdoyoumean.LOGGER.debug("[Engram] Fresh brain for session '{}'", currentSessionId);
        }
    }

    
    private void saveEngramToDisk() {
        if (brain == null) return;
        Path file = getEngramFile();
        try {
            Files.createDirectories(file.getParent());
            byte[] data = brain.toEngramBytes();
            if (data == null) return;

            
            Path tmp = file.resolveSibling(file.getFileName() + ".tmp");
            Files.write(tmp, data);

            
            try {
                Files.move(tmp, file, java.nio.file.StandardCopyOption.REPLACE_EXISTING,
                        java.nio.file.StandardCopyOption.ATOMIC_MOVE);
            } catch (java.nio.file.AtomicMoveNotSupportedException _e) {
                
                Files.move(tmp, file, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            }

            woflo.wdym.Whatdoyoumean.LOGGER.debug(
                    "[Engram] Saved '{}': {} wells", file.getFileName(), brain.getWellCount());
        } catch (Exception e) {
            woflo.wdym.Whatdoyoumean.LOGGER.warn("[Engram] Failed to save '{}': {}", file.getFileName(), e.getMessage());
        }
    }
}
