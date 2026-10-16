package dev.woflo.engram.lab;

import com.google.gson.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents;
import net.fabricmc.fabric.api.gametest.v1.FabricGameTest;
import net.minecraft.core.BlockPos;
import net.minecraft.gametest.framework.GameTest;
import net.minecraft.gametest.framework.GameTestHelper;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.entity.EntityType;
import net.minecraft.world.entity.MoverType;
import net.minecraft.world.entity.npc.Villager;
import net.minecraft.world.level.GameRules;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.phys.Vec3;

/** Development-only lane: native world queries and movement, no policy shim. */
public final class ExperienceLab implements ModInitializer, FabricGameTest {
    private static final Gson JSON = new GsonBuilder().create();
    private static final Map<String,int[]> DIRECTIONS = new TreeMap<>();
    private static final Set<String> SEEN = new HashSet<>();
    private static final Set<String> CELLS = Set.of("0,0","1,0","2,0","3,0","3,-1","3,-2","3,1","3,2");
    private static Path lane;
    private static String session;
    private static ServerLevel world;
    private static Villager npc;
    private static BlockPos origin = new BlockPos(0,100,0);
    private static boolean complete;
    private static int resets, moves, acknowledgements;
    static {
        DIRECTIONS.put("east",new int[]{1,0}); DIRECTIONS.put("west",new int[]{-1,0});
        DIRECTIONS.put("north",new int[]{0,-1}); DIRECTIONS.put("south",new int[]{0,1});
    }

    @Override public void onInitialize() {
        String directory=System.getProperty("engram.lane");
        if(directory==null)return;
        lane=Path.of(directory);session=System.getProperty("engram.session");
        if(session==null || !session.matches("[a-f0-9]{32}"))throw new IllegalArgumentException("unique lane session required");
        ServerTickEvents.END_SERVER_TICK.register(server -> {
            if(world==null || complete)return;
            Path request=lane.resolve("request.json");
            if(!Files.isRegularFile(request))return;
            String id=null;
            try {
                JsonObject input=JsonParser.parseString(Files.readString(request)).getAsJsonObject();
                id=input.get("id").getAsString();
                if(!"engram.minecraft-request/v1".equals(input.get("schema").getAsString()) || !id.matches("[a-f0-9]{32}") || !session.equals(input.get("session").getAsString()) || !SEEN.add(id))
                    throw new IllegalArgumentException("invalid or reused request identity");
                // Consume before acknowledging, so the next request cannot be
                // removed as cleanup for the preceding transaction.
                Files.delete(request);
                String operation=input.get("op").getAsString();
                JsonObject result;
                switch(operation) {
                    case "reset" -> {
                        int cue=input.get("cue").getAsInt();
                        if(cue!=1 && cue!=-1)throw new IllegalArgumentException("cue must be -1 or +1");
                        int offset=input.has("offset")?input.get("offset").getAsInt():0;
                        origin=new BlockPos(offset,100,0);
                        reset(cue,input.has("reversed_goal") && input.get("reversed_goal").getAsBoolean());
                        result=observe();result.addProperty("reward",0);result.addProperty("terminal",false);
                    }
                    case "observe" -> { requireNpc();result=observe(); }
                    case "step" -> { requireNpc();result=step(input.get("action").getAsString()); }
                    case "finish" -> {
                        result=new JsonObject();result.addProperty("resets",resets);result.addProperty("moves",moves);
                        complete=true;write(lane.resolve("complete.json"),result);
                    }
                    default -> throw new IllegalArgumentException("unknown lane operation");
                }
                result.addProperty("schema","engram.minecraft-response/v1");
                result.addProperty("id",id);result.addProperty("session",session);result.addProperty("op",operation);
                result.addProperty("game_tick",world.getGameTime());
                result.addProperty("minecraft","1.21.1");
                write(lane.resolve(id+".json"),result);
                Files.writeString(lane.resolve("native.jsonl"),JSON.toJson(result)+"\n",StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE,StandardOpenOption.APPEND);
                acknowledgements++;
            } catch(Exception error) {
                try {
                    JsonObject result=new JsonObject();result.addProperty("error",error.toString());
                    result.addProperty("id",id);result.addProperty("session",session);
                    Files.deleteIfExists(request);write(lane.resolve(id==null?"error.json":id+".json"),result);
                }catch(Exception fatal){throw new RuntimeException(fatal);}
            }
        });
    }

    @GameTest(template="engram_npc:empty",timeoutTicks=1000000,skyAccess=true)
    public void npcExperience(GameTestHelper helper) {
        world=helper.getLevel();
        world.getGameRules().getRule(GameRules.RULE_DOMOBSPAWNING).set(false,world.getServer());
        world.getGameRules().getRule(GameRules.RULE_RANDOMTICKING).set(0,world.getServer());
        world.getGameRules().getRule(GameRules.RULE_DAYLIGHT).set(false,world.getServer());
        world.getGameRules().getRule(GameRules.RULE_WEATHER_CYCLE).set(false,world.getServer());
        world.setDayTime(6000);
        JsonObject ready=new JsonObject();ready.addProperty("session",session);ready.addProperty("schema","engram.minecraft-ready/v1");
        ready.addProperty("seed",world.getSeed());
        try{write(lane.resolve("ready.json"),ready);}catch(Exception e){throw new RuntimeException(e);}
        helper.startSequence().thenWaitUntil(() -> helper.assertTrue(complete,"experience controller is running"))
            .thenExecute(() -> {
                helper.assertTrue(resets>0 && moves>0 && acknowledgements>moves,"native lane actually ran");
                helper.assertTrue(npc!=null,"a real villager was controlled");
            }).thenSucceed();
    }

    private static void requireNpc(){if(npc==null)throw new IllegalStateException("reset before observing or moving");}

    private static void reset(int cue,boolean reversed) {
        if(npc!=null)npc.discard();
        for(int x=-1;x<=4;x++)for(int z=-3;z<=3;z++) {
            world.setBlock(origin.offset(x,0,z),Blocks.STONE.defaultBlockState(),3);
            for(int y=1;y<=3;y++)world.setBlock(origin.offset(x,y,z),
                (CELLS.contains(x+","+z)?Blocks.AIR:Blocks.STONE).defaultBlockState(),3);
        }
        world.setBlock(origin.offset(-1,1,0),(cue>0?Blocks.BLUE_WOOL:Blocks.RED_WOOL).defaultBlockState(),3);
        int goal=reversed?-cue:cue;
        world.setBlock(origin.offset(3,0,goal*2),Blocks.EMERALD_BLOCK.defaultBlockState(),3);
        world.setBlock(origin.offset(3,0,-goal*2),Blocks.MAGMA_BLOCK.defaultBlockState(),3);
        npc=EntityType.VILLAGER.create(world);
        if(npc==null)throw new IllegalStateException("villager creation failed");
        npc.setNoAi(true);npc.setNoGravity(true);npc.setPos(origin.getX()+.5,origin.getY()+1,origin.getZ()+.5);
        npc.setDeltaMovement(Vec3.ZERO);world.addFreshEntity(npc);resets++;
    }

    private static boolean free(String action) {
        int[] direction=DIRECTIONS.get(action);
        BlockPos below=BlockPos.containing(npc.getX()+direction[0],npc.getY()-1,npc.getZ()+direction[1]);
        return world.noCollision(npc,npc.getBoundingBox().move(direction[0],0,direction[1]))
            && world.getBlockState(below).isSolid();
    }

    private static JsonObject observe() {
        double x=npc.getX()-origin.getX()-.5,z=npc.getZ()-origin.getZ()-.5;
        double cue=0;
        if(Math.abs(x)<.01 && Math.abs(z)<.01) {
            BlockState marker=world.getBlockState(origin.offset(-1,1,0));
            cue=marker.is(Blocks.BLUE_WOOL)?1:marker.is(Blocks.RED_WOOL)?-1:0;
        }
        JsonArray witness=new JsonArray();
        for(double v:new double[]{x/3,cue,z/2,free("north")?1:0,free("south")?1:0,
                free("west")?1:0,free("east")?1:0,npc.getHealth()/npc.getMaxHealth()})witness.add(v);
        JsonArray actions=new JsonArray();for(String action:DIRECTIONS.keySet())if(free(action))actions.add(action);
        JsonArray position=new JsonArray();position.add(x);position.add(z);
        JsonObject result=new JsonObject();result.add("witness",witness);result.add("actions",actions);result.add("position",position);
        result.addProperty("entity_uuid",npc.getUUID().toString());
        result.addProperty("floor",world.getBlockState(npc.blockPosition().below()).getBlock().toString());
        return result;
    }

    private static JsonObject step(String action) {
        if(!DIRECTIONS.containsKey(action))throw new IllegalArgumentException("unknown action");
        Vec3 before=npc.position();int[] direction=DIRECTIONS.get(action);
        npc.move(MoverType.SELF,new Vec3(direction[0],0,direction[1]));npc.setDeltaMovement(Vec3.ZERO);moves++;
        double distance=npc.position().distanceTo(before);
        BlockState floor=world.getBlockState(npc.blockPosition().below());
        boolean terminal=floor.is(Blocks.EMERALD_BLOCK)||floor.is(Blocks.MAGMA_BLOCK);
        double reward=terminal?(floor.is(Blocks.EMERALD_BLOCK)?1:-1):distance<.5?-.1:-.02;
        if(floor.is(Blocks.MAGMA_BLOCK))npc.hurt(npc.damageSources().hotFloor(),4f);
        JsonObject result=observe();result.addProperty("reward",reward);result.addProperty("terminal",terminal);
        result.addProperty("native_move_distance",distance);
        return result;
    }

    private static void write(Path target,JsonObject value) throws Exception {
        Files.createDirectories(target.getParent());
        Path temporary=target.resolveSibling(target.getFileName()+".tmp");
        Files.writeString(temporary,JSON.toJson(value),StandardCharsets.UTF_8);
        Files.move(temporary,target,StandardCopyOption.REPLACE_EXISTING,StandardCopyOption.ATOMIC_MOVE);
    }
}
