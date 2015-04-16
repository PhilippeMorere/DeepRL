package edu.h2r.features;

import burlap.behavior.singleagent.vfa.ActionFeaturesQuery;
import burlap.behavior.singleagent.vfa.FeatureDatabase;
import burlap.behavior.singleagent.vfa.StateFeature;
import burlap.oomdp.core.State;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.visualizer.Visualizer;
import edu.h2r.JNet;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by gabe on 4/16/15.
 */
public class NeuralNetBasis extends StateToImageConverter implements FeatureDatabase {

    public final JNet net;
    protected final int nNodes;
    private Map<String, Integer> actionFeatureMultiplier;
    private Integer nextActionMultiplier;
    protected final String layerName;


    public NeuralNetBasis(String modelFileName, String pretrainedFileName, String layerName, Visualizer visualizer, int width, int height, int imageType) {
        super(visualizer, width, height, imageType);
        net = new JNet(modelFileName,pretrainedFileName, 1.0f / 255.0f);
        this.layerName = layerName;

        actionFeatureMultiplier = new HashMap<>();
        nextActionMultiplier = 0;

        nNodes = net.getNodeCount(layerName);

    }

    @Override
    public List<StateFeature> getStateFeatures(State s) {
        List<StateFeature> out = new ArrayList<>();
        float[] encoding = null;
        try {
            encoding = net.forwardTo(getStateImage(s), layerName);
        } catch (IllegalArgumentException e) {
            e.printStackTrace();
        }

        assert encoding != null;
        for (int i = 0; i < encoding.length; i++)
            out.add(new StateFeature(i, (double) encoding[i]));
        return out;
    }

    @Override
    public List<ActionFeaturesQuery> getActionFeaturesSets(State s, List<GroundedAction> actions) {
        List<ActionFeaturesQuery> lstAFQ = new ArrayList<>();

        Iterable<StateFeature> sfs = this.getStateFeatures(s);

        for(GroundedAction ga : actions){
            int actionMult = this.getActionMultiplier(ga);
            int indexOffset = actionMult*this.nNodes;

            ActionFeaturesQuery afq = new ActionFeaturesQuery(ga);
            for(StateFeature sf : sfs){
                afq.addFeature(new StateFeature(sf.id + indexOffset, sf.value));
            }

            lstAFQ.add(afq);

        }

        return lstAFQ;
    }

    @Override
    public void freezeDatabaseState(boolean toggle) {
        System.out.println("Freezing DB State");
    }

    @Override
    public int numberOfFeatures() {
        if(this.actionFeatureMultiplier.size() == 0){
            return this.nNodes;
        }
        return this.nNodes * this.nextActionMultiplier;
    }

    /**
     * This method returns the action multiplier for the specified grounded action.
     * If the action is not stored, a new action multiplier will created, stored, and returned.
     * If the action is parameterized a runtime exception is thrown.
     * @param ga the grounded action for which the multiplier will be returned
     * @return the action multiplier to be applied to a state feature id.
     */
    protected int getActionMultiplier(GroundedAction ga){

        if(ga.isParameterized() && ga.action.parametersAreObjects()){
            throw new RuntimeException("Fourier Basis Feature Database does not support actions with OO-MDP object parameterizations.");
        }

        Integer stored = this.actionFeatureMultiplier.get(ga.toString());
        if(stored == null){
            this.actionFeatureMultiplier.put(ga.actionName(), this.nextActionMultiplier);
            stored = this.nextActionMultiplier;
            this.nextActionMultiplier++;
        }

        return stored;
    }

//    private BufferedImage getStateImage(State s) {
//        BufferedImage image = new BufferedImage(width, height, imageType);
//        renderLayer.updateState(s);
//        renderLayer.render((Graphics2D) image.getGraphics(), width, height);
//        return image;
//    }
}
