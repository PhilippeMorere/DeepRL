package edu.h2r.features;

import burlap.behavior.singleagent.vfa.StateToFeatureVectorGenerator;
import burlap.oomdp.core.State;
import burlap.oomdp.visualizer.Visualizer;
import edu.h2r.JNet;

/**
 * Created by gabe on 4/16/15.
 */
public class NeuralNetFeatureVectorGenerator extends StateToImageConverter implements StateToFeatureVectorGenerator {

    private final JNet net;
    private final String layerName;

    public NeuralNetFeatureVectorGenerator(String modelFileName, String pretrainedFileName, String layerName, Visualizer visualizer, int width, int height, int imageType) {
        super(visualizer, width, height, imageType);
        net = new JNet(modelFileName, pretrainedFileName, 1.0f / 255.0f);
        this.layerName = layerName;
    }

    @Override
    public double[] generateFeatureVectorFrom(State s) {
        float[] out = net.forwardTo(getStateImage(s), layerName);
        double[] dOut = new double[out.length];
        for (int i = 0; i < out.length; i++)
            dOut[i] = (double) out[i];
        return dOut;
    }

    public float[] generateFloatFeatureVectorFrom(State s) {
        return net.forwardTo(getStateImage(s), layerName);
    }
}
