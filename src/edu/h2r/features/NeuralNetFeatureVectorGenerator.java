package edu.h2r.features;

import burlap.behavior.singleagent.vfa.StateToFeatureVectorGenerator;
import burlap.oomdp.core.State;
import burlap.oomdp.visualizer.Visualizer;
import edu.h2r.JNet;

import java.awt.image.BufferedImage;

/**
 * Created by gabe on 4/16/15.
 */
public class NeuralNetFeatureVectorGenerator implements StateToFeatureVectorGenerator {

    private final JNet net;
    private final String layerName;
    private final StateToImageConverter imageConverter;

    public NeuralNetFeatureVectorGenerator(String modelFileName, String pretrainedFileName, String layerName, Visualizer visualizer) {
        this(modelFileName, pretrainedFileName, layerName, visualizer, BufferedImage.TYPE_BYTE_GRAY);
    }

    public NeuralNetFeatureVectorGenerator(String modelFileName, String pretrainedFileName, String layerName, Visualizer visualizer, int imageType) {
        net = new JNet(modelFileName, pretrainedFileName, 1.0f / 255.0f);
        this.layerName = layerName;
        imageConverter = new StateToImageConverter(visualizer, net.getInputWidth(), net.getInputWidth(), imageType);
    }

    @Override
    public double[] generateFeatureVectorFrom(State s) {
        float[] out = net.forwardTo(imageConverter.getStateImage(s), layerName);
        double[] dOut = new double[out.length];
        for (int i = 0; i < out.length; i++)
            dOut[i] = (double) out[i];
        return dOut;
    }

}
