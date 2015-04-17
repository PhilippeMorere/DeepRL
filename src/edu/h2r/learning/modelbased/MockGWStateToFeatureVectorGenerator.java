package edu.h2r.learning.modelbased;

import burlap.behavior.singleagent.vfa.StateToFeatureVectorGenerator;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.oomdp.core.ObjectInstance;
import burlap.oomdp.core.State;

/**
 * Created by philippe on 16/04/15.
 */
public class MockGWStateToFeatureVectorGenerator implements StateToFeatureVectorGenerator {
    private final GridWorldDomain domain;
    private final int height, width;

    public MockGWStateToFeatureVectorGenerator(GridWorldDomain domain) {
        this.domain = domain;
        height = domain.getHeight();
        width = domain.getHeight();
    }

    @Override
    public double[] generateFeatureVectorFrom(State s) {
        double[] features = new double[height * width];
        ObjectInstance agent = s.getObjectsOfTrueClass(GridWorldDomain.CLASSAGENT).get(0);
        int pos = agent.getDiscValForAttribute(GridWorldDomain.ATTX) +
                agent.getDiscValForAttribute(GridWorldDomain.ATTY) * width;
        for (int i = 0; i < features.length; i++)
            features[i] = (i == pos) ? 1 : 0;
        return features;
    }
}
