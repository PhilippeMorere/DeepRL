package edu.h2r.learning.modelbased;

import burlap.behavior.singleagent.vfa.StateToFeatureVectorGenerator;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.oomdp.core.Attribute;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.ObjectInstance;
import burlap.oomdp.core.State;

/**
 * Created by philippe on 16/04/15.
 */
public class MockGWStateToFeatureVectorGenerator implements StateToFeatureVectorGenerator {
    private final Domain domain;
    private final Attribute x, y;
    private final int height, width;

    public MockGWStateToFeatureVectorGenerator(Domain domain) {
        this.domain = domain;
        x = domain.getAttribute(GridWorldDomain.ATTX);
        y = domain.getAttribute(GridWorldDomain.ATTY);
        height = (int) (y.upperLim - y.lowerLim);
        width = (int) (x.upperLim - x.lowerLim);
    }

    @Override
    public double[] generateFeatureVectorFrom(State s) {
        double[] features = new double[height * width];
        ObjectInstance agent = s.getObjectsOfTrueClass(GridWorldDomain.CLASSAGENT).get(0);
        int pos = agent.getDiscValForAttribute(GridWorldDomain.ATTX) - (int) (x.lowerLim) +
                (agent.getDiscValForAttribute(GridWorldDomain.ATTY) - (int) (y.lowerLim)) * width;
        for (int i = 0; i < features.length; i++)
            features[i] = (i == pos) ? 1 : 0;
        return features;
    }
}
