package edu.h2r.learning.modelbased;

import burlap.behavior.singleagent.vfa.StateToFeatureVectorGenerator;
import burlap.oomdp.core.State;

/**
 * Created by philippe on 16/04/15.
 */
public class MockStateToFeatureVectorGenerator implements StateToFeatureVectorGenerator {
    @Override
    public double[] generateFeatureVectorFrom(State s) {
        return new double[0];
    }
}
