package edu.h2r.learning.modelbased.featurestate;

import burlap.behavior.singleagent.vfa.StateToFeatureVectorGenerator;
import burlap.oomdp.core.State;

/**
 * Created by philippe on 16/04/15.
 */
public class FeatureStateGenerator {
    /**
     * Feature generator that is used to convert the state into features
     */
    protected StateToFeatureVectorGenerator fg;

    public FeatureStateGenerator(StateToFeatureVectorGenerator fg) {
        this.fg = fg;
    }

    public FeatureState fromState(State s) {
        return new FeatureState(s, fg.generateFeatureVectorFrom(s));
    }
}
