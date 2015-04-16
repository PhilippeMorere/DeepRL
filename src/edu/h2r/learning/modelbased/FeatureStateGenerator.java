package edu.h2r.learning.modelbased;

import burlap.behavior.singleagent.vfa.StateToFeatureVectorGenerator;
import burlap.oomdp.core.State;

/**
 * Created by philippe on 16/04/15.
 */
public class FeatureStateGenerator extends State {
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

    public class FeatureState extends State {
        protected float[] features;


        public FeatureState(State s, double[] features) {
            super(s);
            this.features = new float[features.length];
            for (int i = 0; i < features.length; i++)
                this.features[i] = (float) features[i];
        }
    }
}
