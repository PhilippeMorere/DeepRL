package edu.h2r.learning.modelbased;

import burlap.behavior.singleagent.vfa.StateToFeatureVectorGenerator;
import burlap.oomdp.core.State;

import java.util.Arrays;

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

        public FeatureState(State s, float[] features) {
            super(s);
            this.features = new float[features.length];
            for (int i = 0; i < features.length; i++)
                this.features[i] = features[i];
        }

        public State copy() {
            return new FeatureState(this, this.features);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            if (!super.equals(o)) return false;

            FeatureState that = (FeatureState) o;

            return Arrays.equals(features, that.features);

        }

        @Override
        public int hashCode() {
            return Arrays.hashCode(features);
        }
    }
}
