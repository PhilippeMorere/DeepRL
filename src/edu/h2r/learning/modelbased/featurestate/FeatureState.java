package edu.h2r.learning.modelbased.featurestate;

import burlap.oomdp.core.State;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by philippe on 22/04/15.
 */
public class FeatureState extends State {
    public float[] features;


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

        FeatureState that = (FeatureState) o;
        return Arrays.equals(features, that.features);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(features);
    }
}