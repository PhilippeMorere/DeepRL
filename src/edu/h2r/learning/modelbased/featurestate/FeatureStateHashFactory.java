package edu.h2r.learning.modelbased.featurestate;

import burlap.behavior.statehashing.StateHashFactory;
import burlap.behavior.statehashing.StateHashTuple;
import burlap.oomdp.core.State;

/**
 * Created by philippe on 22/04/15.
 */
public class FeatureStateHashFactory implements StateHashFactory {
    @Override
    public StateHashTuple hashState(State s) {
        return new FeatureStateHashTuple(s);
    }

    public class FeatureStateHashTuple extends StateHashTuple {

        /**
         * Initializes the StateHashTuple with the given {@link State} object.
         *
         * @param s the state object this object will wrap
         */
        public FeatureStateHashTuple(State s) {
            super(s);
        }

        @Override
        public void computeHashCode() {
            FeatureState fs = (FeatureState) this.s;
            hashCode = fs.hashCode();
        }
    }
}
