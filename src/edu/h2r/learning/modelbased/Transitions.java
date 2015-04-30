package edu.h2r.learning.modelbased;

import burlap.oomdp.core.State;
import burlap.oomdp.singleagent.GroundedAction;
import edu.h2r.learning.modelbased.featurestate.FeatureState;

import java.util.*;

/**
 * Created by philippe on 23/04/15.
 */
public class Transitions {
    private Set<Transition> transitions = new HashSet<Transition>();

    public void addTransition(State s, GroundedAction a, State sp, double r) {
        Transition t = new Transition(((FeatureState) s).features, a, ((FeatureState) sp).features, r);
        transitions.add(t);
    }

    public int getMemorySize() {
        return transitions.size();
    }

    public DataLabel getNTransitions(int n) {
        if (n > transitions.size()) {
            System.err.println("Not enough recorded transitions. In memory: " + transitions.size() + ", requested: " + n);
            return null;
        }
        List<Integer> list = new ArrayList<Integer>();
        for (int i = 0; i < transitions.size(); i++) {
            list.add(i);
        }
        Collections.shuffle(list);
        list = list.subList(0, n);

        List<Float> data = new ArrayList<>();
        List<Float> label = new ArrayList<>();
        Integer i = 0;
        for (Transition t : transitions) {
            if (list.contains(i)) {
                float[] currData = DeepNNModel.netInputFromStateAction(t.s, t.a);
                for (int j = 0; j < currData.length; j++)
                    data.add(currData[j]);
                for (int j = 0; j < t.sp.length; j++)
                    label.add(t.sp[j]);
            }
            i++;
        }
        return new DataLabel(data.toArray(new Float[data.size()]), label.toArray(new Float[label.size()]));
    }

    private class Transition {
        private final float[] s;
        private final GroundedAction a;
        private final float[] sp;
        private final double r;

        public Transition(float[] s, GroundedAction a, float[] sp, double r) {
            this.s = s;
            this.a = a;
            this.sp = sp;
            this.r = r;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            Transition that = (Transition) o;

            if (Double.compare(that.r, r) != 0) return false;
            if (!Arrays.equals(s, that.s)) return false;
            if (!a.equals(that.a)) return false;
            return Arrays.equals(sp, that.sp);

        }

        @Override
        public int hashCode() {
            int result;
            long temp;
            result = Arrays.hashCode(s);
            result = 31 * result + a.hashCode();
            result = 31 * result + Arrays.hashCode(sp);
            temp = Double.doubleToLongBits(r);
            result = 31 * result + (int) (temp ^ (temp >>> 32));
            return result;
        }
    }

    public class DataLabel {
        public float[] data;
        public float[] label;

        public DataLabel(float[] data, float[] label) {
            this.data = data;
            this.label = label;
        }

        protected DataLabel(Float[] data, Float[] label) {
            this.data = new float[data.length];
            for (int j = 0; j < data.length; j++)
                this.data[j] = data[j];
            this.label = new float[label.length];
            for (int j = 0; j < label.length; j++)
                this.label[j] = label[j];
        }
    }
}
