package edu.h2r.learning.modelbased;

import burlap.oomdp.core.State;
import burlap.oomdp.singleagent.GroundedAction;

/**
 * Created by philippe on 23/04/15.
 */
public class Transitions{
public class Transition {
    private final State s;
    private final GroundedAction a;
    private final State sp;
    private final double r;

    public Transition(State s, GroundedAction a, State sp, double r) {
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
        if (!s.equals(that.s)) return false;
        if (!a.equals(that.a)) return false;
        return sp.equals(that.sp);

    }

    @Override
    public int hashCode() {
        int result;
        long temp;
        result = s.hashCode();
        result = 31 * result + a.hashCode();
        result = 31 * result + sp.hashCode();
        temp = Double.doubleToLongBits(r);
        result = 31 * result + (int) (temp ^ (temp >>> 32));
        return result;
    }}
}
