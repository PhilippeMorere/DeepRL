package edu.h2r.learning.modelbased;

import burlap.behavior.singleagent.learning.modellearning.Model;
import burlap.behavior.singleagent.vfa.StateToFeatureVectorGenerator;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.oomdp.core.*;
import burlap.oomdp.singleagent.Action;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.singleagent.RewardFunction;
import edu.h2r.JNet;

import java.util.*;

/**
 * Created by philippe on 18/02/15.
 */
public class NNMLModel extends Model {

    protected Domain sourceDomain;
    protected DoormaxStateHashFactory hashingFactory;
    protected int termNumber = -1;
    protected Map<GroundedAction, Map<Effect, List<Prediction>>> preds;
    protected List<Prediction> emptyPredList;
    protected List<String> termNames;
    protected Map<GroundedAction, List<Condition>> failureConds;
    /**
     * The set of states marked as terminal states.
     */
    protected Set<String> terminalStates;
    /**
     * The modeled terminal funciton.
     */
    protected TerminalFunction modeledTF;

    /**
     * The modeled reward function.
     */
    protected RewardFunction modeledRF;

    /**
     * The maximum reward to be given when the transition/state is unknown
     */
    protected double rmax;

    /**
     * @param sourceDomain
     * @param initState
     * @param pfs
     * @param rmax
     */

    public NNMLModel(Domain sourceDomain, State initState, List<PropositionalFunction> pfs, double rmax) {
        this.sourceDomain = sourceDomain;
        this.rmax = rmax;
        this.hashingFactory = new DoormaxStateHashFactory();
        this.hashingFactory.addAllPFtoHash(pfs);
        this.preds = new HashMap<GroundedAction, Map<Effect, List<Prediction>>>();
        this.failureConds = new HashMap<GroundedAction, List<Condition>>();
        this.termNames = new ArrayList<String>();
        this.emptyPredList = new ArrayList<Prediction>();
        this.terminalStates = new HashSet<String>();
        for (PropositionalFunction pf : pfs) {
            List<GroundedProp> gps = pf.getAllGroundedPropsForState(initState);
            for (GroundedProp gp : gps)
                this.termNames.add(0, gp.toString());
        }

        this.modeledTF = new TerminalFunction() {

            public boolean isTerminal(State s) {
                String sh = ((DoormaxStateHashFactory.BooleanStateHashTuple) hashingFactory.hashState(s)).hashCodeStr();
                return terminalStates.contains(sh);
            }
        };


        this.modeledRF = new RewardFunction() {

            public double reward(State s, GroundedAction a, State sprime) {
                String sh = ((DoormaxStateHashFactory.BooleanStateHashTuple) hashingFactory.hashState(s)).hashCodeStr();
                if (sh.isEmpty())
                    return NNMLModel.this.rmax;

                if (!preds.containsKey(a))
                    return NNMLModel.this.rmax;

                Condition c = new Condition(sh);
                Map<Effect, List<Prediction>> actPreds = preds.get(a);
                for (Map.Entry<Effect, List<Prediction>> entry : actPreds.entrySet())
                    for (Prediction pred : entry.getValue())
                        if (pred.cond.matches(c))
                            return pred.getReward();


                //System.out.println("notFound: " + c + ", " + a);
                return NNMLModel.this.rmax;
            }
        };
    }

    public static void main(String[] args) {

        JNet toto = new JNet("", "", 0.3f);

        GridWorldDomain gwdg = new GridWorldDomain(11, 11);
        gwdg.setMapToFourRooms();

        Domain d = gwdg.generateDomain();


        //set up the initial state of the tasks
        State s = gwdg.getOneAgentOneLocationState(d);
        gwdg.setAgent(s, 0, 0);
        gwdg.setLocation(s, 0, 10, 10);

        NNMLModel model = new NNMLModel(d, s, d.getPropFunctions(), 0);


        // Try it out
        List<Action> actions = d.getActions();
        List<GroundedAction> gas = new ArrayList<GroundedAction>();
        for (Action a : actions)
            gas.add(new GroundedAction(a, new String[]{GridWorldDomain.CLASSAGENT}));


        // North
        for (int i = 0; i < 1000; i++) {
            if (i % 100 == 0)
                System.out.println("Episode " + i);
            GroundedAction ga = gas.get((int) (Math.random() * gas.size()));
            State sp = ga.executeIn(s);
            model.updateModel(s, ga, sp, -0.1, false);
            s = sp;
        }
        model.displayFailureConditions();
        model.displayTermNames();
        model.displayPredictions();
    }

    @Override
    public RewardFunction getModelRF() {
        return this.modeledRF;
    }

    @Override
    public TerminalFunction getModelTF() {
        return this.modeledTF;
    }

    @Override
    public boolean transitionIsModeled(State s, GroundedAction ga) {
        boolean result = !getTransitionProbabilities(s, ga).isEmpty();
        return result;
    }

    @Override
    public boolean stateTransitionsAreModeled(State s) {
        for (GroundedAction ga : Action.getAllApplicableGroundedActionsFromActionList(sourceDomain.getActions(), s))
            if (!transitionIsModeled(s, ga))
                return false;
        return true;
    }

    @Override
    public List<AbstractGroundedAction> getUnmodeledActionsForState(State s) {
        List<AbstractGroundedAction> notKnown = new ArrayList<AbstractGroundedAction>();
        for (GroundedAction ga : Action.getAllApplicableGroundedActionsFromActionList(sourceDomain.getActions(), s))
            if (!transitionIsModeled(s, ga))
                notKnown.add(ga);
        return notKnown;
    }

    @Override
    public State sampleModelHelper(State s, GroundedAction ga) {
        return this.sampleTransitionFromTransitionProbabilities(s, ga);
    }

    @Override
    public List<TransitionProbability> getTransitionProbabilities(State s, GroundedAction ga) {
        List<TransitionProbability> probs = new ArrayList<TransitionProbability>();
        if (!preds.containsKey(ga))
            return probs;

        boolean found = false;
        State newState = s.copy();
        String sh = ((DoormaxStateHashFactory.BooleanStateHashTuple) hashingFactory.hashState(s)).hashCodeStr();
        Condition c = new Condition(sh);
        Map<Effect, List<Prediction>> actPreds = preds.get(ga);
        for (Map.Entry<Effect, List<Prediction>> entry : actPreds.entrySet()) {
            for (Prediction pred : entry.getValue())
                if (pred.cond.matches(c)) {
                    found = true;
                    entry.getKey().applyToState(newState);
                    break;
                }
        }

        if (found)
            probs.add(new TransitionProbability(newState, 1));
        return probs;
    }

    @Override
    public void updateModel(State s, GroundedAction ga, State sprime, double r, boolean sprimeIsTerminal) {
        StateToFeatureVectorGenerator stfvg;
        //stfvg.generateFeatureVectorFrom(s);
        String sh = ((DoormaxStateHashFactory.BooleanStateHashTuple) hashingFactory.hashState(s)).hashCodeStr();
        String shp = ((DoormaxStateHashFactory.BooleanStateHashTuple) hashingFactory.hashState(sprime)).hashCodeStr();

        if (sprimeIsTerminal)
            this.terminalStates.add(shp);

        //JNet toto;
    }

    private List<Prediction> getPredictionsFor(GroundedAction ga, Effect eff) {
        if (!preds.containsKey(ga))
            return emptyPredList;
        Map<Effect, List<Prediction>> actPreds = preds.get(ga);

        if (!actPreds.containsKey(eff))
            return emptyPredList;
        return actPreds.get(eff);
    }

    private void addNewPrediction(Prediction newPred) {
        if (!preds.containsKey(newPred.act))
            preds.put(newPred.act, new HashMap<Effect, List<Prediction>>());
        Map<Effect, List<Prediction>> actPreds = preds.get(newPred.act);

        if (!actPreds.containsKey(newPred.eff))
            actPreds.put(newPred.eff, new ArrayList<Prediction>());
        List<Prediction> effActPreds = actPreds.get(newPred.eff);
        effActPreds.add(newPred);
    }

    protected void displayTermNames() {
        System.out.println("\nTerm names:");
        for (int i = 0; i < termNames.size(); i++)
            System.out.println(i + ") " + termNames.get(i));
    }

    protected void displayFailureConditions() {
        System.out.println("\nFailure conditions:");
        int count = 0;
        for (Map.Entry<GroundedAction, Map<Effect, List<Prediction>>> entry : preds.entrySet()) {
            for (Map.Entry<Effect, List<Prediction>> entry2 : entry.getValue().entrySet()) {
                if (entry2.getKey().type != EffectTypes.NOEFFECT)
                    continue;
                System.out.println(entry.getKey() + ", " + entry2.getKey() + ":");
                for (Prediction p : entry2.getValue()) {
                    System.out.println("\t" + p.cond);
                    count++;
                }
            }
        }

        System.out.println("Total: " + count);
    }


    public void saveModelRules(String path) {
        List<Prediction> predList = new ArrayList<Prediction>();
        for (Map.Entry<GroundedAction, Map<Effect, List<Prediction>>> entry : preds.entrySet())
            for (Map.Entry<Effect, List<Prediction>> entry2 : entry.getValue().entrySet()) {
                predList.addAll(entry2.getValue());
            }

        DoormaxPredictionParser.rulesToFile(predList, path);
    }

    public void loadModelRules(Domain domain, String filePath) {
        List<Prediction> predList = DoormaxPredictionParser.rulesFromFile(domain, new Prediction(), filePath);
        for (Prediction pred : predList) {
            if (!preds.containsKey(pred.act))
                preds.put(pred.act, new HashMap<Effect, List<Prediction>>());
            Map<Effect, List<Prediction>> actPreds = preds.get(pred.act);
            if (!actPreds.containsKey(pred.eff))
                actPreds.put(pred.eff, new ArrayList<Prediction>());
            List<Prediction> effPred = actPreds.get(pred.eff);
            effPred.add(pred);
        }

    }

    protected void displayPredictions() {
        System.out.println("\nPredictions:");
        int count = 0;
        for (Map.Entry<GroundedAction, Map<Effect, List<Prediction>>> entry : preds.entrySet()) {
            for (Map.Entry<Effect, List<Prediction>> entry2 : entry.getValue().entrySet()) {
                if (entry2.getKey().type == EffectTypes.NOEFFECT)
                    continue;
                System.out.println(entry.getKey() + ", " + entry2.getKey() + ":");
                for (Prediction p : entry2.getValue()) {
                    System.out.println("\t" + p.cond);
                    count++;
                }
            }
        }

        System.out.println("Total: " + count);
    }

    @Override
    public void resetModel() {
        preds.clear();

    }

    public Prediction loadNewPrediction(Condition cond, GroundedAction ga, Effect eff, double cumulatedReward, int nbTries) {
        Prediction pred = new Prediction(cond, ga, eff);
        pred.nbTries = nbTries;
        pred.cumulatedReward = cumulatedReward;
        return pred;
    }

    protected enum EffectTypes {
        TOOGLE, INC, DEC, NOEFFECT
    }

    protected class Prediction {
        protected Condition cond;
        protected GroundedAction act;
        protected Effect eff;
        protected double cumulatedReward;
        protected int nbTries;

        public Prediction() {
            this.cond = new Condition(new int[1]);
            this.eff = new Effect(EffectTypes.DEC, "", "");
        }

        public Prediction(Condition cond, GroundedAction act, Effect eff) {
            this.act = act;
            this.cond = cond;
            this.eff = eff;
            this.cumulatedReward = 0;
            this.nbTries = 0;
        }

        public void groupWith(Prediction p) {
            cond.groupWith(p.cond);
            this.cumulatedReward += p.cumulatedReward;
            this.nbTries += p.nbTries;
        }

        @Override
        public String toString() {
            String strAct = act.toString();
            if (strAct.length() < 11)
                strAct += " ";
            return "Prediction{" +
                    "cond=" + cond +
                    ", act=" + strAct +
                    ", eff=" + eff +
                    '}';
        }

        public void addReward(double r) {
            this.cumulatedReward += r;
            this.nbTries++;
        }

        public double getReward() {
            if (nbTries == 0)
                throw new RuntimeException("BOOM");
            return this.cumulatedReward / this.nbTries;
        }

        public Prediction copy() {
            Prediction cpy = new Prediction();
            if (act != null)
                cpy.act = (GroundedAction) act.copy();
            if (cond != null)
                cpy.cond = cond.copy();
            if (eff != null)
                cpy.eff = eff.copy();
            cpy.nbTries = nbTries;
            cpy.cumulatedReward = cumulatedReward;
            return cpy;
        }
    }

    protected class Condition {
        /* List of terms of the condition.
        The value is 1 for true, 0 for false, -1 for "it can be true or false"  */
        int[] terms;

        public Condition(String hashStr) {
            if (NNMLModel.this.termNumber <= 0)
                NNMLModel.this.termNumber = hashStr.length();
            terms = new int[NNMLModel.this.termNumber];
            for (int i = 0; i < NNMLModel.this.termNumber; i++)
                terms[i] = hashStr.charAt(i) == '1' ? 1 : 0;
        }

        public Condition(int[] terms) {
            this.terms = new int[terms.length];
            for (int i = 0; i < terms.length; i++)
                this.terms[i] = terms[i];
        }

        public boolean matches(Condition cond) {
            // returns true if <this> matches <cond>
            for (int i = 0; i < NNMLModel.this.termNumber; i++)
                if (terms[i] != cond.terms[i] && terms[i] != -1)
                    return false;

            return true;
        }

        // Exact binary distance to the other condition.
        public int dist(Condition c) {
            int dist = 0;
            for (int i = 0; i < NNMLModel.this.termNumber; i++)
                if (terms[i] != c.terms[i])
                    dist++;
            return dist;
        }

        // Replaces differences between the 2 conditions by -1 (*).
        public void groupWith(Condition c) {
            for (int i = 0; i < NNMLModel.this.termNumber; i++)
                if (terms[i] != c.terms[i])
                    terms[i] = -1;
        }

        @Override
        public String toString() {
            String arrayStr = "[";
            for (int i = 0; i < terms.length; i++) {
                if (i != 0)
                    arrayStr += ',';
                arrayStr += terms[i] == -1 ? "*" : terms[i];
            }
            return "Condition{" +
                    "terms=" + arrayStr +
                    "]}";
        }

        public Condition copy() {
            return new Condition(terms);
        }
    }

    protected class Effects {
        protected List<Effect> effects;

        public Effects(State s, State sprime) {
            effects = new ArrayList<Effect>();
            for (ObjectInstance obj : s.getAllObjects()) {
                List<Value> vs = obj.getValues();
                List<Value> vps = sprime.getObject(obj.getName()).getValues();

                for (int i = 0; i < vps.size(); i++)
                    compareTwoValues(vs.get(i), vps.get(i), obj.getName());
            }
            if (effects.isEmpty())
                effects.add(new Effect(EffectTypes.NOEFFECT, "noeffect", "noeffect"));
        }

        private void compareTwoValues(Value v1, Value v2, String objName) {
            if (!v1.attName().equals(v2.attName()))
                throw new RuntimeException("Can't compare two different attributes");
            Attribute a1 = v1.getAttribute();
            Attribute a2 = v2.getAttribute();
            if (a1.type != a2.type)
                throw new RuntimeException("Can't compare two attributes of different type");
            switch (a1.type) {
                case INT:
                    if (v1.getDiscVal() == v2.getDiscVal())
                        break;
                    if (v1.getDiscVal() == v2.getDiscVal() - 1)
                        effects.add(new Effect(EffectTypes.INC, objName, v1.attName()));
                    else if (v1.getDiscVal() == v2.getDiscVal() + 1)
                        effects.add(new Effect(EffectTypes.DEC, objName, v1.attName()));
                    else
                        throw new RuntimeException("Can only handle increment and decrement of 1 for int values");
                    break;
                case BOOLEAN:
                    if (v1.getBooleanValue() != v2.getBooleanValue())
                        effects.add(new Effect(EffectTypes.TOOGLE, objName, v1.attName()));
                    break;
                case DISC:
                    // TODO: support this one
                    break;
                default:
                    throw new RuntimeException("Attribute " + a1.name + " of object " + objName +
                            ": Comparaison of type " + a1.type + " is not supported.");
            }
        }
    }

    protected class Effect {
        protected EffectTypes type;
        protected String objectName;
        protected String attName;

        public Effect(EffectTypes type, String objectName, String attName) {
            this.type = type;
            this.objectName = objectName;
            this.attName = attName;
        }

        @Override
        public String toString() {
            return "Effect{" +
                    "type=" + type.name() +
                    ", objectName='" + objectName + '\'' +
                    ", attName='" + attName + '\'' +
                    '}';
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            Effect effect = (Effect) o;

            if (!attName.equals(effect.attName)) return false;
            if (!objectName.equals(effect.objectName)) return false;
            if (type != effect.type) return false;

            return true;
        }

        @Override
        public int hashCode() {
            int result = type.hashCode();
            result = 31 * result + objectName.hashCode();
            result = 31 * result + attName.hashCode();
            return result;
        }

        public State applyToState(State s) {
            ObjectInstance obj = s.getObject(objectName);
            switch (type) {
                case TOOGLE:
                    obj.setValue(attName, !obj.getBooleanValue(attName));
                    break;
                case INC:
                    obj.setValue(attName, obj.getDiscValForAttribute(attName) + 1);
                    break;
                case DEC:
                    obj.setValue(attName, obj.getDiscValForAttribute(attName) - 1);
                    break;
                case NOEFFECT:
                    break;
            }
            return s;
        }

        public Effect copy() {
            return new Effect(type, objectName, attName);
        }
    }
}
