package edu.h2r.learning.modelbased;

import burlap.behavior.singleagent.learning.modellearning.Model;
import burlap.behavior.statehashing.DiscreteStateHashFactory;
import burlap.behavior.statehashing.StateHashTuple;
import burlap.oomdp.core.*;
import burlap.oomdp.singleagent.Action;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.singleagent.RewardFunction;
import edu.h2r.jSolver;
import edu.h2r.learning.modelbased.FeatureStateGenerator.FeatureState;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Created by philippe on 18/02/15.
 */
public class DeepNNModel extends Model {

    protected Domain sourceDomain;
    protected List<Action> allActions;
    protected DiscreteStateHashFactory hashingFactory;
    /**
     * The set of states marked as terminal states.
     */
    protected Set<StateHashTuple> terminalStates;
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
     * The neural network used to model the transition function
     */
    protected jSolver netTF;

    /**
     * Name of the last layer (or output layer) of the transition function neural network
     */
    private String outputTFLayerName;

    /**
     * @param sourceDomain
     * @param initState
     * @param rmax
     */

    public DeepNNModel(Domain sourceDomain, State initState, String solverFile, String pretrainedParamFile, float scalingFactor, double rmax) {
        this.sourceDomain = sourceDomain;
        this.allActions = sourceDomain.getActions();
        this.rmax = rmax;
        this.hashingFactory = new DiscreteStateHashFactory();
        this.terminalStates = new HashSet<StateHashTuple>();

        // Init the net
        netTF = new jSolver(solverFile);

        // Defining the terminal and the reward functions
        this.modeledTF = new TerminalFunction() {

            public boolean isTerminal(State s) {
                return terminalStates.contains(hashingFactory.hashState(s));
            }
        };

        this.modeledRF = new RewardFunction() {

            public double reward(State s, GroundedAction a, State sprime) {
                StateHashTuple sh = hashingFactory.hashState(s);
                // TODO: Have a 2nd neural network to model the reward function.
                return DeepNNModel.this.rmax;
            }
        };
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
        // Run a forward pass through the net to predict the next state
        float[] netInput = netInputFromStateAction(((FeatureState) s).features, ga);
        float[] netOutput = netTF.getNet().forwardTo(netInput, outputTFLayerName);

        // Construct the new state from the old one and the new features
        FeatureState newState = (FeatureState) s.copy();
        newState.features = netOutput;

        // Return a list of transition probabilities containing only the predicted new state
        List<TransitionProbability> probs = new ArrayList<TransitionProbability>();
        probs.add(new TransitionProbability(newState, 1));
        return probs;
    }

    private float[] netInputFromStateAction(float[] stateFeatures, GroundedAction ga) {
        float[] netInput = new float[stateFeatures.length + allActions.size()];

        // Copy the features into the new array
        for (int i = 0; i < stateFeatures.length; i++)
            netInput[i] = stateFeatures[i];

        // At the end of the new array, add a 1 for the selected action and 0s for the other ones
        for (int i = 0; i < allActions.size(); i++)
            netInput[i + stateFeatures.length] = ga.actionName().equals(allActions.get(i)) ? 1 : 0;
        return netInput;
    }

    @Override
    public void updateModel(State s, GroundedAction ga, State sprime, double r, boolean sprimeIsTerminal) {
        // If the state is terminal, add it to the terminal state list
        if (sprimeIsTerminal && !modeledTF.isTerminal(s))
            this.terminalStates.add(this.hashingFactory.hashState(s));

        // TODO: Keep all experiences in memory to train the net with more data: experienceReplay

        //TODO: give the data to the net
        netTF.train();
    }

    @Override
    public void resetModel() {
        netTF.reset();
    }
}
