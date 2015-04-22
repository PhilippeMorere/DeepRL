package edu.h2r.learning.modelbased;

import burlap.behavior.singleagent.learning.modellearning.Model;
import burlap.behavior.statehashing.StateHashFactory;
import burlap.behavior.statehashing.StateHashTuple;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.oomdp.core.*;
import burlap.oomdp.singleagent.Action;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.singleagent.RewardFunction;
import edu.h2r.jSolver;

import java.util.*;

/**
 * Created by philippe on 18/02/15.
 */
public class DeepNNModel extends Model {

    /**
     * Names of the output/data/label layers of the transition function neural network
     */
    private static final String outputTFLayerName = "output";
    private static final String dataTFLayerName = "data";
    private static final String labelTFLayerName = "label";
    protected Domain sourceDomain;
    protected List<Action> allActions;
    protected StateHashFactory hashingFactory;
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
     * @param sourceDomain
     * @param solverFile
     * @param rmax
     */

    public DeepNNModel(Domain sourceDomain, String solverFile, int featureNumber, StateHashFactory hashingFactory, double rmax) {
        this.sourceDomain = sourceDomain;
        this.allActions = sourceDomain.getActions();
        this.rmax = rmax;
        this.hashingFactory = hashingFactory;
        this.terminalStates = new HashSet<StateHashTuple>();

        // Edit the net file
        // TODO: Modify the net file so that the net size is <featureNumber>

        // Init the net
        netTF = new jSolver(solverFile);
        float[] inputOnes = new float[featureNumber + allActions.size()];
        for (int i = 0; i < featureNumber + allActions.size(); i++)
            inputOnes[i] = 1;
        float[] labelOnes = new float[featureNumber];
        for (int i = 0; i < featureNumber; i++)
            labelOnes[i] = 1;
        netTF.getNet().setMemoryDataLayer(dataTFLayerName, inputOnes);
        netTF.getNet().setMemoryDataLayer(labelTFLayerName, labelOnes);
        float[] fisrt = netTF.getNet().forwardTo(inputOnes, "output");
        System.out.println(Arrays.toString(fisrt));

        // Defining the terminal and the reward functions
        this.modeledTF = new TerminalFunction() {

            public boolean isTerminal(State s) {
                return terminalStates.contains(DeepNNModel.this.hashingFactory.hashState(s));
            }
        };

        this.modeledRF = new RewardFunction() {

            public double reward(State s, GroundedAction a, State sprime) {
                StateHashTuple sh = DeepNNModel.this.hashingFactory.hashState(s);
                // TODO: Have a 2nd neural network to model the reward function.
                return DeepNNModel.this.rmax;
            }
        };
    }

    public static void main(String[] args) {
        GridWorldDomain skbdg = new GridWorldDomain(11, 11);
        skbdg.setMapToFourRooms();

        Domain d = skbdg.generateDomain();
        FeatureStateGenerator fsg = new FeatureStateGenerator(new MockGWStateToFeatureVectorGenerator(d));
        State s = fsg.fromState(skbdg.getOneAgentOneLocationState(d));

        StateHashFactory hashingFactory = new FeatureStateHashFactory();

        DeepNNModel model = new DeepNNModel(d, "res/gridworld_solver.prototxt", fsg.fromState(s).features.length, hashingFactory, 10);

        // Try it out
        List<Action> actions = d.getActions();
        List<GroundedAction> gas = new ArrayList<GroundedAction>();
        for (Action a : actions)
            gas.add(new GroundedAction(a, new String[]{GridWorldDomain.CLASSAGENT}));


        // North
        for (int i = 0; i < 100; i++) {
            System.out.println("Episode " + i);
            GroundedAction ga = gas.get((int) (Math.random() * gas.size()));
            State sp = fsg.fromState(ga.executeIn(s));
            model.updateModel(s, ga, sp, -0.1, false);
            s = sp;
        }
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
        float[] netInput = netInputFromStateAction(s, ga);
        float[] netOutput = netTF.getNet().forwardTo(netInput, outputTFLayerName);

        //System.out.println(Arrays.toString(netOutput));
        // Construct the new state from the old one and the new features
        FeatureState newState = (FeatureState) s.copy();
        newState.features = featuresFromNetOutput(netOutput);

        // Return a list of transition probabilities containing only the predicted new state
        List<TransitionProbability> probs = new ArrayList<TransitionProbability>();
        probs.add(new TransitionProbability(newState, 1));
        return probs;
    }

    private float[] featuresFromNetOutput(float[] netOutput) {
        float[] features = new float[netOutput.length];
        for (int i = 0; i < netOutput.length; i++)
            features[i] = Math.round(netOutput[i]);
        return features;
    }

    private float[] netInputFromStateAction(State s, GroundedAction ga) {
        float[] stateFeatures = ((FeatureState) s).features;
        float[] netInput = new float[stateFeatures.length + allActions.size()];

        // Copy the features into the new array
        for (int i = 0; i < stateFeatures.length; i++)
            netInput[i] = stateFeatures[i];

        // At the end of the new array, add a 1 for the selected action and 0s for the other ones
        for (int i = 0; i < allActions.size(); i++)
            netInput[i + stateFeatures.length] = ga.actionName().equals(allActions.get(i).getName()) ? 1 : 0.5f;
        return netInput;
    }

    @Override
    public void updateModel(State s, GroundedAction ga, State sprime, double r, boolean sprimeIsTerminal) {
        // If the state is terminal, add it to the terminal state list
        if (sprimeIsTerminal && !modeledTF.isTerminal(s))
            this.terminalStates.add(this.hashingFactory.hashState(s));

        // TODO: Keep all experiences in memory to train the net with more data: experienceReplay

        // Set the net's input data and label
        netTF.getNet().setMemoryDataLayer("data", netInputFromStateAction(s, ga));
        netTF.getNet().setMemoryDataLayer("label", ((FeatureState) sprime).features);
        //System.out.println("Data: " + Arrays.toString(netInputFromStateAction(s, ga)));
        //System.out.println("Label: " + Arrays.toString(((FeatureState) sprime).features));

        // Run forward & backward pass to train the net
        netTF.trainOneStep();
    }

    @Override
    public void resetModel() {
        netTF.reset();
    }
}
