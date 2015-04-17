package edu.h2r.learning.modelbased;

import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.Policy;
import burlap.behavior.singleagent.QValue;
import burlap.behavior.singleagent.ValueFunctionInitialization;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.modellearning.Model;
import burlap.behavior.singleagent.learning.modellearning.ModeledDomainGenerator;
import burlap.behavior.singleagent.planning.OOMDPPlanner;
import burlap.behavior.singleagent.planning.QComputablePlanner;
import burlap.behavior.singleagent.planning.ValueFunctionPlanner;
import burlap.behavior.singleagent.planning.commonpolicies.GreedyQPolicy;
import burlap.behavior.statehashing.StateHashFactory;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.oomdp.core.AbstractGroundedAction;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.State;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.singleagent.Action;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.singleagent.RewardFunction;

import java.util.LinkedList;
import java.util.List;

/**
 * Created by philippe on 19/02/15.
 */
public class DeepModelLearner extends OOMDPPlanner implements LearningAgent, QComputablePlanner {

    /**
     * The model of the world that is being learned. The model learns the transition function,
     * reward function and termination function.
     */
    protected Model model;

    /**
     * The modeled domain object, built from the learnt model. It is used by the planner to compute the action to take
     * in a specific state.
     */
    protected Domain modeledDomain;

    /**
     * The planning algorithm to use.
     */
    protected ValueFunctionPlanner planner;

    /**
     * The policy to use with the planner.
     */
    protected Policy policy;

    /**
     * The saved previous learning episodes.
     */
    protected LinkedList<EpisodeAnalysis> episodeHistory = new LinkedList<EpisodeAnalysis>();

    /**
     * The number of the most recent learning episodes to store.
     */
    protected int numEpisodesToStore = 1;

    /**
     * The feature state generator that converts classic State objects to FeatureState objects. Feature objects
     * include a feature array as well, which is used by the underlying neural net.
     */
    protected FeatureStateGenerator fsg;


    public DeepModelLearner(Domain domain, RewardFunction rf, TerminalFunction tf, double gamma, StateHashFactory hashingFactory,
                            State initState, String modelFile, String pretrainedParamFile, float scalingFactor, double rmax) {
        this.plannerInit(domain, rf, tf, gamma, hashingFactory);
        this.fsg = new FeatureStateGenerator(new MockGWStateToFeatureVectorGenerator(domain));

        // Create the model and modeled domain
        this.model = new DeepNNModel(domain, initState, modelFile, pretrainedParamFile, scalingFactor, rmax);
        ModeledDomainGenerator mdg = new ModeledDomainGenerator(domain, this.model, true);
        this.modeledDomain = mdg.generateDomain();

        // Create the planer and the policy to use with it
        this.planner = new NNMLPlanner(mdg.generateDomain(), this.model.getModelRF(), this.model.getModelTF(), gamma, hashingFactory, rmax);
        this.policy = new GreedyQPolicy(this);
    }

    /**
     * Runs one learning episode from the specified initial state. The learning episode stops when a terminal state is
     * reached.
     *
     * @param initialState the state to run the learning episode from.
     * @return an {@link burlap.behavior.singleagent.EpisodeAnalysis} which contains all steps from the learning episode.
     */
    public EpisodeAnalysis runLearningEpisodeFrom(State initialState) {
        return this.runLearningEpisodeFrom(initialState, -1);
    }

    /**
     * Runs one learning episode from the specified initial state. The learning episode stops when a terminal state is
     * reached or the number of steps taken by the agent reaches maxSteps.
     *
     * @param initialState the state to run the learning episode from.
     * @param maxSteps     the maximum number of step before the learning episode stops.
     * @return an {@link burlap.behavior.singleagent.EpisodeAnalysis} which contains all steps from the learning episode.
     */
    public EpisodeAnalysis runLearningEpisodeFrom(State initialState, int maxSteps) {
        EpisodeAnalysis ea = new EpisodeAnalysis(initialState);
        State curState = fsg.fromState(initialState);
        int steps = 0;

        // Run episode until a terminal state is reached OR the agent took too many steps
        while (!this.tf.isTerminal(curState) && (steps < maxSteps || maxSteps == -1)) {
            // Let the policy determine the best action to take in the current state
            GroundedAction ga = (GroundedAction) policy.getAction(curState);

            // Execute the action the policy determined, convert the state to a FeatureState
            State nextState = fsg.fromState(ga.executeIn(curState));

            // Get the real reward from the reward function
            double r = this.rf.reward(curState, ga, nextState);

            // Update the model and planner
            this.model.updateModel(curState, ga, nextState, r, this.tf.isTerminal(nextState));
            this.planner.performBellmanUpdateOn(curState);

            curState = nextState;
            ea.recordTransitionTo(ga, nextState, r);
            steps++;
        }

        // Add episodeAnalysis to history
        if (episodeHistory.size() >= numEpisodesToStore)
            episodeHistory.poll();
        episodeHistory.offer(ea);

        return ea;
    }

    /**
     * Returns the {@link burlap.behavior.singleagent.EpisodeAnalysis} from the last training episode.
     *
     * @return a {@link burlap.behavior.singleagent.EpisodeAnalysis}
     */
    public EpisodeAnalysis getLastLearningEpisode() {
        return episodeHistory.getLast();
    }


    /**
     * Sets the maximum number of {@link burlap.behavior.singleagent.EpisodeAnalysis} to keep in memory.
     */
    public void setNumEpisodesToStore(int numEps) {
        if (numEps > 0)
            numEpisodesToStore = numEps;
        else
            numEpisodesToStore = 1;
    }

    /**
     * Returns a {@link List} of {@link burlap.behavior.singleagent.EpisodeAnalysis} from all recorded training episodes.
     *
     * @return a {@link List} of {@link burlap.behavior.singleagent.EpisodeAnalysis}
     */
    public List<EpisodeAnalysis> getAllStoredLearningEpisodes() {
        return episodeHistory;
    }

    @Override
    public void planFromState(State initialState) {
        throw new RuntimeException("Model learning algorithms should not be used as planning algorithms.");
    }


    @Override
    public void resetPlannerResults() {
        this.model.resetModel();
        this.planner.resetPlannerResults();
        this.episodeHistory.clear();
    }

    @Override
    public List<QValue> getQs(State s) {
        List<QValue> qs = this.planner.getQs(s);
        for (QValue q : qs) {

            // If Q for unknown action, use value initialization of current state
            if (!this.model.transitionIsModeled(s, (GroundedAction) q.a)) {
                q.q = this.planner.getValueFunctionInitialization().qValue(s, q.a);
            }

            // Update action to real world action
            Action realWorldAction = this.domain.getAction(q.a.actionName());
            q.a = new GroundedAction(realWorldAction, q.a.params);
        }
        return qs;
    }

    @Override
    public QValue getQ(State s, AbstractGroundedAction a) {

        QValue q = this.planner.getQ(s, a);

        // If Q for unknown action, use value initialization of current state
        if (!this.model.transitionIsModeled(s, (GroundedAction) q.a)) {
            q.q = this.planner.getValueFunctionInitialization().qValue(s, q.a);
        }

        // Update action to real world action
        Action realWorldAction = this.domain.getAction(q.a.actionName());
        q.a = new GroundedAction(realWorldAction, q.a.params);
        return q;
    }

    protected class NNMLPlanner extends ValueFunctionPlanner {

        /**
         * Initializes
         *
         * @param domain         the modeled domain
         * @param rf             the modeled reward function
         * @param tf             the modeled terminal function
         * @param gamma          the discount factor
         * @param hashingFactory the hashing factory
         * @param vInit          the constant value function initialization to use
         */
        public NNMLPlanner(Domain domain, RewardFunction rf, TerminalFunction tf, double gamma, StateHashFactory hashingFactory, double vInit) {
            this(domain, rf, tf, gamma, hashingFactory, new ValueFunctionInitialization.ConstantValueFunctionInitialization(vInit));
        }


        /**
         * Initializes
         *
         * @param domain         the modeled domain
         * @param rf             the modeled reward function
         * @param tf             the modeled terminal function
         * @param gamma          the discount factor
         * @param hashingFactory the hashing factory
         * @param vInit          the value function initialization to use
         */
        public NNMLPlanner(Domain domain, RewardFunction rf, TerminalFunction tf, double gamma, StateHashFactory hashingFactory, ValueFunctionInitialization vInit) {
            VFPInit(domain, rf, tf, gamma, hashingFactory);

            // Don't cache transition dynamics because our learned model keeps changing!
            this.useCachedTransitions = false;

            this.valueInitializer = vInit;
        }

        @Override
        public void planFromState(State initialState) {
            throw new UnsupportedOperationException("This method should not be called for the inner ARTDP planner");
        }

    }
}
