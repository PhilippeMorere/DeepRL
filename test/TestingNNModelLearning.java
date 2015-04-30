/**
 * Created by philippe on 16/04/15.
 */

import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.planning.StateConditionTest;
import burlap.behavior.singleagent.planning.deterministic.TFGoalCondition;
import burlap.behavior.statehashing.StateHashFactory;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldStateParser;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.oomdp.auxiliary.StateParser;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.State;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.singleagent.RewardFunction;
import burlap.oomdp.singleagent.common.SinglePFTF;
import burlap.oomdp.singleagent.common.UniformCostRF;
import burlap.oomdp.visualizer.Visualizer;
import edu.h2r.learning.modelbased.DeepModelLearner;
import edu.h2r.learning.modelbased.featurestate.FeatureStateHashFactory;

public class TestingNNModelLearning {
    GridWorldDomain gwdg;
    Domain domain;
    StateParser sp;
    RewardFunction rf;
    TerminalFunction tf;
    StateConditionTest goalCondition;
    State initialState;
    StateHashFactory hashingFactory;

    public TestingNNModelLearning() {
    }

    public static void main(String[] args) {
        TestingNNModelLearning example = new TestingNNModelLearning();
        example.testOnGridWorld();
        String outputPath = "output/"; //directory to record results

        //we will call planning and learning algorithms here
        example.DeepModelLearnerExample(outputPath);

        //run the visualizer
        //example.visualizeGridWorld(outputPath);
    }

    public void testOnGridWorld() {
        //create the domain
        gwdg = new GridWorldDomain(new int[10][2]);
        domain = gwdg.generateDomain();

        //create the state parser
        sp = new GridWorldStateParser(domain);

        //define the task
        rf = new UniformCostRF();

        tf = new SinglePFTF(domain.getPropFunction(GridWorldDomain.PFATLOCATION));
        goalCondition = new TFGoalCondition(tf);

        //set up the initial state of the tasks
        initialState = gwdg.getOneAgentOneLocationState(domain);
        gwdg.setAgent(initialState, 0, 0);
        gwdg.setLocation(initialState, 0, 9, 1);

        //set up the state hashing system
        hashingFactory = new FeatureStateHashFactory();
    }

    public void DeepModelLearnerExample(String outputPath) {
        if (!outputPath.endsWith("/")) {
            outputPath = outputPath + "/";
        }
        LearningAgent agent = new DeepModelLearner(domain, rf, tf, 0.99, hashingFactory, initialState,
                "res/gridworld_solver.prototxt", 0);

        //run learning for 1000 episodes
        int maxTimeSteps = 100;
        for (int i = 0; i < 400; i++) {
            EpisodeAnalysis ea = agent.runLearningEpisodeFrom(initialState, maxTimeSteps);

            ea.writeToFile(String.format("%se%03d", outputPath, i), sp);
            System.out.println("Episode " + i + ": " + (ea.numTimeSteps() <= maxTimeSteps ? "won " : "lost") + " in " +
                    ea.numTimeSteps() + " steps.");
            if (ea.numTimeSteps() < 12)
                break;
        }
    }

    public void visualizeGridWorld(String outputPath) {
        Visualizer v = GridWorldVisualizer.getVisualizer(gwdg.getMap());
        new EpisodeSequenceVisualizer(v, domain, sp, outputPath);
    }
}
