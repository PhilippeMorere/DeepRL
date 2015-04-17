package edu.h2r.learning.modelfree;

import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.Policy;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.ArrowActionGlyph;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.LandmarkColorBlendInterpolation;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.PolicyGlyphPainter2D;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.StateValuePainter2D;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.lspi.LSPI;
import burlap.behavior.singleagent.learning.lspi.SARSCollector;
import burlap.behavior.singleagent.learning.lspi.SARSData;
import burlap.behavior.singleagent.learning.tdmethods.vfa.GradientDescentSarsaLam;
import burlap.behavior.singleagent.planning.QComputablePlanner;
import burlap.behavior.singleagent.planning.ValueFunctionPlanner;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.singleagent.vfa.FeatureDatabase;
import burlap.behavior.singleagent.vfa.StateToFeatureVectorGenerator;
import burlap.behavior.singleagent.vfa.common.ConcatenatedObjectFeatureVectorGenerator;
import burlap.behavior.singleagent.vfa.common.LinearVFA;
import burlap.behavior.singleagent.vfa.fourier.FourierBasis;
import burlap.behavior.statehashing.DiscreteStateHashFactory;
import burlap.behavior.statehashing.StateHashFactory;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldRewardFunction;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.oomdp.auxiliary.StateGenerator;
import burlap.oomdp.auxiliary.common.ConstantStateGenerator;
import burlap.oomdp.auxiliary.common.RandomStartStateGenerator;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.State;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.singleagent.RewardFunction;
import burlap.oomdp.singleagent.SADomain;
import burlap.oomdp.visualizer.Visualizer;
import edu.h2r.features.NeuralNetBasis;
import edu.h2r.jSolver;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.List;

/**
 * Created by gabe on 4/16/15.
 */
public class Corridor {

    private static final int CORRIDOR_WIDTH = 10;
    private static final int CORRIDOR_HEIGHT = 2;
    private static final int IMAGE_WIDTH = 80;
    private static final int IMAGE_HEIGHT = 20;
//    private static final int COLOR_CHANNELS = 1;
    private static final int IMAGE_TYPE = BufferedImage.TYPE_BYTE_GRAY;
    private static final String MODEL_FILE = "/home/gabe/deeprl-autoencoder/corridor/deploy.prototxt";
    private static final String MODEL_FILE5 = "/home/gabe/deeprl-autoencoder/corridor/deploy-5.prototxt";
    private static final String PRETRAINED_FILE = "/home/gabe/deeprl-autoencoder/corridor/models/20states-10hidden-sigmoid.caffemodel";


    private static SARSData getSARSData(Domain domain, State state, RewardFunction rf, TerminalFunction tf, int nSamples) {
        StateGenerator stateGenerator = new RandomStartStateGenerator((SADomain)domain, state);
        SARSCollector collector = new SARSCollector.UniformRandomSARSCollector(domain);
        return collector.collectNInstances(stateGenerator, rf, nSamples, 20, tf, null);
    }

    private static QComputablePlanner lspi(Domain domain, State state, RewardFunction rf, TerminalFunction tf) {
//        SARSData data = getSARSData(domain, state, rf, tf, 5000);
        StateGenerator stateGenerator = new RandomStartStateGenerator((SADomain)domain, state);
        SARSCollector collector = new SARSCollector.UniformRandomSARSCollector(domain);
        SARSData data = collector.collectNInstances(stateGenerator, rf, 5000, 20, tf, null);


        StateToFeatureVectorGenerator featureVectorGenerator = new ConcatenatedObjectFeatureVectorGenerator(true, GridWorldDomain.CLASSAGENT);
        System.out.println(Arrays.toString(featureVectorGenerator.generateFeatureVectorFrom(state)));
        FourierBasis fb = new FourierBasis(featureVectorGenerator, 4);
        LSPI lspi = new LSPI(domain, rf, tf, 0.99, fb);
        lspi.setDataset(data);
        lspi.runPolicyIteration(30, 1e-6);
        return lspi;
    }

    private static LearningAgentFactory fourierBasisLearner(final Domain domain, final RewardFunction rf, final TerminalFunction tf, StateHashFactory hashFactory) {
        return new LearningAgentFactory() {
            @Override
            public String getAgentName() {
                return "Fourier Basis";
            }

            @Override
            public LearningAgent generateAgent() {
                FeatureDatabase fd = new FourierBasis(new ConcatenatedObjectFeatureVectorGenerator(true, GridWorldDomain.CLASSAGENT), 4);
                return new GradientDescentSarsaLam(domain, rf, tf, 0.99, new LinearVFA(fd, 1.0), 0.02, 10000, 0.5);
            }
        };
    }

    private static LearningAgentFactory neuralNetLearner(final String agentName, final String pretrainedFileName,
                                                         final String modelFileName, final String layerName,
                                                         final Domain domain, final RewardFunction rf,
                                                         final TerminalFunction tf, final Visualizer visualizer,
                                                         final double lr) {
        return new LearningAgentFactory() {
            @Override
            public String getAgentName() {
                return agentName;
            }

            @Override
            public LearningAgent generateAgent() {
                FeatureDatabase fd = new NeuralNetBasis(modelFileName, pretrainedFileName, layerName, visualizer);
                return new GradientDescentSarsaLam(domain, rf, tf, 0.99, new LinearVFA(fd, 1.0), lr, 10000, 0.5);
            }
        };
    }

    private static QComputablePlanner deepLearning(Domain domain, State initialState, Visualizer visualizer, RewardFunction rf, TerminalFunction tf, StateHashFactory hashFactory) {
//        FeatureDatabase fd = new FourierBasis(new ConcatenatedObjectFeatureVectorGenerator(true, GridWorldDomain.CLASSAGENT), 4);
        NeuralNetBasis fd = new NeuralNetBasis(MODEL_FILE, PRETRAINED_FILE, "encode1neuron", visualizer);
//        LearningRate lr = new SoftTimeInverseDecayLR(0.2, 10);
        GradientDescentSarsaLam agent = new GradientDescentSarsaLam(domain, rf, tf, 0.99, new LinearVFA(fd, 1.0), 0.02, 10000, 0.5);
//        agent.setLearningRate(lr);
        for (int i = 0; i < 5000; i++) {

            EpisodeAnalysis ea = agent.runLearningEpisodeFrom(initialState);
            if (ea.numTimeSteps() > 10000) {
                System.out.println(i + ": " + ea.numTimeSteps() + " " + ea.getActionSequenceString(" ")); //print the performance of this episode
//                visualizeValueFunction(agent, new GreedyQPolicy(agent), initialState, domain, hashFactory);
            } else {
                System.out.println(i + ": " + ea.numTimeSteps()); //print the performance of this episode
            }
        }
//        System.out.println(fd.numberOfFeatures());
        return agent;
    }

    private static ValueFunctionPlanner valueIteration(Domain domain, State initialState, RewardFunction rf, TerminalFunction tf, StateHashFactory hashFactory) {
        ValueFunctionPlanner planner = new ValueIteration(domain, rf, tf, 0.99, hashFactory, 0.001, 100);
        planner.planFromState(initialState);
        return planner;
    }

    public static void visualizeValueFunction(QComputablePlanner planner, Policy p, State initialState, Domain domain, StateHashFactory hashingFactory) {

        List<State> allStates = StateReachability.getReachableStates(initialState,
                (SADomain) domain, hashingFactory);
        LandmarkColorBlendInterpolation rb = new LandmarkColorBlendInterpolation();
        rb.addNextLandMark(0., Color.RED);
        rb.addNextLandMark(1., Color.BLUE);

        StateValuePainter2D svp = new StateValuePainter2D(rb);
        svp.setValueStringRenderingFormat(8, Color.WHITE, 2, 0.0f, 0.8f);
        svp.setXYAttByObjectClass(GridWorldDomain.CLASSAGENT, GridWorldDomain.ATTX,
                GridWorldDomain.CLASSAGENT, GridWorldDomain.ATTY);

        PolicyGlyphPainter2D spp = new PolicyGlyphPainter2D();
        spp.setXYAttByObjectClass(GridWorldDomain.CLASSAGENT, GridWorldDomain.ATTX,
                GridWorldDomain.CLASSAGENT, GridWorldDomain.ATTY);
        spp.setActionNameGlyphPainter(GridWorldDomain.ACTIONNORTH, new ArrowActionGlyph(0));
        spp.setActionNameGlyphPainter(GridWorldDomain.ACTIONSOUTH, new ArrowActionGlyph(1));
        spp.setActionNameGlyphPainter(GridWorldDomain.ACTIONEAST, new ArrowActionGlyph(2));
        spp.setActionNameGlyphPainter(GridWorldDomain.ACTIONWEST, new ArrowActionGlyph(3));
        spp.setRenderStyle(PolicyGlyphPainter2D.PolicyGlyphRenderStyle.DISTSCALED);

        ValueFunctionVisualizerGUI gui = new ValueFunctionVisualizerGUI(allStates, svp, planner);
        gui.setSpp(spp);
        gui.setPolicy(p);
        gui.setBgColor(Color.GRAY);
        gui.initGUI();
    }

    public static void main(String[] args) {
//        Environment.getInstance().setExecutionMode(Kernel.EXECUTION_MODE.GPU);

        GridWorldDomain gridWorldDomain = new GridWorldDomain(CORRIDOR_WIDTH, CORRIDOR_HEIGHT);
        gridWorldDomain.makeEmptyMap();
//        gridWorldDomain.setMapToFourRooms();
        Domain domain = gridWorldDomain.generateDomain();

        // Initialization
        State state = GridWorldDomain.getOneAgentOneLocationState(domain);
        GridWorldDomain.setAgent(state, 0, 0);
        GridWorldDomain.setLocation(state, 0, CORRIDOR_WIDTH - 1, CORRIDOR_HEIGHT - 1);
        StateGenerator stateGenerator = new RandomStartStateGenerator((SADomain)domain, state);
        System.out.println(state.getCompleteStateDescription());

        TerminalFunction tf = new GridWorldTerminalFunction(CORRIDOR_WIDTH - 1, CORRIDOR_HEIGHT - 1);
//        RewardFunction rf = new UniformCostRF();
//        RewardFunction rf = new GoalBasedRF(tf, 5, -0.1);
        RewardFunction rf = new GridWorldRewardFunction(domain, -1.0);


        DiscreteStateHashFactory hashFactory = new DiscreteStateHashFactory();
        hashFactory.setAttributesForClass(GridWorldDomain.CLASSAGENT, domain.getObjectClass(GridWorldDomain.CLASSAGENT).attributeList);

        Visualizer visualizer = GridWorldVisualizer.getVisualizer(gridWorldDomain.getMap());


//        StateVisualizer stateVisualizer = new StateVisualizer(visualizer, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_TYPE, false);
//        StateImageInputProvider inputProvider = new StateImageInputProvider(getSARSData(domain, state, rf, tf, 2000), visualizer, 80, 20, IMAGE_TYPE);

//        for (int i = 0; i < 1000; i++) {
//            State currentState = stateGenerator.generateState();
//            BufferedImage image = stateVisualizer.getStateImage(currentState);
//            int locX = currentState.getFirstObjectOfClass(GridWorldDomain.CLASSAGENT).getDiscValForAttribute(GridWorldDomain.ATTX);
//            int locY = currentState.getFirstObjectOfClass(GridWorldDomain.CLASSAGENT).getDiscValForAttribute(GridWorldDomain.ATTY);
//
//            File f = new File(locY * 10 + locX + ".jpg");
//            try {
//                ImageIO.write(image, "jpg", f);
//            } catch (IOException e) {
//                e.printStackTrace();
//            }
//
//        }
        jSolver solver = new jSolver("/home/gabe/deeprl-autoencoder/corridor/solver.prototxt");
        solver.train();

        LearningAgentFactory fbLearningFactory = fourierBasisLearner(domain, rf, tf, hashFactory);
        LearningAgentFactory ae10 = neuralNetLearner("AE-10", PRETRAINED_FILE, MODEL_FILE, "encode1neuron", domain, rf, tf, visualizer, 0.002);
        LearningAgentFactory relu = neuralNetLearner("AE-10-RELU", "/home/gabe/deeprl-autoencoder/corridor/models/20states-10hidden-relu.caffemodel",
                MODEL_FILE, "encode1neuron", domain, rf, tf, visualizer, 0.002);
        LearningAgentFactory ae20 = neuralNetLearner("AE-20",
                "/home/gabe/deeprl-autoencoder/corridor/models/20states-20hidden-sigmoid.caffemodel",
                "/home/gabe/deeprl-autoencoder/corridor/deploy-20.prototxt", "encode1neuron",
                domain, rf, tf, visualizer, 0.002);
        LearningAgentFactory sae3 = neuralNetLearner("SAE",
                "/home/gabe/deeprl-autoencoder/corridor/models/20states-deep2-sigmoid-150000.caffemodel",
                "/home/gabe/deeprl-autoencoder/corridor/deploy_deep.prototxt", "encode3neuron",
                domain, rf, tf, visualizer, 0.1);
        LearningAgentFactory cae = neuralNetLearner("CAE",
                "/home/gabe/deeprl-autoencoder/cae/models/cae-20states-2fm-10x8k-10x8s.caffemodel",
                "/home/gabe/deeprl-autoencoder/cae/deploy-cae.prototxt", "conv1neuron", domain, rf, tf, visualizer, 0.2);
        LearningAgentFactory scae = neuralNetLearner("SCAE",
                "/home/gabe/deeprl-autoencoder/cae/models/scae-2states-2fm-10x8k-10x8s.caffemodel",
                "/home/gabe/deeprl-autoencoder/cae/deploy.prototxt", "ip1neuron", domain, rf, tf, visualizer, 0.0006);

        StateGenerator sg = new ConstantStateGenerator(state);
        LearningAlgorithmExperimenter experimenter = new LearningAlgorithmExperimenter((SADomain)domain, rf, sg, 10, 1000,
                fbLearningFactory, ae10, ae20, cae, scae);

        experimenter.setUpPlottingConfiguration(800, 400, 2, 1000,
                TrialMode.MOSTRECENTANDAVERAGE,
                PerformanceMetric.STEPSPEREPISODE,
                PerformanceMetric.CUMULTAIVEREWARDPEREPISODE);

        experimenter.startExperiment();

//        QComputablePlanner planner = deepLearning(domain, state, visualizer, rf, tf, hashFactory);
//        QComputablePlanner planner = lspi(domain, state, rf, tf);
//        QComputablePlanner planner = valueIteration(domain, state, rf, tf, hashFactory);
//        GreedyQPolicy p = new GreedyQPolicy(planner);

//        EpisodeAnalysis ea = p.evaluateBehavior(state, rf,tf, 50);
//        System.out.println(state.getCompleteStateDescription());
//        System.out.println(ea.getActionSequenceString(" "));

//        new EpisodeSequenceVisualizer(visualizer, domain, new GridWorldStateParser(domain), "corridor");
//        visualizeValueFunction(planner, p, state, domain, hashFactory);
//        VisualActionObserver observer = new VisualActionObserver(domain, visualizer);
//        ((SADomain)domain).setActionObserverForAllAction(observer);
//        observer.initGUI();
    }
}
