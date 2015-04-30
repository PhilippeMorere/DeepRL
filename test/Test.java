import edu.h2r.jSolver;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Created by philippe on 21/04/15.
 */
public class Test {
    private static int featureNb = 20;
    private static int actionNb = 4;
    private static int batchSize = 80;
    private float[] generatedData;
    private float[] generatedLabel;
    private float[] inputData;
    private float[] labelData;

    public Test() {
        generateDataLabel();
        /*generateRandomBatch();
        System.out.println("Len: " + inputData.length + ".  batch size is: " + (inputData.length / 24));
        for (int i = 1; i <= inputData.length / 24; i++)
            System.out.println(Arrays.toString(Arrays.copyOfRange(inputData, 24 * (i - 1), 24 * i)));

        System.out.println("Len: " + labelData.length + ".  batch size is: " + (labelData.length / 20));
        for (int i = 1; i <= labelData.length / 20; i++)
            System.out.println(Arrays.toString(Arrays.copyOfRange(labelData, 20 * (i - 1), 20 * i)));
        */
    }

    public static void main(String[] args) {
        Test test = new Test();
        test.run();
    }

    public static float[] randomInput() {
        float[] input = new float[featureNb + actionNb];
        for (int i = 0; i < input.length; i++)
            input[i] = (float) Math.random();
        return input;
    }

    public void run() {
        String solverFile = "res/gridworld_solver.prototxt";
        jSolver netTF = new jSolver(solverFile);

        float[] inputData1 = Arrays.copyOfRange(this.generatedData, 0, batchSize * (featureNb + actionNb));
        float[] labelData1 = Arrays.copyOfRange(this.generatedLabel, 0, batchSize * featureNb);

        netTF.setLogLevel(1);

        netTF.getNet().setMemoryDataLayer("data", inputData1, batchSize);
        netTF.getNet().setMemoryDataLayer("label", labelData1, batchSize);
        inputData = generatedData;
        labelData = generatedLabel;
        for (int i = 1; i <= 50000; i++) {
            //generateRandomBatch();
            netTF.getNet().setMemoryDataLayer("data", inputData1, batchSize);
            netTF.getNet().setMemoryDataLayer("label", labelData1, batchSize);
            netTF.trainOneStep();
            if (i % 1000 == 0)
                netTF.setLogLevel(0);
            if (i % 1000 == 1)
                netTF.setLogLevel(1);
        }
        float[] last = netTF.getNet().forwardTo(inputData, "output");
        //System.out.println(Arrays.toString(fisrt));
        //System.out.println(Arrays.toString(last));
        for (int i = 1; i <= last.length / featureNb; i++) {
            System.out.println(">" + Arrays.toString(Arrays.copyOfRange(labelData, featureNb * (i - 1), featureNb * i)));
            System.out.println(Arrays.toString(Arrays.copyOfRange(last, featureNb * (i - 1), featureNb * i)));
        }

    }

    private void generateRandomBatch() {
        List<Integer> list = new ArrayList<Integer>();
        for (int i = 0; i < this.generatedLabel.length / (featureNb + actionNb); i++) {
            list.add(i);
        }
        Collections.shuffle(list);
        if (batchSize < list.size())
            list = list.subList(0, batchSize);

        labelData = new float[batchSize * featureNb];
        inputData = new float[batchSize * (featureNb + actionNb)];
        for (int i = 0; i < list.size(); i++) {
            int index = list.get(i);
            for (int offset = 0; offset < (featureNb + actionNb); offset++)
                inputData[i * (featureNb + actionNb) + offset] = this.generatedData[index * (featureNb + actionNb) + offset];
            for (int offset = 0; offset < featureNb; offset++)
                labelData[i * featureNb + offset] = this.generatedLabel[index * featureNb + offset];
        }
    }

    void generateDataLabel() {
        int batchSize = featureNb;
        this.generatedData = new float[actionNb * batchSize * (featureNb + actionNb)];
        this.generatedLabel = new float[actionNb * batchSize * featureNb];
        for (int b = 0; b < batchSize; b++) {
            for (int a = 0; a < actionNb; a++) {
                generatedData[(featureNb + actionNb) * (b + a * batchSize) + b] = 1;
                int nextPos = a == 0 ? b - 1 : a == 1 ? b + 1 : a == 2 ? b + featureNb / 2 : b - featureNb / 2;
                if (nextPos == -1 && a == 0)
                    nextPos = 0;
                else if (nextPos == featureNb && a == 1)
                    nextPos = featureNb - 1;
                else if (nextPos < 0)
                    nextPos += featureNb / 2;
                else if (nextPos >= featureNb)
                    nextPos -= featureNb / 2;
                generatedLabel[(featureNb) * (b + a * batchSize) + nextPos] = 1;

                generatedData[(featureNb + actionNb) * (b + a * batchSize) + featureNb + a] = 1;
            }
        }
        System.out.println(generatedData.length / (featureNb + actionNb));
    }
}
