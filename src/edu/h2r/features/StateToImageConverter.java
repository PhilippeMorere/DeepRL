package edu.h2r.features;

import burlap.oomdp.core.State;
import burlap.oomdp.visualizer.StateRenderLayer;
import burlap.oomdp.visualizer.Visualizer;

import java.awt.*;
import java.awt.image.BufferedImage;

/**
 * Created by gabe on 4/16/15.
 */
public class StateToImageConverter {

    private final StateRenderLayer renderLayer;
    private final int width;
    private final int height;
    private final int imageType;

    public StateToImageConverter(Visualizer visualizer, int width, int height, int imageType) {
        renderLayer = visualizer.getStateRenderLayer();
        this.width = width;
        this.height = height;
        this.imageType = imageType;
    }

    protected BufferedImage getStateImage(State s) {
        BufferedImage image = new BufferedImage(width, height, imageType);
        renderLayer.updateState(s);
        renderLayer.render((Graphics2D) image.getGraphics(), width, height);
        return image;
    }
}
