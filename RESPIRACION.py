#!/usr/bin/env python3

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

'''
Respiration - Caleta TEAM - June 2021

This software perform non-contact monitorization of the respiration of a person.

    Functions:
    +Performs inference on RGB camera for FACE DETECTION and retrieves depth from inferences.  
    +Also measures depth from stereo-matching using the MONO cameras within a roi relative to the face. 
    +Face anonimization is optional
    +Different parameters are shown and plot.
    +Plot scaling is performed in real-time
    +Beats per minute are measure from the respiration curve
    +Several parameters are adjusted interactively
    
    Description of the keys:
    + WASD move the ROI in x and y
    + Y Auto-adjust the y-scale of the plot
    + P Toggle on-off the plotting of the original curve before smoothing
    + C Toggle on-off anonimization of the face 
    + M Increases the amount of smoothing
    + N Decreases the amount of smoothing
    + Q Quits the program
    + I Shows some extra info on the frame
    + G Save respiratory profile
    
    
'''
labelMap = ["background", "face"]
syncNN = True

# CNN .blob route
nnBlobPath = str((Path(__file__).parent / Path('face-detection-retail-0004_openvino_2021.3_4shave.blob')).resolve().absolute())

if len(sys.argv) > 1:
    nnBlobPath = sys.argv[1]

# Start definition of a pipeline
pipeline = dai.Pipeline()

#-------------------------------------------------------------------------------
# Sources - two mono (grayscale) cameras for depth and rbg for preview.
#-------------------------------------------------------------------------------
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()

height = monoLeft.getResolutionHeight()
width = monoLeft.getResolutionWidth()

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

colorCam = pipeline.createColorCamera()
colorCam.setPreviewSize(300, 300)
colorCam.setInterleaved(False)
colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

#---------------------------------------------------------------------------
# Definition of nodes (stereoDepth and spatialDetectionNetwork) and outputs.
#---------------------------------------------------------------------------

spatialDetectionNetwork = pipeline.createMobileNetSpatialDetectionNetwork()
stereo = pipeline.createStereoDepth()
spatialLocationCalculator = pipeline.createSpatialLocationCalculator()

stereo.setOutputDepth(True)

xoutRgb = pipeline.createXLinkOut()
xoutNN = pipeline.createXLinkOut()
xoutBoundingBoxDepthMapping = pipeline.createXLinkOut()
xoutDepth = pipeline.createXLinkOut()
xoutSpatialData = pipeline.createXLinkOut()
xinSpatialCalcConfig = pipeline.createXLinkIn()

xoutRgb.setStreamName("rgb")
xoutNN.setStreamName("detections")
xoutBoundingBoxDepthMapping.setStreamName("boundingBoxDepthMapping")
xoutDepth.setStreamName("depth")
xoutSpatialData.setStreamName("spatialData")
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")


#---------------------------------------------------------------------------
# Configuration of the node spatialDetectionNetwork.
#---------------------------------------------------------------------------
spatialDetectionNetwork.setBlobPath(nnBlobPath)
spatialDetectionNetwork.setConfidenceThreshold(0.95)
spatialDetectionNetwork.setBoundingBoxScaleFactor(1.0)
spatialDetectionNetwork.setDepthLowerThreshold(500)
spatialDetectionNetwork.setDepthUpperThreshold(900)

#------------------------------------
# Connections between nodes.
#------------------------------------
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
colorCam.preview.link(spatialDetectionNetwork.input)

if(syncNN):
    spatialDetectionNetwork.passthrough.link(xoutRgb.input)
else:
    colorCam.preview.link(xoutRgb.input)

spatialDetectionNetwork.out.link(xoutNN.input)
spatialDetectionNetwork.boundingBoxMapping.link(xoutBoundingBoxDepthMapping.input)

stereo.depth.link(spatialDetectionNetwork.inputDepth)
spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

# Pipeline defined, now the device is connected to

spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
stereo.depth.link(spatialLocationCalculator.inputDepth)

#We need to be accurate, so we use a very small ROI
topLeft = dai.Point2f(0.4, 0.4)
bottomRight = dai.Point2f(0.42, 0.42)

spatialLocationCalculator.setWaitForConfigInput(False)
config = dai.SpatialLocationCalculatorConfigData()

#We measure depth in a very small range
config.depthThresholds.lowerThreshold = 600
config.depthThresholds.upperThreshold = 900

config.roi = dai.Rect(topLeft, bottomRight)
spatialLocationCalculator.initialConfig.addROI(config)
spatialLocationCalculator.out.link(xoutSpatialData.input)
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

# --------------------------------------------------------
# Definitions of parameters

# Cycles of measurement
tim = 0
# Amount of averaging for smoothing the curve
win = 8
win2 = win
rango_x = 1700
rango_y = 2
# Size of the ROI
width_roi = 20
# Position dx and dy of the ROI
dy = 0.9
dx = 1.5
stepSize = 0.01
# if 1 anonimization of the baby is performed
anonymize = 0
# if 1 extra information is displayed on the frame
info = 0
#save cyxle as txt file
save_cycle = 0
# Beats per minute
bmp = 0
plot_resp = 0
cycle = 1
adjust_y = 0
# Defining some variables: spatial locations x, y and z, and the final valued to be plotted y11 is depth and X11 is time
x, y, z, x11, y11 = [], [], [], [], []
x1, x2, y1, y2 = 0, 0, 0, 0
#------------------------------------------------------------------
# Config for plotting using OpenCV
#-----------------------------------------------------------------

fig, ax = plt.subplots(1, 1)
fig.suptitle('Non-contact monitor of respiratory rate', fontsize=18)
ax.set_xlabel('time', fontsize =16)
ax.set_ylabel('amplitude', fontsize =16)
ax.set_xlim(0, rango_x)
# Turning off tick labels
ax.set_yticklabels([])
ax.set_xticklabels([])
# Initial scale
ax.set_ylim(60, 100)

color = (255, 255, 255)
fontType = cv2.FONT_HERSHEY_TRIPLEX
# Optional to show Depth Map.
# cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)
cv2.namedWindow("Plot", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Plot", 600, 400)
#-------------------------------------------------

# Initialization of timer for x-axis
startTime0 = time.monotonic()

#-----------------------------------------------------------
# MAIN LOOP
#----------------------------------------------------------
with dai.Device(pipeline) as device:
    # Start pipeline
    device.startPipeline()

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
    xoutBoundingBoxDepthMapping = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=4, blocking=False)
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
    spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")

    frame = None
    detections = []
    # Initialization of timer for fps
    startTime = time.monotonic()
    counter = 0
    # Frames per second
    fps = 0

    while True:
        inPreview = previewQueue.get()
        inNN = detectionNNQueue.get()
        depth = depthQueue.get()
        inDepthAvg = spatialCalcQueue.get() # Blocking call, will wait until a new data has arrived

        # When the relative position of the ROI is adjusted manually newConfig is set to True (function at the end)
        newConfig = False

        counter += 1
        current_time = time.monotonic()

        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        # Get RGB and Depth frames
        frame = inPreview.getCvFrame()
        height = frame.shape[0]
        width = frame.shape[1]
        depthFrame = depth.getFrame()

        # Preprocessing and inference
        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
        detections = inNN.detections

        # if the frame is available,
        # Get depth in a ROI located at a distance relative to the detections
        if len(detections) == 1:
            spatialData = inDepthAvg.getSpatialLocations()

            for depthData in spatialData:
                roi = depthData.config.roi
                roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])

                # ROI coordinates
                # width_roi is the size of the ROI, user-defined
                # dx and dy can be change interactively using the WASD keys
                xmin = int(x1 + dx*(x2-x1)/2) - int(width_roi/2)
                ymin = int(y2 + dy*(y2-y1))
                xmax = xmin + width_roi
                ymax = ymin + width_roi

                # Section needed to catch some errors when the ROI is outside the frame. In such situations the ROI is kept inside
                if xmin > 1:
                    topLeft.x = xmin / width
                else:
                    topLeft.x = 1 / width
                    bottomRight.x = topLeft.x + width_roi

                if ymin > 1:
                    topLeft.y = ymin / height
                else:
                    topLeft.y = 1 / height
                    bottomRight.y = topLeft.y + width_roi

                if xmax < width:
                    bottomRight.x = xmax / width
                else:
                    bottomRight.x = width / width
                    topLeft.x = bottomRight.x - width_roi

                if ymax < height:
                    bottomRight.y = ymax / height
                else:
                    bottomRight.y = height / height
                    topLeft.y = bottomRight.y - width_roi


        # Main respiration algorithm
        delay= current_time - startTime0

        for detection in detections:
            # Denormalization of the bounding box coordinates
            x1 = int(detection.xmin * width)
            x2 = int(detection.xmax * width)
            y1 = int(detection.ymin * height)
            y2 = int(detection.ymax * height)

            try:
                label = labelMap[detection.label]

            except:
                label = detection.label

            # Measure depth from stereo-matching between left-right cameras and adds the value to the variable z
            zz = int(depthData.spatialCoordinates.z)/10
            # Extract depth from the area of the CNN detection (face)
            z_face = int(detection.spatialCoordinates.z/10)

            x.append(tim)
            y.append(zz)
            z.append(z_face)

        #-------------------------------------------------------------
        # Display the frame with information
        #---------------------------------------------------------------
            # ROI rectangle
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            cv2.putText(frame, f"Z: {z_face} cm", (200, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)

        # FACE rectangle
        if anonymize == 1:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1, cv2.FONT_HERSHEY_SIMPLEX)

        cv2.putText(frame, f"BPM: {int(bmp)}", (200, frame.shape[0] - 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, [0, 0, 255])

        if info == 1:
            cv2.putText(frame, f"Filter: {win2}", (2, frame.shape[0] - 40), cv2.FONT_HERSHEY_TRIPLEX, 0.4, [0, 255, 0])
            cv2.putText(frame, "dx: {:.2f} dy: {:.2f}".format(dx, dy), (200, frame.shape[0] - 40), cv2.FONT_HERSHEY_TRIPLEX, 0.4,
                        [255, 255, 0])
            cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
            cv2.putText(frame, "time: {:.0f} s".format(delay), (2, frame.shape[0] - 20), cv2.FONT_HERSHEY_TRIPLEX, 0.4, [50, 128, 255])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

        # Display the frame
        cv2.imshow("rgb", frame)
        # cv2.imshow("Depth", depthFrameColor

        #Plot the original curve, that is, before smoothing
        if len(detections) == 1 and plot_resp == 1:
            plt.plot(x, y, '-', color='grey')

        # Increase index value of array
        if len(detections) == 1:
            tim += 1

        # Every 'win' values, the average of 'win' values is taken
        if len(detections) == 1 and (tim % win) == 0:
            newConfig = True
            mean_y = sum(y[tim - win:tim]) / win
            mean_x = sum(x[tim - win:tim]) / win
            y11.append(mean_y)
            x11.append(mean_x)
            plt.plot(x11, y11, '-', color='red', linewidth=3.0)
            #resx = [x11[i] for i in range(len(x11)) if i % 2 != 0]
            #resy = [y11[i] for i in range(len(y11)) if i % 2 != 0]
            #plt.plot(resx, resy, '-', color='red', linewidth=3.0)

            # Updating the plot
            if save_cycle == 1:
                ax.set_xlabel(f"BPM: {int(bmp)}  (to save)   time: {int(delay)} s", fontsize=16)
            else:
                ax.set_xlabel(f"BPM: {int(bmp)}    time: {int(delay)} s", fontsize=16)
            ax.set_ylabel('amplitude', fontsize=16)
            ax.set_xlim(0, rango_x)
            # Turn off tick labels
            ax.set_yticklabels([])
            ax.set_xticklabels([])

            # When the software starts the first time (cycle = 1) or if Y is pressed and there are at least 15 values
            # reset and rescaling is done

            if (cycle == 1 or adjust_y == 1) and len(y11) > 15:
                cycle = 0
                tim = 0
                #reset win to the original
                #win = win2
                y11 = list(filter(None, y11))
                max_y = np.max(y11[-10:])
                min_y = np.min(y11[-10:])

                # Updating the plot
                startTime0 = time.monotonic()
                x, y, x11, y11 = [], [], [], []
                ax.clear()
                ax.set_xlim(0, rango_x)
                # Turn off tick labels
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                ax.set_ylim(min_y - rango_y, max_y + rango_y)
                if save_cycle == 1:
                    ax.set_xlabel(f"BPM: {int(bmp)}  (to save)   time: {int(delay)} s", fontsize=16)
                else:
                    ax.set_xlabel(f"BPM: {int(bmp)}    time: {int(delay)} s", fontsize=16)
                ax.set_ylabel('scaling', fontsize = 16, color='green')

            # When the measurements reach the end of the temporal range of 1 cycle, the BPM is measured
            if tim >= rango_x:
                y_array = -1 * np.array(y11)
                peak_indices, _ = find_peaks(y_array, prominence=0.3)
                peak_count = len(peak_indices)  # the number of peaks in the array
                plt.plot(win * peak_indices, -1 * y_array[peak_indices], 'v', markersize=12,  color='yellow', markeredgecolor='black',markeredgewidth=2)
                # Beats per minute
                bmp = peak_count/(delay/60)

                #Plot the minima found by the peak_find algorithm
                # Redraw the canvas
                fig.canvas.draw()
                # convert canvas to image
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                imgfinal = img.reshape(600, 800, 3)
                # img is rgb, convert to opencv's default bgr
                imgfinal = cv2.cvtColor(imgfinal, cv2.COLOR_RGB2BGR)
                cv2.imshow("Plot", imgfinal)

                xarray = np.array(x11)
                yarray = np.array(y11)
                # here is your data, in two numpy arrays
                data = np.column_stack([xarray, yarray])
                if save_cycle == 1:
                    datafile_path = "respiration_cycle.txt"
                    # For saving the curve of measurements
                    np.savetxt(datafile_path, data, fmt=['%f', '%f'])



                # A delay is introduced at the end of one cycle before resetting the plot
                key = cv2.waitKey(2000)
                tim = 0
                win = win2
                y11 = list(filter(None, y11))
                max_y = np.max(y11)
                min_y = np.min(y11)
                peak_indices = []
                startTime0 = time.monotonic()
                x, y, x11, y11 = [], [], [], []
                ax.clear()
                ax.set_xlim(0, rango_x)
                # Turn off tick labels
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                ax.set_ylim(min_y - rango_y, max_y + rango_y)

                if save_cycle == 1:
                    ax.set_xlabel(f"BPM: {int(bmp)}  (to save)   time: {int(delay)} s", fontsize=16)
                else:
                    ax.set_xlabel(f"BPM: {int(bmp)}    time: {int(delay)} s", fontsize=16)

                ax.set_ylabel('measuring', fontsize=16)


            # redraw the canvas
            fig.canvas.draw()
            # convert canvas to image
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            imgfinal = img.reshape(600, 800, 3)
            # img is rgb, convert to opencv's default bgr
            imgfinal = cv2.cvtColor(imgfinal, cv2.COLOR_RGB2BGR)
            cv2.imshow("Plot", imgfinal)


            if adjust_y == 1:
                adjust_y = 0
                cv2.waitKey(2000)

    #----------------------------------------------------------------------
    # KEYS
    #------------------------------------------------------------------------


        key = cv2.waitKey(1)

        if key == ord('q'):
            break
        elif key == ord('w'):
            if topLeft.y - stepSize > 0:
                dy -= stepSize

        elif key == ord('s'):
            if bottomRight.y + stepSize <= 1:
                dy += stepSize

        elif key == ord('a'):
            if topLeft.x - stepSize > 0:
                dx -= 2*stepSize

        elif key == ord('d'):
            if bottomRight.x + stepSize <= 1:
                dx += 2*stepSize

        elif key == ord('m'):
                win2 += 2
                adjust_y = 1


        elif key == ord('n'):
            if win - 2 >= 2:
                win2 -= 2

        elif key == ord('c') and anonymize == 1:
            anonymize = 0

        elif key == ord('c') and anonymize == 0:
            anonymize = 1

        elif key == ord('p') and plot_resp == 0:
            plot_resp = 1

        elif key == ord('p') and plot_resp == 1:
            plot_resp = 0

        elif key == ord('y'):
            adjust_y = 1

        elif key == ord('i') and info == 0:
            info = 1

        elif key == ord('i') and info == 1:
            info = 0

        elif key == ord('g') and save_cycle == 0:
            save_cycle = 1

        elif key == ord('g') and save_cycle == 1:
            save_cycle = 0

        if newConfig:
            config.roi = dai.Rect(topLeft, bottomRight)
            cfg = dai.SpatialLocationCalculatorConfig()
            cfg.addROI(config)
            spatialCalcConfigInQueue.send(cfg)

            newConfig= False
