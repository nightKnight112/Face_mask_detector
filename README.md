# Face_mask_detector
A computer vision application which detects if mask is worn by the subject either in images/videos or in realtime webcam feeds 

## NOTES

1. MobilenetV2 is used as the feature extractor for mask detection
2. caffemodel is used for face detection
3. While training the mask_model(i.e. mask_detector_final.model) using model_trainer.py script, an epoch number between 10-20 is preferred for best accuracy
4. Matplotlib is used to plot the training accuracy and the plotted graph is saved as plot_model_accuracy.png in the project dir
5. Code accepts 3 different feeds(either still image or video or realtime webcam feed), for video and still image we have to provide the code with the exact path of the video/image. For realtime webcam feeds, provide the camera number(i.e. 0 if using laptop inbuilt cam, 1 if only one external webcam is attached etc.)