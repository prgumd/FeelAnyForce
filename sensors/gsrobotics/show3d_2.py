import sys
import numpy as np
import cv2
import os
import gsdevice
import gs3drecon


def main(argv):
    # Set flags
    SAVE_VIDEO_FLAG = False
    FIND_ROI = False
    GPU = False
    MASK_MARKERS_FLAG = False

    # Path to 3d model
    path = '.'

    # Set the camera resolution
    mmpp = 0.0634  # mini gel 18x24mm at 240x320

    # the device ID can change after unplugging and changing the usb ports.
    # on linux run, v4l2-ctl --list-devices, in the terminal to get the device ID for camera
    try:
        dev = gsdevice.Camera("GelSight Mini R0B XXXX-XXXX")
        dev2 = gsdevice.Camera("GelSight Mini R0B XXXX-XXXX")
    except:
        print("modify your tactile sensors ids using \"v4l2-ctl --list-devices\"")
        exit(1)
    net_file_path = 'nnmini.pt'

    dev.connect()
    dev2.connect() 

    ''' Load neural network '''
    model_file_path = path
    net_path = os.path.join(model_file_path, net_file_path)
    print('net path = ', net_path)

    if GPU:
        gpuorcpu = "cuda"
    else:
        gpuorcpu = "cpu"

    nn = gs3drecon.Reconstruction3D(dev)
    nn2 = gs3drecon.Reconstruction3D(dev2)
    net = nn.load_nn(net_path, gpuorcpu)
    net2 = nn2.load_nn(net_path, gpuorcpu)

    f0 = dev.get_raw_image()
    f02 = dev2.get_raw_image()
    roi = (0, 0, f0.shape[1], f0.shape[0])
    roi2 = (0, 0, f02.shape[1], f02.shape[0])

    if SAVE_VIDEO_FLAG:
        #### Below VideoWriter object will create a frame of above defined The output is stored in 'filename.avi' file.
        file_path = './3dnnlive.mov'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(file_path, fourcc, 60, (f0.shape[1], f0.shape[0]), isColor=True)
        print(f'Saving video to {file_path}')

    if FIND_ROI:
        roi = cv2.selectROI(f0)
        roi_cropped = f0[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
        cv2.imshow('ROI', roi_cropped)
        print('Press q in ROI image to continue')
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print('roi = ', roi)
    print('roi2 = ', roi2)
    print('press q on image to exit')

    ''' use this to plot just the 3d '''
    vis3d = gs3drecon.Visualize3D(dev.imgh, dev.imgw, '', mmpp, 'Tactile 1')
    vis3d2 = gs3drecon.Visualize3D(dev2.imgh, dev2.imgw, '', mmpp, 'Tactile 2')

    try:
        while dev.while_condition and dev2.while_condition:

            # get the roi image
            f1 = dev.get_image()
            f12 = dev2.get_image()
            bigframe = cv2.resize(f1, (f1.shape[1] * 2, f1.shape[0] * 2))
            bigrame2 = cv2.resize(f12, (f12.shape[1] * 2, f12.shape[0] * 2))
            cv2.imshow('Image', bigframe)
            cv2.imshow('Image2', bigrame2)

            # compute the depth map
            dm = nn.get_depthmap(f1, MASK_MARKERS_FLAG)
            dm2 = nn2.get_depthmap(f12, MASK_MARKERS_FLAG)

            ''' Display the results '''
            vis3d.update(dm)
            vis3d2.update(dm2)
            print(np.max(dm[:-60, :]))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if SAVE_VIDEO_FLAG:
                out.write(f1)

    except KeyboardInterrupt:
        print('Interrupted!')
        dev.stop_video()


if __name__ == "__main__":
    main(sys.argv[1:])
