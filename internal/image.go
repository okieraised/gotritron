package internal

import "github.com/okieraised/gotritron/opencv"

func ConvertImageToOpenCV(bImage []byte) (*opencv.Mat, error) {
	dstMat := opencv.Mat{}
	imgMat, err := opencv.IMDecode(bImage, opencv.IMReadUnchanged)
	if err != nil {
		return &opencv.Mat{}, err
	}
	if imgMat.Size()[2] == 4 {
		opencv.CvtColor(imgMat, &dstMat, opencv.ColorBGRAToBGR)
	} else if len(imgMat.Size()) == 2 {
		opencv.CvtColor(imgMat, &dstMat, opencv.ColorGrayToBGR)
	}

	return &dstMat, nil
}

// def byte_data_to_opencv(im_bytes):
//    img_as_np = np.frombuffer(im_bytes, dtype=np.uint8)
//    opencv_img = cv2.imdecode(img_as_np, flags=cv2.IMREAD_UNCHANGED)
//    if opencv_img.shape[2] == 4:
//        opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_RGBA2RGB)
//    elif len(opencv_img.shape) == 2:
//        opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_GRAY2RGB)
//    return opencv_img
