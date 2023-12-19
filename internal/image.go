package internal

import (
	"errors"
	"fmt"
	"github.com/okieraised/gotritron/opencv"
)

// ConvertImageToOpenCV converts the raw image into OpenCV Matrix
func ConvertImageToOpenCV(bImage []byte) (*opencv.Mat, error) {
	dstMat := opencv.Mat{}
	srcMat, err := opencv.IMDecode(bImage, opencv.IMReadUnchanged)
	if err != nil {
		return &opencv.Mat{}, err
	}

	// Add the rows, columns, and number of channel to the dimension
	dimension := []int{}
	dimension = append(dimension, srcMat.Size()...)
	dimension = append(dimension, srcMat.Channels())

	if len(dimension) < 3 {
		return &dstMat, errors.New(fmt.Sprintf("invalid number of dimension: %d", len(dimension)))
	}

	if dimension[2] == 4 { // RGBA
		opencv.CvtColor(srcMat, &dstMat, opencv.ColorBGRAToBGR)
	} else if dimension[2] == 1 { // Grayscale
		opencv.CvtColor(srcMat, &dstMat, opencv.ColorGrayToBGR)
	} else {
		dstMat = srcMat
	}
	return &dstMat, nil
}
