package internal

import (
	"fmt"
	"io"
	"os"
	"testing"
)

func TestNewRetinaFaceDetection(t *testing.T) {
	rfd := NewRetinaFaceDetection()

	f, err := os.Open("../test_data/harrison.jpeg")
	if err != nil {
		fmt.Println(err)
		return
	}
	defer f.Close()

	content, err := io.ReadAll(f)
	if err != nil {
		fmt.Println(err)
		return
	}

	res, err := ConvertImageToOpenCV(content)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res)

	rfd.preprocessOpenCVImg(res)
}
