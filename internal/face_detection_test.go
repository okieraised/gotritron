package internal

import (
	"fmt"
	"io"
	"os"
	"testing"
)

func TestNewRetinaFaceDetection(t *testing.T) {
	rfd, err := NewRetinaFaceDetection()
	if err != nil {
		fmt.Println(err)
		return
	}

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

	_, err = rfd.Forward(res)
	if err != nil {
		fmt.Println(err)
		return
	}

	//fmt.Println(processedImg)
	//fmt.Println(scale)
}

func TestUtil(t *testing.T) {
	x := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	fmt.Println(x[:4])

	x2 := [][][]int{
		[][]int{
			[]int{
				1, 2, 3,
			},
		},
		[][]int{
			[]int{
				4, 5, 6,
			},
		},
		[][]int{[]int{
			7, 8, 9,
		},
		},
	}
	fmt.Println(x2[:2][:2][:2])
}
