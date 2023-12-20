package internal

import (
	"encoding/binary"
	"fmt"
	"github.com/okieraised/gotritron/opencv"
	"image"
)

var (
	landmarkSTD   float32 = 1.0
	bBoxSTD               = []float32{1.0, 1.0, 1.0, 1.0}
	pixelScale    float32 = 1.0
	pixelSTD              = []float32{1.0, 1.0, 1.0}
	pixelMean             = []float32{0.0, 0.0, 0.0}
	featStrideFPN         = []int{32, 16, 8}
	ratio                 = []float64{1.0}
	denseAnchor           = false
)

type Anchor struct {
	Scale         []float64
	BaseSize      int
	Ratio         []float64
	AllowedBorder int
}

type RetinaFaceDetection struct {
	Config *RetinaFaceDetectionConfig
}

func NewRetinaFaceDetection() (*RetinaFaceDetection, error) {
	anchorConfig := map[string]Anchor{
		"32": {
			Scale:         []float64{32, 16},
			BaseSize:      16,
			Ratio:         ratio,
			AllowedBorder: 9999,
		},
		"16": {
			Scale:         []float64{8, 4},
			BaseSize:      16,
			Ratio:         ratio,
			AllowedBorder: 9999,
		},
		"8": {
			Scale:         []float64{2, 1},
			BaseSize:      16,
			Ratio:         ratio,
			AllowedBorder: 9999,
		},
	}

	var fpnKeys []string
	for _, val := range featStrideFPN {
		fpnKeys = append(fpnKeys, fmt.Sprintf("stride%d", val))
	}
	anchors, err := GenerateAnchorsFPN2(false, anchorConfig)
	if err != nil {
		return nil, err
	}

	anchorsFPN := make(map[string][][]float64)
	numAnchors := make(map[string]int)
	for idx, fpnKey := range fpnKeys {
		anchorsFPN[fpnKey] = anchors[idx]
		numAnchors[fpnKey] = len(anchors[idx])
	}
	return &RetinaFaceDetection{
		Config: DefaultRetinaFaceDetectionConfig(),
	}, nil
}

func (rfd *RetinaFaceDetection) PreprocessOpenCVImg(src *opencv.Mat) (*opencv.Mat, float64, error) {

	var newHeight, newWidth int

	imgRatio := float64(src.Rows()) / float64(src.Cols())
	modelRatio := float64(rfd.Config.ImageSize[1]) / float64(rfd.Config.ImageSize[0])

	if imgRatio > modelRatio {
		newHeight = rfd.Config.ImageSize[1]
		newWidth = int(float64(newHeight) / imgRatio)
	} else {
		newWidth = rfd.Config.ImageSize[0]
		newHeight = int(float64(newWidth) * imgRatio)
	}
	detScale := float64(newHeight) / float64(src.Rows())

	dst := opencv.Mat{}
	dst = src.Clone()
	opencv.Resize(*src, &dst, image.Pt(newWidth, newHeight), 0, 0, opencv.InterpolationArea)
	err := src.Close()
	if err != nil {
		return nil, detScale, err
	}

	return &dst, detScale, err
}

func (rfd *RetinaFaceDetection) Forward(src *opencv.Mat) ([]byte, error) {

	rawOpenCVBytes, err := src.DataPtrUint8()
	if err != nil {
		return nil, err
	}

	detImg := make([][][]float32, rfd.Config.ImageSize[1])
	for i := 0; i < rfd.Config.ImageSize[1]; i++ {
		detImg[i] = make([][]float32, rfd.Config.ImageSize[0])
		for j := 0; j < rfd.Config.ImageSize[0]; j++ {
			detImg[i][j] = make([]float32, 3)
		}
	}
	startIdx := 0
	idxDelta := 3
	for row := 0; row < src.Rows(); row++ {
		for col := 0; col < src.Cols(); col++ {
			bArr := rawOpenCVBytes[startIdx : startIdx+idxDelta]
			bF32 := make([]float32, 0, 3)
			for _, b := range bArr {
				val := float32(b)
				bF32 = append(bF32, val)
			}
			detImg[row][col] = bF32
			startIdx += idxDelta
		}
	}

	imgTensor := make([][][][]float32, 1)
	for i := 0; i < 1; i++ {
		imgTensor[i] = make([][][]float32, 3)
		for j := 0; j < 3; j++ {
			imgTensor[i][j] = make([][]float32, rfd.Config.ImageSize[1])
			for k := 0; k < rfd.Config.ImageSize[1]; k++ {
				imgTensor[i][j][k] = make([]float32, rfd.Config.ImageSize[0])
			}
		}
	}
	for idx := 0; idx < 3; idx++ {
		for i := 0; i < 1; i++ {
			for j := 0; j < 3; j++ {
				for k := 0; k < rfd.Config.ImageSize[0]; k++ {
					for l := 0; l < rfd.Config.ImageSize[1]; l++ {
						imgTensor[i][idx][k][l] = (detImg[k][l][2-i]/pixelScale - pixelMean[2-idx]) / pixelSTD[2-idx]
					}
				}
			}
		}
	}

	imgTensorFlatten := make([]float32, rfd.Config.ImageSize[0]*rfd.Config.ImageSize[0]*3*1, rfd.Config.ImageSize[0]*rfd.Config.ImageSize[0]*3*1)

	for i := 0; i < 3; i++ {
		for j := 0; j < rfd.Config.ImageSize[0]; j++ {
			for k := 0; k < rfd.Config.ImageSize[1]; k++ {
				imgTensorFlatten[k*rfd.Config.ImageSize[0]*2+j*rfd.Config.ImageSize[1]+i] = imgTensor[0][i][j][k]
			}
		}
	}

	var rawInputs []byte
	bs := make([]byte, 4)
	for i := 0; i < len(imgTensorFlatten); i++ {
		binary.LittleEndian.PutUint32(bs, uint32(imgTensorFlatten[i]))
		rawInputs = append(rawInputs, bs...)
	}

	return rawInputs, nil
}
