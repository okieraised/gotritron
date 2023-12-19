package internal

import (
	"bytes"
	"fmt"
	"github.com/okieraised/gotritron/opencv"
	"image"
	"image/draw"
	"image/jpeg"
)

var (
	landmarkSTD   float64 = 1.0
	bBoxSTD               = []float32{1.0, 1.0, 1.0, 1.0}
	pixelScale    float64 = 1.0
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

func NewRetinaFaceDetection() *RetinaFaceDetection {
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
		fmt.Println(err)
		return nil
	}

	anchorsFPN := make(map[string][][]float64)
	numAnchors := make(map[string]int)
	for idx, fpnKey := range fpnKeys {
		anchorsFPN[fpnKey] = anchors[idx]
		numAnchors[fpnKey] = len(anchors[idx])
	}

	fmt.Println(anchorsFPN)
	fmt.Println(numAnchors)

	return &RetinaFaceDetection{
		Config: DefaultRetinaFaceDetectionConfig(),
	}
}

func (rfd *RetinaFaceDetection) preprocessOpenCVImg(src *opencv.Mat) (*opencv.Mat, float64, error) {

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

	fmt.Println("detScale", detScale)
	fmt.Println("newWidth", newWidth)
	fmt.Println("newHeight", newHeight)

	resizedOpenCVImg := opencv.Mat{}
	resizedOpenCVImg = src.Clone()
	opencv.Resize(*src, &resizedOpenCVImg, image.Pt(newWidth, newHeight), 0, 0, opencv.InterpolationArea)
	err := src.Close()
	if err != nil {
		return nil, detScale, err
	}
	detImg := image.NewRGBA(image.Rect(0, 0, rfd.Config.ImageSize[0], rfd.Config.ImageSize[1]))

	resizedImg, err := resizedOpenCVImg.ToImage()
	if err != nil {
		return nil, detScale, err
	}
	draw.Draw(detImg, image.Rect(0, 0, newWidth, newHeight), resizedImg, image.Point{}, draw.Src)

	news := detImg.SubImage(resizedImg.Bounds())
	buf := new(bytes.Buffer)
	err = jpeg.Encode(buf, news, nil)
	if err != nil {
		return nil, detScale, err
	}

	detOpenCVImg, err := opencv.NewMatFromBytes(newHeight, newWidth, opencv.MatTypeCV8UC3, buf.Bytes())

	fmt.Println(detOpenCVImg.Rows(), detOpenCVImg.Cols())
	return &detOpenCVImg, detScale, nil
}

//func (self *Detectortor) postprocess(predictedOutput []interface{}, preprocessParam float64) ([]interface{}, []interface{}) {
//	det := predictedOutput[0].([]interface{})
//	kpss := predictedOutput[1].([]interface{})
//	detScale := preprocessParam
//	detArr := det.([][]float64)
//	for i := 0; i < len(detArr); i++ {
//		detArr[i][0] /= detScale
//		detArr[i][1] /= detScale
//		detArr[i][2] /= detScale
//		detArr[i][3] /= detScale
//	}
//	if kpss != nil {
//		kpssArr := kpss.([][]float64)
//		for i := 0; i < len(kpssArr); i++ {
//			kpssArr[i][0] /= detScale
//			kpssArr[i][1] /= detScale
//		}
//	}
//	return det, kpss
//}

//func landmarkPred(boxes *mat.Dense, landmarkDeltas *mat.Dense) *mat.Dense {
//	rows, _ := boxes.Dims()
//	if rows == 0 {
//		return mat.NewDense(0, landmarkDeltas.RawMatrix().Cols, nil)
//	}
//	boxesCopy := mat.DenseCopyOf(boxes)
//	boxesCopy.Apply(func(_, _ int, v float64) float64 {
//		return float64(float32(v))
//	}, boxesCopy)
//	widths := make([]float64, rows)
//	heights := make([]float64, rows)
//	ctrX := make([]float64, rows)
//	ctrY := make([]float64, rows)
//	for i := 0; i < rows; i++ {
//		widths[i] = boxesCopy.At(i, 2) - boxesCopy.At(i, 0) + 1.0
//		heights[i] = boxesCopy.At(i, 3) - boxesCopy.At(i, 1) + 1.0
//		ctrX[i] = boxesCopy.At(i, 0) + 0.5*(widths[i]-1.0)
//		ctrY[i] = boxesCopy.At(i, 1) + 0.5*(heights[i]-1.0)
//	}
//	pred := mat.DenseCopyOf(landmarkDeltas)
//	pred.Apply(func(_, i, j float64) float64 {
//		return landmarkDeltas.At(i, j)*widths[i] + ctrX[i]
//	}, pred)
//	pred.Apply(func(_, i, j float64) float64 {
//		return landmarkDeltas.At(i, j)*heights[i] + ctrY[i]
//	}, pred)
//	return pred
//}
