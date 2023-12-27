package gotritron

import (
	"fmt"
	grpc_client "github.com/okieraised/gotritron/grpc-client"
	"github.com/okieraised/gotritron/opencv"
	"github.com/okieraised/gotritron/triton_client"
	"github.com/okieraised/gotritron/utils"
	"gorgonia.org/tensor"
	"image"
	"strings"
	"time"
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
	Config       *RetinaFaceDetectionConfig
	TritonClient *triton_client.TritonGRPCClient
	numAnchor    map[string]int
	anchorsFPN   map[string][][]float64
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
		Config:       DefaultRetinaFaceDetectionConfig(),
		TritonClient: triton_client.GetGRPCInstance(),
		numAnchor:    numAnchors,
		anchorsFPN:   anchorsFPN,
	}, nil
}

func (rfd *RetinaFaceDetection) Preprocess(src *opencv.Mat) (*tensor.Dense, float64, error) {

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
	opencv.Resize(*src, &dst, image.Pt(newWidth, newHeight), 0, 0, opencv.InterpolationDefault)
	err := src.Close()
	if err != nil {
		return nil, detScale, err
	}
	rawOpenCVBytes, err := dst.DataPtrUint8()
	if err != nil {
		return nil, detScale, err
	}
	rawFloat := make([]float32, len(rawOpenCVBytes))
	for i, val := range rawOpenCVBytes {
		rawFloat[i] = float32(val)
	}

	resizedImg := tensor.New(tensor.WithBacking(rawFloat), tensor.WithShape(dst.Rows(), dst.Cols(), 3))
	detImg := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(rfd.Config.ImageSize[1], rfd.Config.ImageSize[0], 3))
	subDetImg, err := detImg.Slice(tensor.S(0, newHeight), tensor.S(0, newWidth), nil)
	if err != nil {
		return nil, detScale, err
	}

	err = tensor.Copy(subDetImg, resizedImg)
	_ = dst.Close()

	return detImg, detScale, err
}

func (rfd *RetinaFaceDetection) Forward(src *tensor.Dense) ([]byte, error) {
	imTensor := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(1, 3, rfd.Config.ImageSize[0], rfd.Config.ImageSize[1]))

	for idx := 0; idx < 3; idx++ {
		subImTensor, err := imTensor.Slice(tensor.S(0), tensor.S(idx), nil, nil)
		if err != nil {
			return nil, err
		}

		subSrc, err := src.Slice(nil, nil, tensor.S(2-idx))
		if err != nil {
			return nil, err
		}

		divByPixelScale, err := subSrc.(*tensor.Dense).DivScalar(pixelScale, true)
		if err != nil {
			return nil, err
		}

		subByPixelMeans, err := divByPixelScale.SubScalar(pixelMean[2-idx], true)
		if err != nil {
			return nil, err
		}

		divByPixelStd, err := subByPixelMeans.DivScalar(pixelSTD[2-idx], true)
		if err != nil {
			return nil, err
		}

		err = tensor.Copy(subImTensor, divByPixelStd)
		if err != nil {
			return nil, err
		}
	}

	return imTensor.Raw, nil
}

func (rfd *RetinaFaceDetection) infer(rawInput []byte) error {
	modelConf, err := triton_client.GetGRPCInstance().GetModelConfiguration(
		rfd.Config.ModelName,
		"",
		10*time.Second)
	if err != nil {
		return err
	}

	dataType := ""
	dataTypes := strings.Split(modelConf.Config.Input[0].DataType.String(), "_")
	if len(dataTypes) == 2 {
		dataType = dataTypes[1]
	}

	inferInputs := []*grpc_client.ModelInferRequest_InferInputTensor{
		{
			Name:     modelConf.Config.Input[0].Name,
			Datatype: dataType,
			Shape:    modelConf.Config.Input[0].Dims,
		},
	}

	// run the inference models
	infer, err := triton_client.GetGRPCInstance().ModelGRPCInfer(inferInputs, nil,
		[][]byte{rawInput}, rfd.Config.ModelName, "", 10*time.Second)
	if err != nil {
		return err
	}

	netOuts := make([]*tensor.Dense, len(modelConf.Config.Output))
	for idx, out := range modelConf.Config.Output {
		dims := make([]int, len(out.Dims))
		for j, val := range out.Dims {
			dims[j] = int(val)
		}
		inferLens := utils.MultiplyAll(dims)
		res := make([]float32, inferLens)
		for i := 0; i < inferLens; i++ {
			res[i] = utils.BytesToFloat32(infer.RawOutputContents[idx][i*4 : i*4+4])
		}
		outTensor := tensor.New(tensor.WithBacking(res), tensor.WithShape(dims...))
		netOuts[idx] = outTensor
	}

	symIdx := 0
	for idx, s := range featStrideFPN {
		scores := netOuts[symIdx]
		subScores, err := scores.Slice(nil, tensor.S(rfd.numAnchor[fmt.Sprintf("stride%d", s)], scores.Shape()[1]), nil, nil)
		if err != nil {
			return err
		}
		if subScores == nil {
			return err
		}

		bboxDeltas := netOuts[symIdx+1]
		height := bboxDeltas.Shape()[2]
		width := bboxDeltas.Shape()[3]
		A := rfd.numAnchor[fmt.Sprintf("stride%d", s)]
		K := height * width
		anchorFPN := rfd.anchorsFPN[fmt.Sprintf("stride%d", s)]

		anchors, err := AnchorPlane(height, width, s, anchorFPN)
		if err != nil {
			return err
		}
		err = anchors.Reshape(K*A, 4)
		if err != nil {
			return err
		}

		newT, err := subScores.(*tensor.Dense).SafeT(0, 2, 3, 1)
		if err != nil {
			return err
		}
		//if idx == 0 {
		//	fmt.Println(newT.Shape())
		//}

		err = newT.Reshape(newT.Shape()[1]*newT.Shape()[2]*newT.Shape()[3], 1)
		if err != nil {
			return err
		}

		//if idx == 0 {
		//	fmt.Println(newT.Shape())
		//	fmt.Println(newT)
		//}

		tBboxDeltas, err := bboxDeltas.SafeT(0, 2, 3, 1)
		if err != nil {
			return err
		}

		//if idx == 0 {
		//	fmt.Println(tBboxDeltas.Shape())
		//	fmt.Println(tBboxDeltas)
		//}

		bboxPredLen := tBboxDeltas.Shape()[3] / A
		//if idx == 0 {
		//	fmt.Println(bboxPredLen)
		//}

		err = tBboxDeltas.Reshape(tBboxDeltas.Shape()[1]*tBboxDeltas.Shape()[2]*tBboxDeltas.Shape()[3]/bboxPredLen, bboxPredLen)
		if err != nil {
			return err
		}
		//if idx == 0 {
		//	fmt.Println(tBboxDeltas)
		//	fmt.Println(tBboxDeltas.Shape())
		//}
		//---------------------------
		tBboxDeltas04, err := tBboxDeltas.Slice(nil, tensor.S(0, tBboxDeltas.Shape()[1]+1, 4))
		mulTensor04, err := tBboxDeltas04.(*tensor.Dense).MulScalar(bBoxSTD[0], true)
		if err != nil {
			return err
		}
		err = tensor.Copy(tBboxDeltas04, mulTensor04)
		if err != nil {
			return err
		}

		//---------------------------
		tBboxDeltas14, err := tBboxDeltas.Slice(nil, tensor.S(1, tBboxDeltas.Shape()[1], 4))
		mulTensor14, err := tBboxDeltas14.(*tensor.Dense).MulScalar(bBoxSTD[1], true)
		if err != nil {
			return err
		}
		err = tensor.Copy(tBboxDeltas14, mulTensor14)
		if err != nil {
			return err
		}

		//---------------------------
		tBboxDeltas24, err := tBboxDeltas.Slice(nil, tensor.S(2, tBboxDeltas.Shape()[1], 4))
		mulTensor24, err := tBboxDeltas24.(*tensor.Dense).MulScalar(bBoxSTD[2], true)
		if err != nil {
			return err
		}
		err = tensor.Copy(tBboxDeltas24, mulTensor24)
		if err != nil {
			return err
		}

		//---------------------------
		tBboxDeltas34, err := tBboxDeltas.Slice(nil, tensor.S(3, tBboxDeltas.Shape()[1], 4))
		mulTensor34, err := tBboxDeltas34.(*tensor.Dense).MulScalar(bBoxSTD[3], true)
		if err != nil {
			return err
		}
		err = tensor.Copy(tBboxDeltas34, mulTensor34)
		if err != nil {
			return err
		}

		if idx == 0 {
			fmt.Println(tBboxDeltas)
			fmt.Println(tBboxDeltas.Shape())
		}

		_, err = rfd.bboxPred(anchors, tBboxDeltas)
		if err != nil {
			return err
		}

		if idx == -1 {
			return nil
		}
	}

	//fmt.Println("netOuts", netOuts)

	return nil
}

func (rfd *RetinaFaceDetection) bboxPred(boxes, boxDelta *tensor.Dense) (*tensor.Dense, error) {
	if boxes.Shape()[0] == 0 {
		return tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(0, boxDelta.Shape()[1])), nil
	}

	bRows2, err := boxes.Slice(nil, tensor.S(2))
	if err != nil {
		return nil, err
	}
	bRows0, err := boxes.Slice(nil, tensor.S(0))
	if err != nil {
		return nil, err
	}
	bSubRow, err := tensor.Sub(bRows2, bRows0)
	if err != nil {
		return nil, err
	}
	widths, err := tensor.Add(bSubRow, float32(1))
	if err != nil {
		return nil, err
	}

	fmt.Println(widths, widths.Shape())

	return nil, nil
}
