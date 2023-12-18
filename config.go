package gotritron

type ModelConfig struct {
}

const (
	FaceQualityClassBad = iota
	FaceQualityClassGood
	FaceQualityClassWearingMask
	FaceQualityClassWearingSunglasses
)

const (
	AgeEstimatorClassRange0_2 = iota
	AgeEstimatorClassRange3_9
	AgeEstimatorClassRange10_19
	AgeEstimatorClassRange20_29
	AgeEstimatorClassRange30_39
	AgeEstimatorClassRange40_49
	AgeEstimatorClassRange50_59
	AgeEstimatorClassRange60_69
	AgeEstimatorClassRange70_100
)

var FaceQualityClassMapper = map[int]string{
	FaceQualityClassBad:               "Bad",
	FaceQualityClassGood:              "Good",
	FaceQualityClassWearingMask:       "WearingMask",
	FaceQualityClassWearingSunglasses: "WearingSunGlasses",
}

type RetinaFaceDetectionConfig struct {
	// ModelName defines the name of the model to use
	ModelName string
	// Timeout defines duration in seconds
	Timeout             int64
	ImageSize           [2]int32
	MaxBatchSize        int32
	ConfidenceThreshold float32
	IOUThreshold        float32
}

func DefaultRetinaFaceDetectionConfig() *RetinaFaceDetectionConfig {
	return &RetinaFaceDetectionConfig{
		ModelName:           "face_detection_retina",
		Timeout:             20,
		ImageSize:           [2]int32{640, 640},
		MaxBatchSize:        1,
		ConfidenceThreshold: 0.7,
		IOUThreshold:        0.45,
	}
}

type VectorF32 [][]float32

type FaceAlignConfig struct {
	ImageSize         [2]int32
	StandardLandmarks VectorF32
}

func DefaultFaceAlignConfig() *FaceAlignConfig {
	return &FaceAlignConfig{
		ImageSize: [2]int32{640, 640},
		StandardLandmarks: VectorF32{
			{38.2946, 51.6963},
			{73.5318, 51.5014},
			{56.0252, 71.7366},
			{41.5493, 92.3655},
			{70.7299, 92.2041},
		},
	}
}

type ARCFaceRecognitionConfig struct {
	// ModelName defines the name of the model to use
	ModelName string
	// Timeout defines duration in seconds
	Timeout   int64
	ImageSize [2]int32
	BatchSize int32
}

func DefaultARCFaceRecognitionConfig() *ARCFaceRecognitionConfig {
	return &ARCFaceRecognitionConfig{
		ModelName: "face_identification",
		Timeout:   20,
		ImageSize: [2]int32{112, 112},
		BatchSize: 1,
	}
}

type FaceQualityConfig struct {
	// ModelName defines the name of the model to use
	ModelName string
	// Timeout defines duration in seconds
	Timeout   int64
	ImageSize [2]int32
	BatchSize int32
	Threshold float32
}

func DefaultFaceQualityConfig() *FaceQualityConfig {
	return &FaceQualityConfig{
		ModelName: "face_quality",
		Timeout:   20,
		ImageSize: [2]int32{112, 112},
		BatchSize: 1,
		Threshold: 0.5,
	}
}

type FaceSelectionConfig struct {
	MarginCenterLeftRatio   float32
	MarginCenterRightRatio  float32
	MarginEdgeRatio         float32
	MinimumFaceRatio        float32
	MinimumWidthHeightRatio float32
	MaximumWidthHeightRatio float32
}

func DefaultFaceSelectionConfig() *FaceSelectionConfig {
	return &FaceSelectionConfig{
		MarginCenterLeftRatio:   0.3,
		MarginCenterRightRatio:  0.3,
		MarginEdgeRatio:         0.1,
		MinimumFaceRatio:        0.0075,
		MinimumWidthHeightRatio: 0.65,
		MaximumWidthHeightRatio: 1.1,
	}
}

type AgeEstimatorConfig struct {
	// ModelName defines the name of the model to use
	ModelName string
	// Timeout defines duration in seconds
	Timeout   int64
	ImageSize [2]int32
	BatchSize int32
}

func DefaultAgeEstimatorConfig() *AgeEstimatorConfig {
	return &AgeEstimatorConfig{
		ModelName: "age_estimator",
		Timeout:   20,
		ImageSize: [2]int32{224, 224},
		BatchSize: 1,
	}
}
