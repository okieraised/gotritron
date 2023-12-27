package triton_client

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
	"testing"
	"time"
)

func TestNewTritonGRPCClient(t *testing.T) {
	err := NewTritonGRPCClient(
		"210.211.99.18:8301",
		[]grpc.DialOption{
			grpc.WithTransportCredentials(insecure.NewCredentials()),
			grpc.WithKeepaliveParams(keepalive.ClientParameters{PermitWithoutStream: true}),
		},
	)
	assert.NoError(t, err)
	assert.NotNil(t, tritonGRPCClient.grpcClient)

	isAlive, err := tritonGRPCClient.ServerAlive(5 * time.Second)
	assert.NoError(t, err)
	assert.Equal(t, true, isAlive)

	isReady, err := tritonGRPCClient.ServerReady(5 * time.Second)
	assert.NoError(t, err)
	assert.Equal(t, true, isReady)

	meta, err := tritonGRPCClient.ServerMetadata(5 * time.Second)
	assert.NoError(t, err)
	fmt.Println(meta)

	index, err := tritonGRPCClient.ModelRepositoryIndex("", true, 5*time.Second)
	assert.NoError(t, err)
	fmt.Println(index)

	modelConf, err := tritonGRPCClient.GetModelConfiguration("face_detection_retina", "1", 5*time.Second)
	assert.NoError(t, err)
	fmt.Println(modelConf)
}

func TestTritonGRPCClient_ModelGRPCInfer(t *testing.T) {
	err := NewTritonGRPCClient(
		"210.211.99.18:8301",
		[]grpc.DialOption{
			grpc.WithTransportCredentials(insecure.NewCredentials()),
			grpc.WithKeepaliveParams(keepalive.ClientParameters{PermitWithoutStream: true}),
		},
	)
	assert.NoError(t, err)
	assert.NotNil(t, tritonGRPCClient.grpcClient)

	modelConf, err := tritonGRPCClient.GetModelConfiguration("face_detection_retina", "", 5*time.Second)
	assert.NoError(t, err)
	fmt.Println(modelConf.Config.Input)

	//rfd, err := gotritron.NewRetinaFaceDetection()
	//if err != nil {
	//	fmt.Println(err)
	//	return
	//}

	//f, err := os.Open("../test_data/harrison.jpeg")
	//if err != nil {
	//	fmt.Println(err)
	//	return
	//}
	//defer f.Close()
	//
	//content, err := io.ReadAll(f)
	//if err != nil {
	//	fmt.Println(err)
	//	return
	//}
	//
	//res, err := internal.ConvertImageToOpenCV(content)
	//if err != nil {
	//	fmt.Println(err)
	//	return
	//}
	//
	//processedImg, _, err := rfd.Preprocess(res)
	//if err != nil {
	//	fmt.Println(err)
	//	return
	//}
	//
	//input, err := rfd.Forward(processedImg)
	//if err != nil {
	//	fmt.Println(err)
	//	return
	//}
	//
	//fmt.Println(len(input))
	//
	//rawInputs := [][]byte{input}
	//
	//inferInputs := []*grpc_client.ModelInferRequest_InferInputTensor{
	//	{
	//		Name:     "data",
	//		Datatype: "FP32",
	//		Shape:    []int64{1, 3, 640, 640},
	//	},
	//}
	//
	//infer, err := GetGRPCInstance().ModelGRPCInfer(inferInputs, nil, rawInputs, "face_detection_retina", "1", 10*time.Second)
	//if err != nil {
	//	fmt.Println(err)
	//	return
	//}
	//
	//fmt.Println(infer.RawOutputContents)

}
