package gotritron

import (
	"fmt"
	"github.com/okieraised/gotritron/triton_client"
	"github.com/okieraised/gotritron/utils"
	"github.com/stretchr/testify/assert"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
	"io"
	"os"
	"testing"
)

func TestNewRetinaFaceDetection(t *testing.T) {
	err := triton_client.NewTritonGRPCClient(
		"210.211.99.18:8301",
		[]grpc.DialOption{
			grpc.WithTransportCredentials(insecure.NewCredentials()),
			grpc.WithKeepaliveParams(keepalive.ClientParameters{PermitWithoutStream: true}),
		},
	)
	assert.NoError(t, err)

	f, err := os.Open("./test_data/harrison.jpeg")
	assert.NoError(t, err)
	defer f.Close()

	content, err := io.ReadAll(f)
	assert.NoError(t, err)

	res, err := utils.ConvertImageToOpenCV(content)
	assert.NoError(t, err)

	rfd, err := NewRetinaFaceDetection()
	assert.NoError(t, err)

	preprocessed, _, err := rfd.Preprocess(res)
	assert.NoError(t, err)
	assert.NotNil(t, preprocessed)

	img, err := rfd.Forward(preprocessed)
	assert.NoError(t, err)
	assert.NotNil(t, img)

	err = rfd.infer(img)
	assert.NoError(t, err)
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

func Test2(t *testing.T) {
	baseAnchor := [][]float64{
		{-56, -56, 71, 71},
		{-24, -24, 39, 39},
	}

	flatten := []float32{}
	for row, _ := range baseAnchor {
		for col, _ := range baseAnchor[row] {
			flatten = append(flatten, float32(baseAnchor[row][col]))
		}
	}
	fmt.Println("flatten", flatten)

	AnchorPlane(20, 20, 16, baseAnchor)
}
