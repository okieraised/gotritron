package triton

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
		"127.0.0.1:8203",
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
