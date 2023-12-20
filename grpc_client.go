package gotritron

import (
	"context"
	grpc_client "github.com/okieraised/gotritron/grpc-client"
	"google.golang.org/grpc"
	"time"
)

var tritonGRPCClient *TritonGRPCClient

type TritonGRPCClient struct {
	serverURL   string
	grpcConn    *grpc.ClientConn
	grpcClient  grpc_client.GRPCInferenceServiceClient
	modelConfig interface{}
	modelName   string
}

// GetGRPCInstance returns an instance of the Triton gRPC client
func GetGRPCInstance() *TritonGRPCClient {
	return tritonGRPCClient
}

// NewTritonGRPCClient inits a new gRPC client
func NewTritonGRPCClient(serverURL string, grpcOpts []grpc.DialOption) error {
	grpcConn, err := grpc.Dial(serverURL, grpcOpts...)
	if err != nil {
		return err
	}
	grpcClient := grpc_client.NewGRPCInferenceServiceClient(grpcConn)
	tritonGRPCClient = &TritonGRPCClient{
		serverURL:  serverURL,
		grpcConn:   grpcConn,
		grpcClient: grpcClient,
	}
	return nil
}

// ServerAlive check server is alive.
func (tc *TritonGRPCClient) ServerAlive(timeout time.Duration) (bool, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	serverLiveResponse, err := tc.grpcClient.ServerLive(ctx, &grpc_client.ServerLiveRequest{})
	if err != nil {
		return false, err
	}
	return serverLiveResponse.Live, nil
}

// ServerReady check server is ready.
func (tc *TritonGRPCClient) ServerReady(timeout time.Duration) (bool, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	serverReadyResponse, err := tc.grpcClient.ServerReady(ctx, &grpc_client.ServerReadyRequest{})
	if err != nil {
		return false, err
	}
	return serverReadyResponse.Ready, nil
}

// ServerMetadata Get server metadata.
func (tc *TritonGRPCClient) ServerMetadata(timeout time.Duration) (*grpc_client.ServerMetadataResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	serverMetadataResponse, err := tc.grpcClient.ServerMetadata(ctx, &grpc_client.ServerMetadataRequest{})
	return serverMetadataResponse, err
}

// ModelRepositoryIndex Get model repo index.
func (tc *TritonGRPCClient) ModelRepositoryIndex(repoName string, isReady bool, timeout time.Duration) (*grpc_client.RepositoryIndexResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	repositoryIndexResponse, err := tc.grpcClient.RepositoryIndex(ctx, &grpc_client.RepositoryIndexRequest{RepositoryName: repoName, Ready: isReady})
	return repositoryIndexResponse, err
}

// GetModelConfiguration Get model configuration.
func (tc *TritonGRPCClient) GetModelConfiguration(modelName, modelVersion string, timeout time.Duration) (*grpc_client.ModelConfigResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	modelConfigResponse, err := tc.grpcClient.ModelConfig(ctx, &grpc_client.ModelConfigRequest{Name: modelName, Version: modelVersion})
	return modelConfigResponse, err
}

// ModelInferStats Get Model infer stats.
func (tc *TritonGRPCClient) ModelInferStats(modelName, modelVersion string, timeout time.Duration) (*grpc_client.ModelStatisticsResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	modelStatisticsResponse, err := tc.grpcClient.ModelStatistics(ctx, &grpc_client.ModelStatisticsRequest{Name: modelName, Version: modelVersion})
	return modelStatisticsResponse, err
}

// ModelLoadWithGRPC Load Model with grpc.
func (tc *TritonGRPCClient) ModelLoadWithGRPC(repoName, modelName string, modelConfigBody map[string]*grpc_client.ModelRepositoryParameter, timeout time.Duration) (*grpc_client.RepositoryModelLoadResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	loadResponse, err := tc.grpcClient.RepositoryModelLoad(ctx, &grpc_client.RepositoryModelLoadRequest{
		RepositoryName: repoName,
		ModelName:      modelName,
		Parameters:     modelConfigBody,
	})
	return loadResponse, err
}

// ModelUnloadWithGRPC Unload model with grpc modelConfigBody if not is nil.
func (tc *TritonGRPCClient) ModelUnloadWithGRPC(repoName, modelName string, modelConfigBody map[string]*grpc_client.ModelRepositoryParameter, timeout time.Duration) (*grpc_client.RepositoryModelUnloadResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	unloadResponse, err := tc.grpcClient.RepositoryModelUnload(ctx, &grpc_client.RepositoryModelUnloadRequest{
		RepositoryName: repoName,
		ModelName:      modelName,
		Parameters:     modelConfigBody,
	})
	return unloadResponse, err
}

// ShareMemoryStatus Get share memory / cuda memory status.
func (tc *TritonGRPCClient) ShareMemoryStatus(isCUDA bool, regionName string, timeout time.Duration) (interface{}, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	if isCUDA {
		cudaSharedMemoryStatusResponse, err := tc.grpcClient.CudaSharedMemoryStatus(ctx, &grpc_client.CudaSharedMemoryStatusRequest{Name: regionName})
		return cudaSharedMemoryStatusResponse, err
	}
	systemSharedMemoryStatusResponse, err := tc.grpcClient.SystemSharedMemoryStatus(ctx, &grpc_client.SystemSharedMemoryStatusRequest{Name: regionName})
	if err != nil {
		return nil, err
	}
	return systemSharedMemoryStatusResponse, nil
}

// ShareCUDAMemoryRegister cuda share memory register.
func (tc *TritonGRPCClient) ShareCUDAMemoryRegister(regionName string, cudaRawHandle []byte, cudaDeviceID int64, byteSize uint64, timeout time.Duration) (*grpc_client.CudaSharedMemoryRegisterResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	cudaSharedMemoryRegisterResponse, err := tc.grpcClient.CudaSharedMemoryRegister(
		ctx, &grpc_client.CudaSharedMemoryRegisterRequest{
			Name:      regionName,
			RawHandle: cudaRawHandle,
			DeviceId:  cudaDeviceID,
			ByteSize:  byteSize,
		},
	)
	return cudaSharedMemoryRegisterResponse, err
}

// ShareCUDAMemoryUnRegister cuda share memory unregister.
func (tc *TritonGRPCClient) ShareCUDAMemoryUnRegister(regionName string, timeout time.Duration) (*grpc_client.CudaSharedMemoryUnregisterResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	cudaSharedMemoryUnRegisterResponse, err := tc.grpcClient.CudaSharedMemoryUnregister(ctx, &grpc_client.CudaSharedMemoryUnregisterRequest{Name: regionName})
	return cudaSharedMemoryUnRegisterResponse, err
}

// ShareSystemMemoryRegister system share memory register.
func (tc *TritonGRPCClient) ShareSystemMemoryRegister(regionName, cpuMemRegionKey string, byteSize, cpuMemOffset uint64, timeout time.Duration) (*grpc_client.SystemSharedMemoryRegisterResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	systemSharedMemoryRegisterResponse, err := tc.grpcClient.SystemSharedMemoryRegister(
		ctx, &grpc_client.SystemSharedMemoryRegisterRequest{
			Name:     regionName,
			Key:      cpuMemRegionKey,
			Offset:   cpuMemOffset,
			ByteSize: byteSize,
		},
	)
	return systemSharedMemoryRegisterResponse, err
}

// ShareSystemMemoryUnRegister system share memory unregister.
func (tc *TritonGRPCClient) ShareSystemMemoryUnRegister(regionName string, timeout time.Duration) (*grpc_client.SystemSharedMemoryUnregisterResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	systemSharedMemoryUnRegisterResponse, err := tc.grpcClient.SystemSharedMemoryUnregister(ctx, &grpc_client.SystemSharedMemoryUnregisterRequest{Name: regionName})
	return systemSharedMemoryUnRegisterResponse, err
}

// GetModelTracingSetting get model tracing setting.
func (tc *TritonGRPCClient) GetModelTracingSetting(modelName string, timeout time.Duration) (*grpc_client.TraceSettingResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	traceSettingResponse, err := tc.grpcClient.TraceSetting(ctx, &grpc_client.TraceSettingRequest{ModelName: modelName})
	return traceSettingResponse, err
}

// SetModelTracingSetting set model tracing setting.
func (tc *TritonGRPCClient) SetModelTracingSetting(modelName string, settingMap map[string]*grpc_client.TraceSettingRequest_SettingValue, timeout time.Duration) (*grpc_client.TraceSettingResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	traceSettingResponse, err := tc.grpcClient.TraceSetting(ctx, &grpc_client.TraceSettingRequest{ModelName: modelName, Settings: settingMap})
	return traceSettingResponse, err
}

func (tc *TritonGRPCClient) Disconnect() error {
	err := tc.grpcConn.Close()
	return err
}

// ModelGRPCInfer Call Triton with GRPC
func (tc *TritonGRPCClient) ModelGRPCInfer(inferInputs []*grpc_client.ModelInferRequest_InferInputTensor, inferOutputs []*grpc_client.ModelInferRequest_InferRequestedOutputTensor, rawInputs [][]byte, modelName, modelVersion string, timeout time.Duration) (*grpc_client.ModelInferResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	// Create infer request for specific model/version.
	modelInferRequest := grpc_client.ModelInferRequest{
		ModelName:        modelName,
		ModelVersion:     modelVersion,
		Inputs:           inferInputs,
		Outputs:          inferOutputs,
		RawInputContents: rawInputs,
	}
	modelInferResponse, err := tc.grpcClient.ModelInfer(ctx, &modelInferRequest)
	if err != nil {
		return nil, err
	}
	return modelInferResponse, nil
}
