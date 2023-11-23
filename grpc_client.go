package triton

import (
	"context"
	grpc_client "ekyc-backend-common/clients/triton/grpc-client"
	"google.golang.org/grpc"
	"time"
)

var tritonGRPCClient *TritonGRPCClientService

type TritonGRPCClientService struct {
	serverURL   string
	grpcConn    *grpc.ClientConn
	grpcClient  grpc_client.GRPCInferenceServiceClient
	modelConfig interface{}
	modelName   interface{}
}

func GetGRPCInstance() *TritonGRPCClientService {
	return tritonGRPCClient
}

func NewTritonGRPCClient(serverURL string, opts []grpc.DialOption) error {
	grpcConn, err := grpc.Dial(serverURL, opts...)
	if err != nil {
		return err
	}
	grpcClient := grpc_client.NewGRPCInferenceServiceClient(grpcConn)
	tritonGRPCClient = &TritonGRPCClientService{
		serverURL:  serverURL,
		grpcConn:   grpcConn,
		grpcClient: grpcClient,
	}
	return nil
}

// ServerAlive check server is alive.
func (t *TritonGRPCClientService) ServerAlive(timeout time.Duration) (bool, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	serverLiveResponse, err := t.grpcClient.ServerLive(ctx, &grpc_client.ServerLiveRequest{})
	if err != nil {
		return false, err
	}
	return serverLiveResponse.Live, nil
}

// ServerReady check server is ready.
func (t *TritonGRPCClientService) ServerReady(timeout time.Duration) (bool, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	serverReadyResponse, err := t.grpcClient.ServerReady(ctx, &grpc_client.ServerReadyRequest{})
	if err != nil {
		return false, err
	}
	return serverReadyResponse.Ready, nil
}

// ServerMetadata Get server metadata.
func (t *TritonGRPCClientService) ServerMetadata(timeout time.Duration) (*grpc_client.ServerMetadataResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	serverMetadataResponse, err := t.grpcClient.ServerMetadata(ctx, &grpc_client.ServerMetadataRequest{})
	return serverMetadataResponse, err
}

// ModelRepositoryIndex Get model repo index.
func (t *TritonGRPCClientService) ModelRepositoryIndex(repoName string, isReady bool, timeout time.Duration) (*grpc_client.RepositoryIndexResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	repositoryIndexResponse, err := t.grpcClient.RepositoryIndex(ctx, &grpc_client.RepositoryIndexRequest{RepositoryName: repoName, Ready: isReady})
	return repositoryIndexResponse, err
}

// GetModelConfiguration Get model configuration.
func (t *TritonGRPCClientService) GetModelConfiguration(modelName, modelVersion string, timeout time.Duration) (*grpc_client.ModelConfigResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	modelConfigResponse, err := t.grpcClient.ModelConfig(ctx, &grpc_client.ModelConfigRequest{Name: modelName, Version: modelVersion})
	return modelConfigResponse, err
}

// ModelInferStats Get Model infer stats.
func (t *TritonGRPCClientService) ModelInferStats(modelName, modelVersion string, timeout time.Duration) (*grpc_client.ModelStatisticsResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	modelStatisticsResponse, err := t.grpcClient.ModelStatistics(ctx, &grpc_client.ModelStatisticsRequest{Name: modelName, Version: modelVersion})
	return modelStatisticsResponse, err
}

// ModelLoadWithGRPC Load Model with grpc.
func (t *TritonGRPCClientService) ModelLoadWithGRPC(repoName, modelName string, modelConfigBody map[string]*grpc_client.ModelRepositoryParameter, timeout time.Duration) (*grpc_client.RepositoryModelLoadResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	loadResponse, err := t.grpcClient.RepositoryModelLoad(ctx, &grpc_client.RepositoryModelLoadRequest{
		RepositoryName: repoName,
		ModelName:      modelName,
		Parameters:     modelConfigBody,
	})
	return loadResponse, err
}

// ModelUnloadWithGRPC Unload model with grpc modelConfigBody if not is nil.
func (t *TritonGRPCClientService) ModelUnloadWithGRPC(repoName, modelName string, modelConfigBody map[string]*grpc_client.ModelRepositoryParameter, timeout time.Duration) (*grpc_client.RepositoryModelUnloadResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	unloadResponse, err := t.grpcClient.RepositoryModelUnload(ctx, &grpc_client.RepositoryModelUnloadRequest{
		RepositoryName: repoName,
		ModelName:      modelName,
		Parameters:     modelConfigBody,
	})
	return unloadResponse, err
}

// ShareMemoryStatus Get share memory / cuda memory status.
func (t *TritonGRPCClientService) ShareMemoryStatus(isCUDA bool, regionName string, timeout time.Duration) (interface{}, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	if isCUDA {
		cudaSharedMemoryStatusResponse, err := t.grpcClient.CudaSharedMemoryStatus(ctx, &grpc_client.CudaSharedMemoryStatusRequest{Name: regionName})
		return cudaSharedMemoryStatusResponse, err
	}
	systemSharedMemoryStatusResponse, err := t.grpcClient.SystemSharedMemoryStatus(ctx, &grpc_client.SystemSharedMemoryStatusRequest{Name: regionName})
	if err != nil {
		return nil, err
	}
	return systemSharedMemoryStatusResponse, nil
}

// ShareCUDAMemoryRegister cuda share memory register.
func (t *TritonGRPCClientService) ShareCUDAMemoryRegister(regionName string, cudaRawHandle []byte, cudaDeviceID int64, byteSize uint64, timeout time.Duration) (*grpc_client.CudaSharedMemoryRegisterResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	cudaSharedMemoryRegisterResponse, err := t.grpcClient.CudaSharedMemoryRegister(
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
func (t *TritonGRPCClientService) ShareCUDAMemoryUnRegister(regionName string, timeout time.Duration) (*grpc_client.CudaSharedMemoryUnregisterResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	cudaSharedMemoryUnRegisterResponse, err := t.grpcClient.CudaSharedMemoryUnregister(ctx, &grpc_client.CudaSharedMemoryUnregisterRequest{Name: regionName})
	return cudaSharedMemoryUnRegisterResponse, err
}

// ShareSystemMemoryRegister system share memory register.
func (t *TritonGRPCClientService) ShareSystemMemoryRegister(regionName, cpuMemRegionKey string, byteSize, cpuMemOffset uint64, timeout time.Duration) (*grpc_client.SystemSharedMemoryRegisterResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	systemSharedMemoryRegisterResponse, err := t.grpcClient.SystemSharedMemoryRegister(
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
func (t *TritonGRPCClientService) ShareSystemMemoryUnRegister(regionName string, timeout time.Duration) (*grpc_client.SystemSharedMemoryUnregisterResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	systemSharedMemoryUnRegisterResponse, err := t.grpcClient.SystemSharedMemoryUnregister(ctx, &grpc_client.SystemSharedMemoryUnregisterRequest{Name: regionName})
	return systemSharedMemoryUnRegisterResponse, err
}

// GetModelTracingSetting get model tracing setting.
func (t *TritonGRPCClientService) GetModelTracingSetting(modelName string, timeout time.Duration) (*grpc_client.TraceSettingResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	traceSettingResponse, err := t.grpcClient.TraceSetting(ctx, &grpc_client.TraceSettingRequest{ModelName: modelName})
	return traceSettingResponse, err
}

// SetModelTracingSetting set model tracing setting.
func (t *TritonGRPCClientService) SetModelTracingSetting(modelName string, settingMap map[string]*grpc_client.TraceSettingRequest_SettingValue, timeout time.Duration) (*grpc_client.TraceSettingResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	traceSettingResponse, err := t.grpcClient.TraceSetting(ctx, &grpc_client.TraceSettingRequest{ModelName: modelName, Settings: settingMap})
	return traceSettingResponse, err
}

func (t *TritonGRPCClientService) Disconnect() error {
	err := t.grpcConn.Close()
	return err
}

// modelGRPCInfer Call Triton with GRPC
func (t *TritonGRPCClientService) modelGRPCInfer(inferInputs []*grpc_client.ModelInferRequest_InferInputTensor, inferOutputs []*grpc_client.ModelInferRequest_InferRequestedOutputTensor, rawInputs [][]byte, modelName, modelVersion string, timeout time.Duration) (*grpc_client.ModelInferResponse, error) {
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
	modelInferResponse, err := t.grpcClient.ModelInfer(ctx, &modelInferRequest)
	if err != nil {
		return nil, err
	}
	return modelInferResponse, nil
}
