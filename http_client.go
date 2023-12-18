package gotritron

import (
	"google.golang.org/grpc"
	"net/http"
	"time"
)

const (
	DefaultHTTPClientReadTimeout                = 5 * time.Second
	DefaultHTTPClientWriteTimeout               = 5 * time.Second
	DefaultHTTPClientMaxConnPerHost      int    = 16384
	HTTPPrefix                           string = "http://"
	JSONContentType                      string = "application/json"
	TritonAPIForModelVersionPrefix       string = "/versions/"
	TritonAPIPrefix                      string = "/v2"
	TritonAPIForServerIsLive                    = TritonAPIPrefix + "/health/live"
	TritonAPIForServerIsReady                   = TritonAPIPrefix + "/health/ready"
	TritonAPIForRepoIndex                       = TritonAPIPrefix + "/repository/index"
	TritonAPIForRepoModelPrefix                 = TritonAPIPrefix + "/repository/models/"
	TritonAPIForModelPrefix                     = TritonAPIPrefix + "/models/"
	TritonAPIForCudaMemoryRegionPrefix          = TritonAPIPrefix + "/cudasharememory/region/"
	TritonAPIForSystemMemoryRegionPrefix        = TritonAPIPrefix + "/systemsharememory/region/"
)

var tritonHTTPClient *TritonGRPCClient

type TritonHTTPClientService struct {
	serverURL   string
	httpConn    *grpc.ClientConn
	httpClient  *http.Client
	modelConfig interface{}
	modelName   interface{}
}

func GetHTTPInstance() *TritonGRPCClient {
	return tritonHTTPClient
}

func NewTritonHTTPClient(serverURL string, opts []grpc.DialOption) error {

	return nil
}
