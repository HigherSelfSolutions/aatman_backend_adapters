package llm

import (
	"context"
	"io"
)

// MockAdapter is a mock implementation for testing
type MockAdapter struct {
	CompleteFunc       func(ctx context.Context, req CompletionRequest) (*CompletionResponse, error)
	StreamCompleteFunc func(ctx context.Context, req CompletionRequest) (io.ReadCloser, error)
	HealthCheckFunc    func(ctx context.Context) error
}

func (m *MockAdapter) Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	if m.CompleteFunc != nil {
		return m.CompleteFunc(ctx, req)
	}
	return &CompletionResponse{
		Content:      "mock response",
		TokensUsed:   100,
		FinishReason: "stop",
	}, nil
}

func (m *MockAdapter) StreamComplete(ctx context.Context, req CompletionRequest) (io.ReadCloser, error) {
	if m.StreamCompleteFunc != nil {
		return m.StreamCompleteFunc(ctx, req)
	}
	return io.NopCloser(nil), nil
}

func (m *MockAdapter) HealthCheck(ctx context.Context) error {
	if m.HealthCheckFunc != nil {
		return m.HealthCheckFunc(ctx)
	}
	return nil
}