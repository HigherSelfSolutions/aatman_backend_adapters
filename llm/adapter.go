package llm

import (
	"context"
	"io"
)

// Adapter abstracts LLM providers (OpenAI, Anthropic, custom models)
type Adapter interface {
	Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error)
	StreamComplete(ctx context.Context, req CompletionRequest) (io.ReadCloser, error)
	HealthCheck(ctx context.Context) error
}

// CompletionRequest represents a request to the LLM
type CompletionRequest struct {
	SystemPrompt string
	Messages     []Message
	MaxTokens    int
	Temperature  float64
}

// Message represents a single message in the conversation
type Message struct {
	Role    string // "user", "assistant", "system"
	Content string
}

// CompletionResponse represents the LLM's response
type CompletionResponse struct {
	Content      string
	TokensUsed   int
	FinishReason string
}