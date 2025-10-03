package llm

import (
	"context"
	"fmt"
	"io"

	openai "github.com/sashabaranov/go-openai"
)

// OpenAIAdapter implements Adapter for OpenAI's API
type OpenAIAdapter struct {
	client *openai.Client
	model  string
}

// NewOpenAIAdapter creates a new OpenAI adapter
func NewOpenAIAdapter(apiKey, model string) *OpenAIAdapter {
	return &OpenAIAdapter{
		client: openai.NewClient(apiKey),
		model:  model,
	}
}

// Complete sends a completion request to OpenAI
func (a *OpenAIAdapter) Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	messages := a.buildMessages(req)

	resp, err := a.client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model:       a.model,
		Messages:    messages,
		MaxTokens:   req.MaxTokens,
		Temperature: float32(req.Temperature),
	})

	if err != nil {
		return nil, fmt.Errorf("openai api error: %w", err)
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("no completion choices returned")
	}

	return &CompletionResponse{
		Content:      resp.Choices[0].Message.Content,
		TokensUsed:   resp.Usage.TotalTokens,
		FinishReason: string(resp.Choices[0].FinishReason),
	}, nil
}

// StreamComplete sends a streaming completion request to OpenAI
func (a *OpenAIAdapter) StreamComplete(ctx context.Context, req CompletionRequest) (io.ReadCloser, error) {
	messages := a.buildMessages(req)

	stream, err := a.client.CreateChatCompletionStream(ctx, openai.ChatCompletionRequest{
		Model:       a.model,
		Messages:    messages,
		MaxTokens:   req.MaxTokens,
		Temperature: float32(req.Temperature),
		Stream:      true,
	})

	if err != nil {
		return nil, fmt.Errorf("openai stream error: %w", err)
	}

	return &streamReader{stream: stream}, nil
}

// HealthCheck verifies connectivity to OpenAI
func (a *OpenAIAdapter) HealthCheck(ctx context.Context) error {
	_, err := a.client.ListModels(ctx)
	if err != nil {
		return fmt.Errorf("openai health check failed: %w", err)
	}
	return nil
}

// buildMessages converts our message format to OpenAI's format
func (a *OpenAIAdapter) buildMessages(req CompletionRequest) []openai.ChatCompletionMessage {
	messages := make([]openai.ChatCompletionMessage, 0, len(req.Messages)+1)

	if req.SystemPrompt != "" {
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleSystem,
			Content: req.SystemPrompt,
		})
	}

	for _, msg := range req.Messages {
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}

	return messages
}

// streamReader wraps OpenAI's stream to implement io.ReadCloser
type streamReader struct {
	stream *openai.ChatCompletionStream
}

func (r *streamReader) Read(p []byte) (n int, err error) {
	response, err := r.stream.Recv()
	if err != nil {
		return 0, err
	}

	if len(response.Choices) == 0 {
		return 0, io.EOF
	}

	content := response.Choices[0].Delta.Content
	copy(p, content)
	return len(content), nil
}

func (r *streamReader) Close() error {
	r.stream.Close()
	return nil
}