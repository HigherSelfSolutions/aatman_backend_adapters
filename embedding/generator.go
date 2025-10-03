package embedding

import "context"

// Generator abstracts embedding providers (OpenAI, custom models)
type Generator interface {
	Generate(ctx context.Context, text string) ([]float64, error)
	GenerateBatch(ctx context.Context, texts []string) ([][]float64, error)
	Dimensions() int
}

// EmbeddingResponse represents an embedding result
type EmbeddingResponse struct {
	Embedding []float64
	TokensUsed int
}