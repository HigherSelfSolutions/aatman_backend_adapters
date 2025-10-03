package embedding

import (
	"context"
	"fmt"

	openai "github.com/sashabaranov/go-openai"
)

// OpenAIGenerator implements Generator for OpenAI embeddings
type OpenAIGenerator struct {
	client     *openai.Client
	model      string
	dimensions int
}

// NewOpenAIGenerator creates a new OpenAI embedding generator
func NewOpenAIGenerator(apiKey, model string) *OpenAIGenerator {
	dims := 1536 // text-embedding-3-small default
	if model == "text-embedding-3-large" {
		dims = 3072
	}

	return &OpenAIGenerator{
		client:     openai.NewClient(apiKey),
		model:      model,
		dimensions: dims,
	}
}

// Generate creates an embedding for a single text
func (g *OpenAIGenerator) Generate(ctx context.Context, text string) ([]float64, error) {
	resp, err := g.client.CreateEmbeddings(ctx, openai.EmbeddingRequest{
		Input: []string{text},
		Model: openai.EmbeddingModel(g.model),
	})

	if err != nil {
		return nil, fmt.Errorf("openai embedding error: %w", err)
	}

	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}

	// Convert []float32 to []float64
	return float32ToFloat64(resp.Data[0].Embedding), nil
}

// GenerateBatch creates embeddings for multiple texts
func (g *OpenAIGenerator) GenerateBatch(ctx context.Context, texts []string) ([][]float64, error) {
	if len(texts) == 0 {
		return [][]float64{}, nil
	}

	resp, err := g.client.CreateEmbeddings(ctx, openai.EmbeddingRequest{
		Input: texts,
		Model: openai.EmbeddingModel(g.model),
	})

	if err != nil {
		return nil, fmt.Errorf("openai batch embedding error: %w", err)
	}

	embeddings := make([][]float64, len(resp.Data))
	for i, data := range resp.Data {
		// Convert []float32 to []float64
		embeddings[i] = float32ToFloat64(data.Embedding)
	}

	return embeddings, nil
}

// Dimensions returns the embedding dimension size
func (g *OpenAIGenerator) Dimensions() int {
	return g.dimensions
}

// float32ToFloat64 converts []float32 to []float64
func float32ToFloat64(f32 []float32) []float64 {
	f64 := make([]float64, len(f32))
	for i, v := range f32 {
		f64[i] = float64(v)
	}
	return f64
}
