package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/HigherSelfSolutions/aatman_backend_adapters/embedding"
	"github.com/HigherSelfSolutions/aatman_backend_adapters/llm"
)

func main() {
	ctx := context.Background()
	apiKey := os.Getenv("OPENAI_API_KEY")

	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY environment variable not set")
	}

	// Example 1: LLM Completion
	fmt.Println("=== LLM Example ===")
	llmExample(ctx, apiKey)

	// Example 2: Embedding Generation
	fmt.Println("\n=== Embedding Example ===")
	embeddingExample(ctx, apiKey)
}

func llmExample(ctx context.Context, apiKey string) {
	adapter := llm.NewOpenAIAdapter(apiKey, "gpt-4o-mini")

	resp, err := adapter.Complete(ctx, llm.CompletionRequest{
		SystemPrompt: "You are a wise spiritual guide. Be concise.",
		Messages: []llm.Message{
			{Role: "user", Content: "How can I find inner peace?"},
		},
		MaxTokens:   200,
		Temperature: 0.7,
	})

	if err != nil {
		log.Fatalf("LLM error: %v", err)
	}

	fmt.Printf("Response: %s\n", resp.Content)
	fmt.Printf("Tokens used: %d\n", resp.TokensUsed)
}

func embeddingExample(ctx context.Context, apiKey string) {
	gen := embedding.NewOpenAIGenerator(apiKey, "text-embedding-3-small")

	emb, err := gen.Generate(ctx, "How can I find inner peace?")
	if err != nil {
		log.Fatalf("Embedding error: %v", err)
	}

	fmt.Printf("Embedding dimensions: %d\n", len(emb))
	fmt.Printf("First 5 values: %v\n", emb[:5])
}
