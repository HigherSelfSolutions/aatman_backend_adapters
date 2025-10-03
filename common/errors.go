package common

import (
	"errors"
	"fmt"
)

var (
	// ErrRateLimited indicates the provider rate limit was hit
	ErrRateLimited = errors.New("rate limit exceeded")

	// ErrTimeout indicates the request timed out
	ErrTimeout = errors.New("request timeout")

	// ErrInvalidInput indicates invalid input was provided
	ErrInvalidInput = errors.New("invalid input")

	// ErrProviderUnavailable indicates the provider is down
	ErrProviderUnavailable = errors.New("provider unavailable")
)

// AdapterError wraps provider-specific errors
type AdapterError struct {
	Provider string
	Op       string
	Err      error
}

func (e *AdapterError) Error() string {
	return fmt.Sprintf("%s adapter %s: %v", e.Provider, e.Op, e.Err)
}

func (e *AdapterError) Unwrap() error {
	return e.Err
}

// NewAdapterError creates a new adapter error
func NewAdapterError(provider, op string, err error) *AdapterError {
	return &AdapterError{
		Provider: provider,
		Op:       op,
		Err:      err,
	}
}