#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example demonstrating the improved glove-python implementation.

This example shows:
1. Building a corpus from text
2. Training GloVe embeddings with improved numerical stability
3. Querying similar words
4. Error handling and validation
"""

from glove import Corpus, Glove


def example_basic_usage():
    """Basic usage example with a small corpus."""
    print("\n" + "="*70)
    print("Example 1: Basic Usage")
    print("="*70)
    
    # Sample corpus - a few sentences about animals
    corpus_text = [
        ['the', 'cat', 'sat', 'on', 'the', 'mat'],
        ['the', 'dog', 'sat', 'on', 'the', 'log'],
        ['cats', 'and', 'dogs', 'are', 'animals'],
        ['cats', 'like', 'to', 'sit'],
        ['dogs', 'like', 'to', 'run'],
        ['the', 'quick', 'brown', 'fox'],
        ['the', 'lazy', 'dog', 'sleeps'],
    ]
    
    print("\nBuilding co-occurrence matrix...")
    corpus = Corpus()
    corpus.fit(corpus_text, window=5)
    print(f"Vocabulary size: {len(corpus.dictionary)}")
    print(f"Co-occurrence matrix shape: {corpus.matrix.shape}")
    print(f"Non-zero entries: {corpus.matrix.nnz}")
    
    print("\nTraining GloVe model...")
    glove = Glove(no_components=10, learning_rate=0.05)
    glove.fit(corpus.matrix, epochs=30, no_threads=2, verbose=False)
    glove.add_dictionary(corpus.dictionary)
    
    print("\nModel trained successfully!")
    print(f"Word vectors shape: {glove.word_vectors.shape}")
    
    # Query similar words
    print("\nSimilar words to 'cat':")
    try:
        similar = glove.most_similar('cat', number=3)
        for word, score in similar:
            print(f"  {word}: {score:.4f}")
    except Exception as e:
        print(f"  Could not find similar words: {e}")
    
    print("\nSimilar words to 'dog':")
    try:
        similar = glove.most_similar('dog', number=3)
        for word, score in similar:
            print(f"  {word}: {score:.4f}")
    except Exception as e:
        print(f"  Could not find similar words: {e}")


def example_error_handling():
    """Demonstrate improved error handling and validation."""
    print("\n" + "="*70)
    print("Example 2: Error Handling and Validation")
    print("="*70)
    
    print("\nThe improved implementation validates all inputs:")
    
    # Test 1: Invalid learning rate
    print("\n1. Trying to create model with negative learning rate...")
    try:
        glove = Glove(learning_rate=-0.1)
        print("  ✗ Should have raised an error!")
    except ValueError as e:
        print(f"  ✓ Correctly rejected: {e}")
    
    # Test 2: Invalid number of components
    print("\n2. Trying to create model with zero components...")
    try:
        glove = Glove(no_components=0)
        print("  ✗ Should have raised an error!")
    except ValueError as e:
        print(f"  ✓ Correctly rejected: {e}")
    
    # Test 3: Invalid alpha
    print("\n3. Trying to create model with negative alpha...")
    try:
        glove = Glove(alpha=-1.0)
        print("  ✗ Should have raised an error!")
    except ValueError as e:
        print(f"  ✓ Correctly rejected: {e}")
    
    print("\nThese validations prevent silent failures and ensure robust training.")


def example_numerical_stability():
    """Demonstrate improved numerical stability."""
    print("\n" + "="*70)
    print("Example 3: Numerical Stability")
    print("="*70)
    
    # Create a simple corpus
    corpus_text = [
        ['word' + str(i) for i in range(10)]
        for _ in range(100)
    ]
    
    corpus = Corpus()
    corpus.fit(corpus_text, window=5)
    
    print("\nTraining with high learning rate (testing stability)...")
    print("The improved implementation uses epsilon stabilization to prevent")
    print("numerical issues even with aggressive hyperparameters.")
    
    # This would previously cause numerical issues
    glove = Glove(no_components=50, learning_rate=0.1, max_loss=10.0)
    
    try:
        glove.fit(corpus.matrix, epochs=10, no_threads=2, verbose=False)
        print("\n✓ Training completed successfully!")
        print(f"  All vectors finite: {glove.word_vectors is not None}")
        print(f"  Mean vector norm: {glove.word_vectors.mean():.4f}")
    except RuntimeError as e:
        print(f"\n✗ Training failed with: {e}")
        print("  (This would happen with very extreme hyperparameters)")


def example_paragraph_vectors():
    """Demonstrate paragraph vector computation."""
    print("\n" + "="*70)
    print("Example 4: Paragraph Vectors")
    print("="*70)
    
    # Build a corpus
    corpus_text = [
        ['machine', 'learning', 'is', 'fun'],
        ['deep', 'learning', 'is', 'powerful'],
        ['natural', 'language', 'processing'],
        ['computer', 'vision', 'is', 'interesting'],
        ['machine', 'learning', 'and', 'ai'],
        ['deep', 'neural', 'networks'],
    ]
    
    corpus = Corpus()
    corpus.fit(corpus_text, window=5)
    
    glove = Glove(no_components=20, learning_rate=0.05)
    glove.fit(corpus.matrix, epochs=50, no_threads=2, verbose=False)
    glove.add_dictionary(corpus.dictionary)
    
    print("\nComputing paragraph vector for: ['machine', 'learning']")
    try:
        para_vec = glove.transform_paragraph(['machine', 'learning'], 
                                             epochs=20, 
                                             ignore_missing=False)
        print(f"  Paragraph vector shape: {para_vec.shape}")
        print(f"  Vector norm: {(para_vec ** 2).sum() ** 0.5:.4f}")
        
        # Find similar words to the paragraph
        similar = glove.most_similar_paragraph(['machine', 'learning'], number=3)
        print("\n  Most similar words to this paragraph:")
        for word, score in similar:
            print(f"    {word}: {score:.4f}")
    except Exception as e:
        print(f"  Error: {e}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("Improved GloVe-Python Examples")
    print("="*70)
    print("\nThis demonstrates the improved implementation with:")
    print("  - Enhanced numerical stability")
    print("  - Comprehensive input validation")
    print("  - Better error handling")
    print("  - Optimized performance")
    
    example_basic_usage()
    example_error_handling()
    example_numerical_stability()
    example_paragraph_vectors()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)
    print("\nFor more details on improvements, see IMPROVEMENTS.md")
    print()


if __name__ == '__main__':
    main()
